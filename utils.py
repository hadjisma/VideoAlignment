# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Util functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import math
import os
import time

from absl import flags
from absl import logging

from easydict import EasyDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import tensorflow.compat.v2 as tf
import yaml

from config import CONFIG

FLAGS = flags.FLAGS

def get_warmup_lr(lr, global_step, lr_params):
  """Returns learning rate during warm up phase."""
  if lr_params.NUM_WARMUP_STEPS > 0:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(lr_params.NUM_WARMUP_STEPS, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_lr = lr_params.INITIAL_LR * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    lr = (1.0 - is_warmup) * lr + is_warmup * warmup_lr
  return lr

def get_lr_fn(optimizer_config):
  """Returns function that provides current learning rate based on config."""
  lr_params = optimizer_config.LR
  if lr_params.DECAY_TYPE == 'exp_decay':
    lr_fn = lambda lr, global_step: tf.train.exponential_decay(
        lr,
        global_step,
        lr_params.EXP_DECAY_STEPS,
        lr_params.EXP_DECAY_RATE,
        staircase=True)()
  elif lr_params.DECAY_TYPE == 'fixed':
    lr_fn = lambda lr, global_step: lr_params.INITIAL_LR
  elif lr_params.DECAY_TYPE == 'poly':
    lr_fn = lambda lr, global_step: tf.train.polynomial_decay(
        lr,
        global_step,
        CONFIG.TRAIN.MAX_ITERS,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)
  else:
    raise ValueError('Learning rate decay type %s not supported. Only support'
                     'the following decay types: fixed, exp_decay', 'and poly.')

  return (lambda lr, global_step: get_warmup_lr(lr_fn(lr, global_step),
                                                global_step, lr_params))


def get_optimizer(optimizer_config, learning_rate):
  """Returns optimizer based on config and learning rate."""
  if optimizer_config.TYPE == 'AdamOptimizer':
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  elif optimizer_config.TYPE == 'MomentumOptimizer':
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
  else:
    raise ValueError('Optimizer %s not supported. Only support the following'
                     'optimizers: AdamOptimizer, MomentumOptimizer .')
  return opt


def get_lr_opt_global_step():
  """Intializes learning rate, optimizer and global step."""
  optimizer = get_optimizer(CONFIG.OPTIMIZER, CONFIG.OPTIMIZER.LR.INITIAL_LR)
  global_step = optimizer.iterations
  learning_rate = optimizer.learning_rate
  return learning_rate, optimizer, global_step


def restore_ckpt(logdir, **ckpt_objects):
  """Create and restore checkpoint (if one exists on the path)."""
  # Instantiate checkpoint and restore from any pre-existing checkpoint.
  # Since model is a dict we can insert multiple modular networks in this dict.
  checkpoint = tf.train.Checkpoint(**ckpt_objects)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=logdir,
      max_to_keep=10,
      keep_checkpoint_every_n_hours=1)
  if CONFIG.MODE == 'train':
    status = checkpoint.restore(ckpt_manager.latest_checkpoint)
  else:
    status = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
  return ckpt_manager, status, checkpoint


def to_dict(config):
  if isinstance(config, list):
    return [to_dict(c) for c in config]
  elif isinstance(config, EasyDict):
    return dict([(k, to_dict(v)) for k, v in config.items()])
  else:
    return config


def setup_train_dir(logdir):
  """Setups directory for training."""
  tf.io.gfile.makedirs(logdir)
  config_path = os.path.join(logdir, 'config.yml')
  if not os.path.exists(config_path):
    logging.info(
        'Using config from config.py as no config.yml file exists in '
        '%s', logdir)
    with  tf.io.gfile.GFile(config_path, 'w') as config_file:
      config = dict([(k, to_dict(v)) for k, v in CONFIG.items()])
      yaml.safe_dump(config, config_file, default_flow_style=False)
  else:
    logging.info('Using config from config.yml that exists in %s.', logdir)
    with tf.io.gfile.GFile(config_path, 'r') as config_file:
      config_dict = yaml.safe_load(config_file)
    CONFIG.update(config_dict)

  train_logs_dir = os.path.join(logdir, 'train_logs')
  if os.path.exists(train_logs_dir) and not FLAGS.force_train:
    raise ValueError('You might be overwriting a directory that already '
                     'has train_logs. Please provide a new logdir name in '
                     'config or pass --force_train while launching script.')
  tf.io.gfile.makedirs(train_logs_dir)

def get_context_steps(step):
  num_steps = CONFIG.DATA.NUM_CONTEXT_FRAMES
  stride = CONFIG.DATA.FRAME_STRIDE
  # We don't want to see the future.
  steps = np.arange(step - (num_steps - 1) * stride, step + stride, stride)
  return steps

def get_indices(curr_idx, num_steps, seq_len):
  steps = range(curr_idx, curr_idx + num_steps)
  single_steps = np.concatenate([get_context_steps(step) for step in steps])
  single_steps = np.maximum(0, single_steps)
  single_steps = np.minimum(seq_len, single_steps)
  return single_steps

def get_framewise_embeddings(model, data, batch, frames_per_batch=20, frame_labels=False):
  """ extract embedding for each frame """
  # get models
  cnn = model['cnn']
  emb = model['emb']
  # initialization
  seq_len = data['seq_lens'].numpy()[batch]
  video = data['frames'][batch][np.newaxis,:,:,:,:]
  video_labels = data['frame_labels'][batch][np.newaxis,:]
  num_sub_batches = int(math.ceil(float(seq_len)/frames_per_batch))
  labels = []
  embeddings = []
  feat_maps = []
  for i in range(num_sub_batches):
    # select frames to embed
    if (i + 1) * frames_per_batch > seq_len:
      num_steps = seq_len - i * frames_per_batch
    else:
      num_steps = frames_per_batch
    
    curr_idx = i * frames_per_batch
    # get correponding context frames
    idxes = get_indices(curr_idx, num_steps, seq_len)
    curr_data = tf.gather(video, idxes, axis=1)
    # extract cnn_features
    cnn_feats = cnn(curr_data)
    embs, f_maps = emb(cnn_feats, num_steps)
    embeddings.append(embs.numpy())
    feat_maps.append(f_maps)
    
  embeddings = np.concatenate(embeddings, axis=0)
  feat_maps = np.concatenate(feat_maps, axis=0)
  
  return embeddings, video_labels, feat_maps

class Stopwatch(object):
  """Simple timer for measuring elapsed time."""

  def __init__(self):
    self.reset()

  def elapsed(self):
    return time.time() - self.time

  def done(self, target_interval):
    return self.elapsed() >= target_interval

  def reset(self):
    self.time = time.time()


def set_learning_phase(f):
  """Sets the correct learning phase before calling function f."""
  def wrapper(*args, **kwargs):
    """Calls the function f after setting proper learning phase."""
    if 'training' not in kwargs:
      raise ValueError('Function called with set_learning_phase decorator which'
                       ' does not have training argument.')
    training = kwargs['training']
    if training:
      # Set learning_phase to True to use models in training mode.
      tf.keras.backend.set_learning_phase(1)
    else:
      # Set learning_phase to False to use models in inference mode.
      tf.keras.backend.set_learning_phase(0)
    return f(*args, **kwargs)
  return wrapper

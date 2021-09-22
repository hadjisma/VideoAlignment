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

"""Base class for defining training algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v2 as tf
from config import CONFIG
from d2tw.smoothDTW import compute_alignment_loss
import Networks

FLAGS = flags.FLAGS


def BuildModel():
  """ Build the backbone models used for training """
  BigModel = {}
  BigModel['cnn'] = Networks.BaseModel()
  BigModel['emb'] = Networks.ConvEmbedder()
  return BigModel


def getModelFeatures(model, data, training):
  """One pass through the model."""
  cnn = model['cnn']
  emb = model['emb']
  if training:
    num_steps = CONFIG.TRAIN.NUM_FRAMES
  else:
    num_steps = CONFIG.EVAL.NUM_FRAMES
  # going through 2D CNN
  cnn_feats = cnn(data['frames'])
  # going through 3D embedder
  embs, _ = emb(cnn_feats, num_steps)
  channels = embs.shape[-1]
  embs = tf.reshape(embs, [-1, num_steps, channels])
  embs_all = embs

  return embs_all, cnn_feats

def compute_loss(embs, global_step,
                 training, frame_labels=None, seq_labels=None):
  if training:
    batch_size = CONFIG.TRAIN.BATCH_SIZE
  else:
    batch_size = CONFIG.EVAL.BATCH_SIZE

  loss = compute_alignment_loss(
        embs,
        batch_size,
        alignment_type=CONFIG.AL_TYPE,
        loss_type=CONFIG.ALIGNMENT.LOSS_TYPE,
        similarity_type=CONFIG.ALIGNMENT.SIMILARITY_TYPE,
        label_smoothing=CONFIG.ALIGNMENT.LABEL_SMOOTHING,
        softning=CONFIG.ALIGNMENT.DTW_RELAXATION,
        gamma_s=CONFIG.ALIGNMENT.DTW_GAMMA_S,
        gamma_f=CONFIG.ALIGNMENT.DTW_GAMMA_F)

  return loss

def get_base_and_embedding_variables(model):
  """Gets list of trainable vars from model's base and embedding networks."""

  if (CONFIG.MODEL.TRAIN_BASE == 'train_all') or (CONFIG.MODEL.TRAIN_BASE == 'scratch'):
    variables = model['cnn'].variables
  elif CONFIG.MODEL.TRAIN_BASE == 'only_bn':
    variables = [x for x in model['cnn'].variables
                   if 'batch_norm' in x.name or 'bn' in x.name]
  elif CONFIG.MODEL.TRAIN_BASE == 'frozen':
    variables = []
  else:
    raise ValueError('train_base values supported right now: train_all, '
                       'only_bn or frozen.')
  if CONFIG.MODEL.TRAIN_EMBEDDING:
    variables += model['emb'].variables
  return variables

def compute_gradients(variables, loss, tape=None):
  """This is to be used in Eager mode when a GradientTape is available."""
  if tf.executing_eagerly():
    assert tape is not None
    gradients = tape.gradient(loss, variables)
  else:
    gradients = tf.gradients(loss, variables)
  return gradients

def apply_gradients(variables, optimizer, grads):
  """Functional style apply_grads for `tfe.defun`."""
  optimizer.apply_gradients(zip(grads, variables))


class Algorithm(tf.keras.Model):
  """Base class for defining algorithms."""
  def __init__(self, model=None, model_cls=None):
    """ define backbone model to train """
    super(Algorithm, self).__init__()
    self.model = BuildModel()

  def train_one_iter(self, data, global_step, optimizer):
    """ define steps for training one iteration """
    with tf.GradientTape() as tape:
      embs, cnn_feats = getModelFeatures(self.model, data, training=True)
      logits = embs
      loss = compute_loss(logits, global_step,
                               training=True, frame_labels=None,
                               seq_labels=None)

      # Add regularization losses.
      reg_loss = tf.reduce_mean(tf.stack(self.losses))
      tf.summary.scalar('reg_loss', reg_loss, step=global_step)
      loss += reg_loss
      strategy = tf.distribute.get_strategy()
      num_replicas = strategy.num_replicas_in_sync
      loss *= (1. / num_replicas)
      
      variables = get_base_and_embedding_variables(self.model)

      # compute and apply grdaients
      gradients = compute_gradients(variables, loss, tape)
      apply_gradients(variables, optimizer, gradients)
      return loss

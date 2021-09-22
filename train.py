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

"""Training code of SmoothDTW"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
#**********************************************************************
# including tf.v1 just to allow growth with limited GPU memory for now
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#**********************************************************************
from config import CONFIG
from datasets import create_dataset
from utils import get_lr_fn
from utils import get_lr_opt_global_step
from utils import restore_ckpt
from utils import setup_train_dir
from utils import Stopwatch
import algorithm as train_algo

flags.DEFINE_boolean(
    'force_train', False, 'Continue with training even when '
    'train_logs exist. Useful if one has to resume training. '
    'By default switched off to prevent overwriting existing '
    'experiments.')

FLAGS = flags.FLAGS

#%% Start training
def train():
  """Trains model."""
  
  # define path to log dir
  logdir = CONFIG.LOGDIR

  setup_train_dir(logdir)
  
  # Common code for multigpu and single gpu
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    # get training algorithm	  
    algo = train_algo.Algorithm()

    # Setup summary writer.
    summary_writer = tf.summary.create_file_writer(
        os.path.join(logdir, 'train_logs'), flush_millis=10000)
                    
    # setup learning_rate schedule, optimizer ...
    learning_rate, optimizer, global_step = get_lr_opt_global_step()
    ckpt_manager, status, _ = restore_ckpt(
        logdir=logdir, optimizer=optimizer, **algo.model)
    
    global_step_value = global_step.numpy()

    lr_fn = get_lr_fn(CONFIG.OPTIMIZER)

    # Setup Dataset Iterators.
    batch_size_per_replica = CONFIG.TRAIN.BATCH_SIZE
    total_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    # Setup train iterator
    train_ds = create_dataset(split='train',mode=CONFIG.MODE, batch_size=total_batch_size)
    train_iterator = strategy.make_dataset_iterator(train_ds)
    
    # define one training step
    def train_step(data):
      loss = algo.train_one_iter(data, global_step, optimizer)
      return loss
  
    # gathering loss across different GPUs    
    def dist_train(it):
      total_loss = strategy.reduce(
        tf.distribute.ReduceOp.SUM, strategy.experimental_run(train_step, it),
        axis=None)
      return total_loss
  
    dist_train = tf.function(dist_train)

    stopwatch = Stopwatch()


    try:
      while global_step_value < CONFIG.TRAIN.MAX_ITERS:
        with summary_writer.as_default():
          with tf.summary.record_if(
              global_step_value % CONFIG.LOGGING.REPORT_INTERVAL == 0):
            
            # training loss
            loss = dist_train(train_iterator)
            # Update learning rate based in lr_fn.
            learning_rate.assign(lr_fn(learning_rate, global_step))

            tf.summary.scalar('loss', loss, step=global_step)
            tf.summary.scalar('learning_rate', learning_rate, step=global_step)

            # Save checkpoint.
            if global_step_value % CONFIG.CHECKPOINT.SAVE_INTERVAL == 0:
              ckpt_manager.save()
              logging.info('Checkpoint saved at iter %d.', global_step_value)

            # Update global step.
            global_step_value = global_step.numpy()

            time_per_iter = stopwatch.elapsed()

            tf.summary.scalar(
                'timing/time_per_iter', time_per_iter, step=global_step)

            logging.info('Iter[{}/{}], {:.1f}s/iter, Loss: {:.3f}'.format(
                global_step_value, CONFIG.TRAIN.MAX_ITERS, time_per_iter,
                loss.numpy()))
            
            # Reset stopwatch after iter is complete.
            stopwatch.reset()

    except KeyboardInterrupt:
      logging.info('Caught keyboard interrupt. Saving model before quitting.')

    finally:
      # Save the final checkpoint.
      ckpt_manager.save()
      logging.info('Checkpoint saved at iter %d', global_step_value)


def main(_):
  tf.enable_v2_behavior()
  tf.keras.backend.set_learning_phase(1)

  train()

if __name__ == '__main__':
  app.run(main)

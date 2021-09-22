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

"""Datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf

from config import CONFIG
from dataset_splits import DATASETS
from preprocessors import preprocess_sequence

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_parallel_calls', 60, 'Number of parallel calls while'
                     'preprocessing data on CPU.')


def normalize_input(frame, new_max=1., new_min=0.0, old_max=255.0, old_min=0.0):
  x = tf.cast(frame, tf.float32)
  x = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
  return x


def preprocess_input(frames, augment=True):
  """Preprocesses raw frames and optionally performs data augmentation."""

  preprocessing_ranges = {
      preprocess_sequence.IMAGE_TO_FLOAT: (),
      preprocess_sequence.RESIZE: {
          'new_size': [CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE],
      },
      preprocess_sequence.CLIP: {
          'lower_limit': 0.0,
          'upper_limit': 1.0,
      },
      preprocess_sequence.NORMALIZE_MEAN_STDDEV: {
          'mean': 0.5,
          'stddev': 0.5,
      }
  }

  if augment:
    if CONFIG.AUGMENTATION.RANDOM_FLIP:
      preprocessing_ranges[preprocess_sequence.FLIP] = {
          'dim': 2,
          'probability': 0.5,
      }
    if CONFIG.AUGMENTATION.RANDOM_CROP:
      preprocessing_ranges[preprocess_sequence.RANDOM_CROP] = {
          'image_size': tf.shape(frames)[1:4],
          'min_scale': 0.8,
      }
    if CONFIG.AUGMENTATION.BRIGHTNESS:
      preprocessing_ranges[preprocess_sequence.BRIGHTNESS] = {
          'max_delta': CONFIG.AUGMENTATION.BRIGHTNESS_MAX_DELTA,
      }
    if CONFIG.AUGMENTATION.CONTRAST:
      preprocessing_ranges[preprocess_sequence.CONTRAST] = {
          'lower': CONFIG.AUGMENTATION.CONTRAST_LOWER,
          'upper': CONFIG.AUGMENTATION.CONTRAST_UPPER
      }
    if CONFIG.AUGMENTATION.HUE:
      preprocessing_ranges[preprocess_sequence.HUE] = {
          'max_delta': CONFIG.AUGMENTATION.HUE_MAX_DELTA,
      }
    if CONFIG.AUGMENTATION.SATURATION:
      preprocessing_ranges[preprocess_sequence.SATURATION] = {
          'lower': CONFIG.AUGMENTATION.SATURATION_LOWER,
          'upper': CONFIG.AUGMENTATION.SATURATION_UPPER
      }
  else:
    if CONFIG.AUGMENTATION.RANDOM_CROP:
      preprocessing_ranges[preprocess_sequence.CENTRAL_CROP] = {
          'image_size': tf.shape(frames)[1:3]
      }

  frames, = preprocess_sequence.preprocess_sequence(
      ((frames, preprocess_sequence.IMAGE),), preprocessing_ranges)

  return frames


def decode(serialized_example):
  """Decode serialized SequenceExample."""

  context_features = {
    'name': tf.io.FixedLenFeature([], dtype=tf.string),
    'len': tf.io.FixedLenFeature([], dtype=tf.int64),
  }

  seq_features = {}

  seq_features['video'] = tf.io.FixedLenSequenceFeature([], dtype=tf.string)
  
  if CONFIG.DATA.FRAME_LABELS:
    seq_features['frame_labels'] = tf.io.FixedLenSequenceFeature(
        [], dtype=tf.int64)
  
  # Extract features from serialized data.
  context_data, sequence_data = tf.io.parse_single_sequence_example(
      serialized=serialized_example,
      context_features=context_features,
      sequence_features=seq_features)
  
  name = tf.cast(context_data['name'], tf.string)
  seq_len = context_data['len']
  
  video = sequence_data.get('video', [])
  frame_labels = sequence_data.get('frame_labels', [])

  return video, frame_labels, seq_len, name

def get_steps(step):
  """Sample multiple context steps for a given step."""
  num_steps = CONFIG.DATA.NUM_CONTEXT_FRAMES
  stride = CONFIG.DATA.FRAME_STRIDE
  if num_steps < 1:
    raise ValueError('num_steps should be >= 1.')
  if stride < 1:
    raise ValueError('stride should be >= 1.')
  # 1) We don't want to encode information from the future.
  # 2) tf.range() start uses (num_steps-1) because we want to include current frame in context	
  # 3) tf.range() (limit = step+stride) to make sure limit is equal to current step always  
  steps = tf.range(step - (num_steps - 1) * stride, step + stride, stride)
  return steps


def sample_and_preprocess(video,
                          frame_labels,
                          seq_len,
                          name
                          ):
  """Samples frames and prepares them for training."""
  
  # STEP 0: DECIDE NUMBER OF FRAMES TO SAMPLE AND AUGMENTATION STRATEGY
  # ACCORDING TO MODE (i.e. train vs test/val)
  if CONFIG.MODE == 'train':
    augment = True
    offset = 1
    max_num_steps = CONFIG.TRAIN.NUM_FRAMES 
    sampling_strategy = CONFIG.DATA.SAMPLING_STRATEGY
    sample_all = False	  
    sample_all_stride = None
  else:
    sampling_strategy = CONFIG.DATA.SAMPLING_STRATEGY
    augment = False
    offset = 1  
    if sampling_strategy == 'all':
      sample_all = True
      sample_all_stride = 1
      max_num_steps = seq_len #400
    else:
      sample_all = False
      sample_all_stride = None	  
      max_num_steps = CONFIG.EVAL.NUM_FRAMES
  # choose number of steps to sample
  num_steps = max_num_steps
    
  # STEP1: SAMPLE STEPS AND GET THEIR CONTEXT FRAMES FOR THE EMBEDDER
  if sample_all:
    steps = tf.range(0, seq_len, sample_all_stride)
    chosen_steps = steps
  else:    
    if sampling_strategy == 'stride':
      num_steps = tf.cast(num_steps, tf.int64)
      stride = (seq_len/num_steps)      
      stride = tf.cast(stride, tf.int64)
      if stride <= 0:
        stride = tf.cast(CONFIG.DATA.STRIDE, tf.int64)
      # Offset can be set between 0 and maximum location from which we can get
      # total coverage of the video without having to pad.
      offset = tf.cast(offset, tf.int64)
      if offset is None:	
        offset = tf.random.uniform(
          (), 0, tf.maximum(tf.cast(1, tf.int64), seq_len  - stride * num_steps),
          dtype=tf.int64)
      # This handles sampling over shorter sequences by padding the last frame
      # many times. This is not ideal for the way alignment training batches are
      # created.
      cur_steps = tf.minimum(
          seq_len  - 1,
          tf.range(offset, offset + num_steps * stride + 1, stride))
      cur_steps = cur_steps[:num_steps]
      
    elif sampling_strategy == 'random':
      # Sample a random offset less than a provided max offset. Among all frames
      # higher than the chosen offset, randomly sample num_frames
      check1 = tf.debugging.assert_greater_equal(
          seq_len,
          tf.cast(CONFIG.DATA.RANDOM_OFFSET, tf.int64),
          message='Random offset is more than sequence length.')
      check2 = tf.less_equal(
          tf.cast(num_steps, tf.int64),
          seq_len - tf.cast(CONFIG.DATA.RANDOM_OFFSET, tf.int64),
      )

      def _sample_random():
        with tf.control_dependencies([tf.identity(check1.outputs[0])]):
          offset = CONFIG.DATA.RANDOM_OFFSET
          steps = tf.random.shuffle(tf.range(offset, seq_len))
          steps = tf.gather(steps, tf.range(0, num_steps))
          #steps = tf.gather(steps, tf.range(0, seq_len))
          #steps = tf.gather(steps, tf.random.uniform(shape=(num_steps,), minval=offset, maxval=seq_len, dtype=tf.int64))
          steps = tf.gather(steps,
                            tf.nn.top_k(steps, k=num_steps).indices[::-1])
          steps = steps[:num_steps]
          return steps

      def _sample_all():
        return tf.range(0, num_steps, dtype=tf.int64)

      cur_steps = tf.cond(check2, _sample_random, _sample_all)
    else:
      raise ValueError('Sampling strategy %s is unknown. Supported values are '
                       'stride, offset_uniform and all for now.' % sampling_strategy)
    
    # Get multiple context steps depending on config at selected steps.
    steps = tf.reshape(tf.map_fn(get_steps, cur_steps), [-1])
    # make sure that frame ID is never less than 1 or greater than (seq_len-1)	
    steps = tf.maximum(tf.cast(0, tf.int64), steps)
    steps = tf.minimum(seq_len - 1, steps)
    # Store chosen indices.
    chosen_steps = cur_steps
 
  # Select data based on steps/ 
  video = tf.gather(video, steps)
  
  if CONFIG.DATA.FRAME_LABELS:
    frame_labels = tf.gather(frame_labels, steps)
  
  # Decode the encoded JPEG images
  video = tf.map_fn(
      tf.image.decode_jpeg,
      video,
      parallel_iterations=FLAGS.num_parallel_calls,
      dtype=tf.uint8)
  # Take images in range [0, 255] and normalize to [0, 1]
  video = tf.map_fn(
      normalize_input,
      video,
      parallel_iterations=FLAGS.num_parallel_calls,
      dtype=tf.float32)
  # Perform data-augmentation and return images in range [-1, 1]
  video = preprocess_input(video, augment)
  

  if CONFIG.MODE == 'train':
    shape_all_steps = CONFIG.DATA.NUM_CONTEXT_FRAMES * max_num_steps # should be similar to shape of steps
    video.set_shape([shape_all_steps, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, 3])
  if CONFIG.MODE == 'train' and CONFIG.DATA.FRAME_LABELS:
    shape_all_steps = CONFIG.DATA.NUM_CONTEXT_FRAMES * max_num_steps # should be similar to shape of steps
    frame_labels.set_shape([shape_all_steps])
  
  return {
      'frames': video,
      'chosen_steps': chosen_steps,
      'seq_lens': seq_len,
      'frame_labels': frame_labels,
      'name': name,
      'num_steps':num_steps,
  }

def get_tfrecords(dataset, split, path, per_class=False):
  """Get TFRecord files based on dataset and split."""

  if per_class:
    path_to_tfrecords = os.path.join(path % dataset, '*%s*'%split)
    logging.info('Loading %s data from: %s', split, path_to_tfrecords)
    tfrecord_files = sorted(tf.io.gfile.glob(path_to_tfrecords))
  else:
    path_to_tfrecords = os.path.join(path % dataset,
                                     '%s_%s*' % (dataset, split))

    logging.info('Loading %s data from: %s', split, path_to_tfrecords)
    tfrecord_files = sorted(tf.io.gfile.glob(path_to_tfrecords))

  if not tfrecord_files:
    raise ValueError('No tfrecords found at path %s' % path_to_tfrecords)

  return tfrecord_files

def concatenate_datasets(datasets):
  """ fucntion to concatenate different datasets and merge them into one """
  # initialize big_dataset
  big_dataset = datasets[0]
  # iteratively concatenate all elements in datasets
  for d in range(1,len(datasets)):
    big_dataset = big_dataset.concatenate(datasets[d])
  
  return big_dataset	

def create_dataset(split, mode, batch_size):
  """Creates a single-class dataset iterator based on config and split."""
  per_class = CONFIG.DATA.PER_CLASS
  #mode = CONFIG.MODE
  datasets = []
  with tf.device('/cpu:0'):
    # Loop though dataset classes	  
    for dataset_name in CONFIG.DATASETS:
      # Get TFrecord file for this dataset/class		
      tfrecord_files = get_tfrecords(
          dataset_name, split, CONFIG.PATH_TO_TFRECORDS, per_class=per_class)
      # create a dataset for this class	  
      dataset = tf.data.TFRecordDataset(
          tfrecord_files, num_parallel_reads=FLAGS.num_parallel_calls)

      # num of sample in this dataset split   
      num_samples = DATASETS[dataset_name][split]
      if mode == 'train':	  
	    # suffle videos within this class
        dataset = dataset.shuffle(num_samples)
        # repeat dataset indefinitely (should use number of epochs here)		
        dataset = dataset.repeat()
	  
	  # organize dataset into batches of batch_size each
      dataset = dataset.batch(batch_size)
	  # add the batches from this class to the big dataset
      datasets.append(dataset)

    if mode == 'train':  
      # select one batch from each class
      dataset = tf.data.experimental.sample_from_datasets(datasets,
                                                        len(datasets) * [1.0])
    else:
	  # use all samples in datasets
      dataset = concatenate_datasets(datasets)
    
	# unbatching simply to treat each video separately during decoding
    dataset = dataset.unbatch()
    # decode dataset saved as tfRecord    
    dataset = dataset.map(decode,
                          num_parallel_calls=FLAGS.num_parallel_calls)
      
    # for each video in dataset sample needed number of frames and pre-process
    dataset = dataset.map(sample_and_preprocess,
                          num_parallel_calls=FLAGS.num_parallel_calls)

    # drop_remainder adds batch size in shape else first dim remains as None.
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch batches
    dataset = dataset.prefetch(1)

    return dataset

def create_one_epoch_dataset(dataset_name, split, path_to_tfrecords):
  """Creates a dataset iterator that gives one epoch of dataset."""
  batch_size = 1
  per_class = CONFIG.DATA.PER_CLASS
  
  with tf.device('/cpu:0'):
    tfrecord_files = get_tfrecords(dataset_name, split, path_to_tfrecords, per_class=per_class)
    dataset = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=FLAGS.num_parallel_calls)
    dataset = dataset.map(decode, num_parallel_calls=FLAGS.num_parallel_calls)

    dataset = dataset.map(sample_and_preprocess,
                          num_parallel_calls=FLAGS.num_parallel_calls)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    # Prefetch batches
    dataset = dataset.prefetch(1)

  return dataset

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

"""Configuration of an experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()
CONFIG.AL_TYPE = 'D2TW_consistency'
CONFIG.REVCONS = True
CONFIG.D2TW_NORM = True
# ******************************************************************************
# Experiment params
# ******************************************************************************
CONFIG.MODE = 'train'
# Directory for the experiment logs.
CONFIG.LOGDIR = '/path/to/logdir'
# Dataset for training alignment.
CONFIG.DATASETS = [
     'baseball_pitch',
     'baseball_swing',
     'bench_press',
     'bowl',
     'clean_and_jerk',
     'golf_swing',
     'jumping_jacks',
     'pushup',
     'pullup',
     'situp',
     'squat',
     'tennis_forehand',
     'tennis_serve',
]

# Path to tfrecords.
CONFIG.PATH_TO_TFRECORDS = '/path/to/all/tfrecords/%s_tfrecords/'
# Size of images/frames.
CONFIG.IMAGE_SIZE = 224  # For ResNet50
CONFIG.IMAGE_CHANNELS = 3
# ******************************************************************************
# Training params
# ******************************************************************************
# Number of training steps.
CONFIG.TRAIN = edict()
CONFIG.TRAIN.MAX_ITERS = 150000
# Number of samples in each batch.
CONFIG.TRAIN.BATCH_SIZE = 4
# Number of frames to use while training.
CONFIG.TRAIN.NUM_FRAMES = 20
CONFIG.TRAIN.SKIP_FRAMES = 1
# ******************************************************************************
# Eval params
# ******************************************************************************
CONFIG.EVAL = edict()
# Number of samples in each batch.
CONFIG.EVAL.BATCH_SIZE = 1
# Number of frames to use while evaluating. Only used to see loss in eval mode.
CONFIG.EVAL.NUM_FRAMES = 20

CONFIG.EVAL.FRAMES_PER_BATCH = 25
CONFIG.EVAL.KENDALLS_TAU_STRIDE = 5
CONFIG.EVAL.KENDALLS_TAU_DISTANCE = 'sqeuclidean'  # cosine, sqeuclidean
# ******************************************************************************
# Model params
# ******************************************************************************
CONFIG.MODEL = edict()

CONFIG.MODEL.EMBEDDER_TYPE = 'conv'

CONFIG.MODEL.BASE_MODEL = edict()
# Resnet50
CONFIG.MODEL.BASE_MODEL.NETWORK = 'Resnet50_pretrained' # ResNet50-V2
CONFIG.MODEL.BASE_MODEL.LAYER = 'conv4_block3_out'

# Select which layers to train.
# train_base defines how we want proceed with fine-tuning the base model.
# 'frozen' : Weights are fixed and batch_norm stats are also fixed.
# 'train_all': Everything is trained and batch norm stats are updated.
# 'only_bn': Only tune batch_norm variables and update batch norm stats.
CONFIG.MODEL.TRAIN_BASE = 'only_bn'
CONFIG.MODEL.TRAIN_EMBEDDING = True
CONFIG.MODEL.RESNET_PRETRAINED_WEIGHTS = '/path/to/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5'


CONFIG.MODEL.CONV_EMBEDDER_MODEL = edict()
# List of conv layers defined as (channels, kernel_size, activate).
CONFIG.MODEL.CONV_EMBEDDER_MODEL.CONV_LAYERS = [
    (512, 3, True),
    (512, 3, True),
]
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD = 'max_pool'
# List of fc layers defined as (channels, activate).
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_LAYERS = [
    (512, True),
    (512, True),
]
CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE = 128
CONFIG.MODEL.CONV_EMBEDDER_MODEL.L2_NORMALIZE = True
CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_RATE = 0.0
CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_SPATIAL = False
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_DROPOUT_RATE = 0.1
CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN = True

CONFIG.MODEL.L2_REG_WEIGHT = 0.00001

# ******************************************************************************
# Alignment params
# ******************************************************************************
CONFIG.ALIGNMENT = edict()
CONFIG.ALIGNMENT.LABEL_SMOOTHING = 0.1
CONFIG.ALIGNMENT.LOSS_TYPE = 'D2TW_consistency'
CONFIG.ALIGNMENT.NORMALIZE_INDICES = True
CONFIG.ALIGNMENT.SIMILARITY_TYPE = 'cosine'
CONFIG.ALIGNMENT.DTW_RELAXATION = 'dtw_prob'
CONFIG.ALIGNMENT.DTW_GAMMA_S = 0.1
CONFIG.ALIGNMENT.DTW_GAMMA_F = 0.1

# ******************************************************************************
# Optimizer params
# ******************************************************************************
CONFIG.OPTIMIZER = edict()
# Supported optimizers are: AdamOptimizer, MomentumOptimizer
CONFIG.OPTIMIZER.TYPE = 'AdamOptimizer'

CONFIG.OPTIMIZER.LR = edict()
# Initial learning rate for optimizer.
CONFIG.OPTIMIZER.LR.INITIAL_LR = 0.0001
# Learning rate decay strategy.
# Currently Supported strategies: fixed, exp_decay
CONFIG.OPTIMIZER.LR.DECAY_TYPE = 'fixed'
CONFIG.OPTIMIZER.LR.EXP_DECAY_RATE = 0.97
CONFIG.OPTIMIZER.LR.EXP_DECAY_STEPS = 1000
CONFIG.OPTIMIZER.LR.NUM_WARMUP_STEPS = 0

# ******************************************************************************
# Data params
# ******************************************************************************
CONFIG.DATA = edict()
CONFIG.DATA.NUM_PREFETCH_BATCHES = 1
CONFIG.DATA.RANDOM_OFFSET = 1
CONFIG.DATA.STRIDE = 15 # half the frame rate (assuming frame rate is 30fps)
CONFIG.DATA.SAMPLING_STRATEGY = 'random'  # random for training BUT all for testing
CONFIG.DATA.NUM_CONTEXT_FRAMES = 2  # number of frames that will be embedded jointly,(i.e. context frames)
CONFIG.DATA.FRAME_STRIDE = 15  # stride between context frames
CONFIG.DATA.FRAME_LABELS = False # True if framewise labels are available (PS: only used at test time)

CONFIG.DATA.PER_DATASET_FRACTION = 1.0
CONFIG.DATA.PER_CLASS = False
# stride of frames while embedding a video during evaluation.
CONFIG.DATA.SAMPLE_ALL_STRIDE = 1
# ******************************************************************************
# Augmentation params
# ******************************************************************************
CONFIG.AUGMENTATION = edict()
CONFIG.AUGMENTATION.RANDOM_FLIP = True
CONFIG.AUGMENTATION.RANDOM_CROP = False
CONFIG.AUGMENTATION.BRIGHTNESS = True
CONFIG.AUGMENTATION.BRIGHTNESS_MAX_DELTA = 32.0 / 255
CONFIG.AUGMENTATION.CONTRAST = True
CONFIG.AUGMENTATION.CONTRAST_LOWER = 0.5
CONFIG.AUGMENTATION.CONTRAST_UPPER = 1.5
CONFIG.AUGMENTATION.HUE = False
CONFIG.AUGMENTATION.HUE_MAX_DELTA = 0.2
CONFIG.AUGMENTATION.SATURATION = False
CONFIG.AUGMENTATION.SATURATION_LOWER = 0.5
CONFIG.AUGMENTATION.SATURATION_UPPER = 1.5

# ******************************************************************************
# Logging params
# ******************************************************************************
CONFIG.LOGGING = edict()
# Number of steps between summary logging.
CONFIG.LOGGING.REPORT_INTERVAL = 100

# ******************************************************************************
# Checkpointing params
# ******************************************************************************
CONFIG.CHECKPOINT = edict()
# Number of steps between consecutive checkpoints.
CONFIG.CHECKPOINT.SAVE_INTERVAL = 30000

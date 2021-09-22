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


"""Model Zoo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from keras_applications import resnet_v2

import tensorflow.compat.v2 as tf

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

from config import CONFIG

FLAGS = flags.FLAGS
layers = tf.keras.layers


def get_pretrained_ckpt(network):
  """Return path to pretrained ckpt."""
  pretrained_paths = {
      'Resnet50_pretrained': CONFIG.MODEL.RESNET_PRETRAINED_WEIGHTS,
  }
  ckpt = pretrained_paths.get(network, None)
  return ckpt

class BaseModel(tf.keras.Model):
  """CNN to extract features framewsie features."""
  def __init__(self):
    super(BaseModel, self).__init__()
    # define network parameters
    layer = CONFIG.MODEL.BASE_MODEL.LAYER
    network = CONFIG.MODEL.BASE_MODEL.NETWORK
    local_ckpt = get_pretrained_ckpt(network)
    # create the different layers of the network
    base_model = resnet_v2.ResNet50V2(include_top=False,
                                        weights=local_ckpt,
                                        pooling='max',
                                        backend=tf.keras.backend,
                                        layers=tf.keras.layers,
                                        models=tf.keras.models,
                                        utils=tf.keras.utils)
    self.base_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layer).output)

  # pass data through the network
  def call(self, inputs):
    # Reorganize video into frames such that they can be passed through 2D network.
    batch_size, num_steps, h, w, c = inputs.shape
    images = tf.reshape(inputs, [batch_size * num_steps, h, w, c])

    # If base model is frozen, then training is set to False
    training = (tf.keras.backend.learning_phase() and
                CONFIG.MODEL.TRAIN_BASE != 'frozen')
    x = self.base_model(images, training=training)
    _, h, w, c = x.shape
    x = tf.reshape(x, [batch_size, num_steps, h, w, c])

    return x


class ConvEmbedder(tf.keras.Model):
  """3D embedder network."""

  def __init__(self):
    """Passes convolutional features through  3Dembedding network."""
    super(ConvEmbedder, self).__init__()
    # define network parameters
    conv_params = CONFIG.MODEL.CONV_EMBEDDER_MODEL.CONV_LAYERS
    fc_params = CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_LAYERS
    use_bn = CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN
    l2_reg_weight = CONFIG.MODEL.L2_REG_WEIGHT
    embedding_size = CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE

    # create the different layers of the network
    conv_params = [(x[0], x[1], x[2]) for x in conv_params]
    fc_params = [(x[0], x[1]) for x in fc_params]
    conv_bn_activations = get_conv_bn_layers(conv_params, use_bn, conv_dims=3)
    self.conv_layers = conv_bn_activations[0]
    self.bn_layers = conv_bn_activations[1]
    self.activations = conv_bn_activations[2]

    self.fc_layers = get_fc_layers(fc_params)

    self.embedding_layer = layers.Dense(
        embedding_size,
        kernel_regularizer=regularizers.l2(l2_reg_weight),
        bias_regularizer=regularizers.l2(l2_reg_weight))
  
  # pass data through the network
  def call(self, x, num_frames):
    base_dropout_rate = CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_RATE
    fc_dropout_rate = CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_DROPOUT_RATE
    # stack frames together for 3D ConvEmbedder
    batch_size, total_num_steps, h, w, c = x.shape
    num_context = total_num_steps // num_frames
    	
    x = tf.reshape(x, [batch_size * num_frames, num_context, h, w, c])

    # Dropout on output tensor from base.
    if CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_SPATIAL:
      x = layers.SpatialDropout3D(base_dropout_rate)(x)
    else:
      x = layers.Dropout(base_dropout_rate)(x)

    # Pass through convolution layers
    for i, conv_layer in enumerate(self.conv_layers):
      x = conv_layer(x)
      if CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN:
        bn_layer = self.bn_layers[i]
        x = bn_layer(x)
      if self.activations[i]:
        x = self.activations[i](x)

    # Perform spatial pooling
    if CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD == 'max_pool':
      xx = layers.GlobalMaxPooling3D()(x)
    elif CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD == 'avg_pool':
      xx = layers.GlobalAveragePooling3D()(x)
    elif CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD == 'flatten':
      xx = layers.Flatten()(x)
    else:
      raise ValueError('Supported flatten methods: max_pool, avg_pool and '
                       'flatten.')

    # Pass through fully connected layers
    for fc_layer in self.fc_layers:
      xx = layers.Dropout(fc_dropout_rate)(xx)
      xx = fc_layer(xx)

    xx = self.embedding_layer(xx)

    if CONFIG.MODEL.CONV_EMBEDDER_MODEL.L2_NORMALIZE:
      xx = tf.nn.l2_normalize(xx, axis=-1)

    return xx, x

def get_conv_bn_layers(conv_params, use_bn, conv_dims=2):
  """Returns convolution and batch norm layers."""
  if conv_dims == 1:
    conv_layer = layers.Conv1D
  elif conv_dims == 2:
    conv_layer = layers.Conv2D
  elif conv_dims == 3:
    conv_layer = layers.Conv3D
  else:
    raise ValueError('Invalid number of conv_dims')
  l2_reg_weight = CONFIG.MODEL.L2_REG_WEIGHT

  conv_layers = []
  bn_layers = []
  activations = []
  for channels, kernel_size, activate in conv_params:
    if activate:
      activation = tf.nn.relu
    else:
      activation = None
    conv_layers.append(conv_layer(
        channels, kernel_size,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg_weight),
        bias_regularizer=regularizers.l2(l2_reg_weight),
        kernel_initializer='he_normal',
        ))
    if use_bn:
      bn_layers.append(layers.BatchNormalization())
    activations.append(activation)

  return conv_layers, bn_layers, activations

def get_fc_layers(fc_params):
  """Return fully connected layers."""
  l2_reg_weight = CONFIG.MODEL.L2_REG_WEIGHT
  fc_layers = []
  for channels, activate in fc_params:
    if activate:
      activation = tf.nn.relu
    else:
      activation = None
    fc_layers.append(
        layers.Dense(channels, activation=activation,
                     kernel_regularizer=regularizers.l2(l2_reg_weight),
                     bias_regularizer=regularizers.l2(l2_reg_weight)))
  return fc_layers

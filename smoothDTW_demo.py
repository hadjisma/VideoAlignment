# This code is shared under the 
# Attribution-NonCommercial-ShareAlike 4.0 International
# Please find the full license in the main directory under LICENSE.MD

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo for getting SmoothDTW-based alignment score
@author: isma.hadji
@email: isma.hadji@samsung.com
"""
from absl import app
import numpy as np
import tensorflow as tf
import random

def assign2Tensor(tensor,i,j,new_val):
  """ function to deal with tf.Tensors being non-assignable """
  # create mask
  mask = np.ones(tensor.shape, dtype=np.float32)
  # hack to assign a new value to tensor at position (i,j)
  mask[i,j] = 0
  mask = tf.convert_to_tensor(mask, dtype=tf.float32)
  tensor = (tensor*mask) + (new_val * (1-mask))
  return tensor

def minGamma(inputs,gamma=1):
    """ continuous relaxation of min defined in the D3TW paper"""
    if gamma == 0:
        minG = tf.reduce_min(inputs)
    else:
        # log-sum-exp stabilization trick
        zi = (-inputs / gamma)
        max_zi = tf.reduce_max(zi)
        log_sum_G = max_zi + tf.math.log(tf.reduce_sum(tf.math.exp(zi-max_zi)))
        minG = -gamma * log_sum_G
    return minG


def softDTW(embs1, embs2, distance_type='cosine', softning='dtw_prob', gamma_s=0.1, gamma_f=0.1, D2TW_NORM=True):
  """ function to obtain a soft (differentiable version of DTW) 
      embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
                                                   and D: dimensionality of of the embedding vector)
      
  """
  # first get a pairwise distance matrix
  if distance_type == 'cosine':
    dist = tf.matmul(embs1, embs2, transpose_b=True) 
  else:
    raise ValueError('distance_type %s not supported for now' % distance_type)
  # normalize distance column-wise
  if D2TW_NORM:
    dist = -tf.math.log(tf.nn.softmax(dist/gamma_f,axis=0)) # eq 8
  
  nrows, ncols = dist.shape
  # initialize soft-DTW table
  sdtw = tf.zeros((nrows+1,ncols+1), dtype=tf.float32)
  # obtain dtw table using min_gamma or softMin relaxation
  for i in range(0,nrows+1):
    for j in range(0,ncols+1):
      if (i==0) and (j==0):
        new_val = tf.cast(0.0, tf.float32)
        sdtw = assign2Tensor(sdtw,i,j,new_val)
      elif (i==0) and (j!=0):
        new_val = tf.float32.max
        sdtw = assign2Tensor(sdtw,i,j,new_val)
      elif (i!=0) and (j==0):
        new_val = tf.float32.max
        sdtw = assign2Tensor(sdtw,i,j,new_val)
      else:
        neighbors = tf.stack([sdtw[i,j-1], sdtw[i-1,j-1], sdtw[i-1,j]])
        if softning == 'dtw_minGamma':
          new_val = dist[i-1,j-1] + minGamma(neighbors, gamma_s) # eq 6
          sdtw = assign2Tensor(sdtw,i,j,new_val)
        elif softning == 'dtw_prob':
          probs = tf.nn.softmax((-neighbors)/gamma_s)
          new_val = dist[i-1,j-1] + (probs[0] * sdtw[i,j-1]) + (probs[1] * sdtw[i-1,j-1]) + (probs[2] * sdtw[i-1,j]) # eq 5
          sdtw = assign2Tensor(sdtw,i,j,new_val)
        elif softning == 'non-diff':
          new_val = dist[i-1,j-1] +  tf.reduce_min([sdtw[i,j-1], sdtw[i-1,j-1], sdtw[i-1,j]]) # non-differentiable version
          sdtw = assign2Tensor(sdtw,i,j,new_val)
        else:
          raise ValueError('only softning based on dtw_minGamma or dtw_prob supported for now.')
  return sdtw, dist

def main(_):
    """ run demo """
    np.random.seed(0)
    embs1 = np.random.rand(20,128).astype(np.float32)
    embs2 = np.random.rand(30,128).astype(np.float32)
    sdtw, _ = softDTW(embs1, embs2)
    print('sdtw score:%0.3f' % sdtw[-1,-1])
    
if __name__ == '__main__':
  app.run(main)

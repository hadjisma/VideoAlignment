# This code is shared under the 
# Attribution-NonCommercial-ShareAlike 4.0 International
# Please find the full license in the main directory under LICENSE.MD

"""
Definition of our SmoothDTW-based alignment loss
@author: isma.hadji
@email: isma.hadji@samsung.com
"""

"""definition of the loss"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf
from config import CONFIG

FLAGS = flags.FLAGS


def classification_loss(logits, labels, label_smoothing):
  """Classification loss """
  # stop gradients from labels
  labels = tf.stop_gradient(labels)
  cls_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
      y_true=labels, y_pred=logits, from_logits=True,
      label_smoothing=label_smoothing))
  return cls_loss


def DTW_loss(logits, loss_type, batch_size, beta=1):
  """ calculate loss function given DTW table """
  if 'D2TW' in loss_type:
    """ non-discriminative DTW loss """
    loss = tf.reduce_mean(logits)
  else:
    raise ValueError('%s is an unsupported DTW loss' % loss_type)
  return loss

def assign2Tensor(tensor,i,j,new_val):
  """ function to deal with tf.Tensors being non-assignable """
  # create mask
  mask = np.ones(tensor.shape, dtype=np.float32)
  # hack to assign a new value to tensor at position (i,j)
  mask[i,j] = 0
  mask = tf.convert_to_tensor(mask, dtype=tf.float32)
  tensor = (tensor*mask) + (new_val * (1-mask))
  return tensor

def smoothDTW(embs1, embs2, distance_type, softning, gamma_s, gamma_f):
  """ function to obtain a soft (differentiable version of DTW) """
  # first get a pairwise distance matrix
  if distance_type == 'cosine':
    dist = tf.matmul(embs1, embs2, transpose_b=True) 
  else:
    raise ValueError('distance_type %s not supported for now' % distance_type)
  
  # normalize distance column-wise
  if CONFIG.D2TW_NORM:
    dist = -tf.math.log(tf.nn.softmax(dist/gamma_f,axis=0))
  
  nrows, ncols = dist.shape
  
  # calculate soft-DTW table
  sdtw = tf.zeros((nrows+1,ncols+1), dtype=tf.float32)
  # obtain dtw table using min_gamma or prob relaxation
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
          new_val = dist[i-1,j-1] + minGamma(neighbors, gamma_s)
          sdtw = assign2Tensor(sdtw,i,j,new_val)
        elif softning == 'dtw_prob':
          probs = tf.nn.softmax((-neighbors)/gamma_s)
          
          new_val = dist[i-1,j-1] + (probs[0] * sdtw[i,j-1]) + (probs[1] * sdtw[i-1,j-1]) + (probs[2] * sdtw[i-1,j])
          sdtw = assign2Tensor(sdtw,i,j,new_val)
        elif softning == 'non-diff':
          new_val = dist[i-1,j-1] +  tf.reduce_min([sdtw[i,j-1], sdtw[i-1,j-1], sdtw[i-1,j]])
          sdtw = assign2Tensor(sdtw,i,j,new_val)
        else:
          raise ValueError('only softning based on dtw_minGamma or dtw_prob supported for now.')
  return sdtw, dist

def minGamma(inputs,gamma=1):
    """ continuous relaxation of min defined in the D3TW paper"""
    if gamma == 0:
        minG = tf.reduce_min(inputs)
    else:
        # log-sum-exp stabilization trick
        zi = (-inputs / gamma)
        max_zi = tf.reduce_max(zi)
        log_sum_G = max_zi + tf.math.log(tf.reduce_sum(tf.math.exp(zi-max_zi))) #+ 1e-10)
        minG = -gamma * log_sum_G
    return minG


def compute_dtw_alignment_loss(embs,
                               batch_size,
                               loss_type,
                               distance_type,
                               softning,
                               gamma_s,
                               gamma_f
                               ):
  """Compute d2tw loss for all steps in each sequence.
  Args:
    embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
      batch size, T is the number of timesteps in the sequence, D is the size
      of the embeddings.
    loss_type: define the loss type used in our dtw alignment
    distance_type: String, Currently supported distance metrics: 'cosine'
    softning: relaxation used for dtw. currently supported: 'dtw_minGamma' and 'dtw_prob'
  Returns:
    loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
        dtw loss.
  """
  
  logits_list = []

  i = 0
  for j in range(i+1, batch_size):
    logits, _ = smoothDTW(embs[i], embs[j], distance_type, softning, gamma_s, gamma_f)
    logits_list.append(logits[-1,-1])
  
  logits = tf.stack(logits_list, axis=0)
  # calculate the loss
  loss = DTW_loss(logits, loss_type, batch_size)
  return loss


def compute_dtw_alignment_consistency_loss(embs,
                               batch_size,
                               loss_type,
                               distance_type,
                               softning,
                               gamma_s,
                               gamma_f,
                               label_smoothing
                               ):
  """Compute d2tw loss with Global Cycle Consistency for all steps in each sequence.
  Args:
    embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
      batch size, T is the number of timesteps in the sequence, D is the size
      of the embeddings.
    loss_type: define the loss type used in our dtw alignment
    distance_type: String, Currently supported distance metrics: 'cosine'
    softning: relaxation used for dtw. currently supported: 'dtw_minGamma' and 'dtw_prob'
  Returns:
    loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
        dtw loss.
  """
  
  logits_list = []
  logits_ij_list = []
  logits_ji_list = []
  labels_list = []
  
  i = 0
  if CONFIG.MODE == 'train':
    skip = CONFIG.TRAIN.SKIP_FRAMES 
  else:
    skip = 1
    
  for j in range(i+1, batch_size):
    logits_ij, _ = smoothDTW(embs[i,::skip,:], embs[j], distance_type, softning, gamma_s, gamma_f)
    logits_ij_list.append(logits_ij[-1,-1])
    logits_ij = tf.nn.softmax(-logits_ij[1:,1:],axis=0)
    logits_ji, _ = smoothDTW(embs[j], embs[i,::skip,:], distance_type, softning, gamma_s, gamma_f)
    logits_ji_list.append(logits_ji[-1,-1])
    logits_ji = tf.nn.softmax(-logits_ji[1:,1:],axis=0)
    if CONFIG.REVCONS:
      logits = tf.matmul(logits_ij, logits_ji)
      # transpose to make sure that the each row sums to 1 (to use categorical cross entropy loss that reads tensors by rows)
      logits = tf.transpose(logits)
      logits_list.append(logits)
      labels = tf.eye(logits.shape[0])
      labels_list.append(labels)
    
  if CONFIG.REVCONS:
    logits = tf.concat(logits_list, axis=0)
    labels = tf.concat(labels_list, axis=0)
  
  logits_ij_list = tf.stack(logits_ij_list, axis=0)
  logits_ji_list = tf.stack(logits_ji_list, axis=0)

  # calculate the loss
  loss_sdtw_ij = DTW_loss(logits_ij_list, loss_type, batch_size)
  loss_sdtw_ji = DTW_loss(logits_ji_list, loss_type, batch_size)
  
  if CONFIG.REVCONS:
    loss_con = classification_loss(logits, labels, label_smoothing)
    loss = loss_con + 0.1*loss_sdtw_ij + 0.1*loss_sdtw_ji
  else:
    loss = 0.1*loss_sdtw_ij + 0.1*loss_sdtw_ji
  return loss

#%% COMPUTE ALIGNMENT LOSS
def compute_alignment_loss(embs,
                           batch_size,
                           alignment_type='D2TW_consistency',
                           loss_type='D2TW_consistency',
                           similarity_type='cosine',
                           label_smoothing=0.1,
                           softning='dtw_prob',
                           gamma_s=0.1,
                           gamma_f=0.1):
  """Computes DTW alignment loss between sequences of embeddings."""

  if alignment_type == 'D2TW':
    loss = compute_dtw_alignment_loss(embs=embs,
                               batch_size=batch_size,
                               loss_type=loss_type,
                               distance_type=similarity_type,
                               softning=softning,
                               gamma_s=gamma_s,
                               gamma_f=gamma_f
                               )
  elif alignment_type == 'D2TW_consistency':
    loss = compute_dtw_alignment_consistency_loss(embs=embs,
                               batch_size=batch_size,
                               loss_type=loss_type,
                               distance_type=similarity_type,
                               softning=softning,
                               gamma_s=gamma_s,
                               gamma_f=gamma_f,
                               label_smoothing=label_smoothing
                               )
  else:
    print('CHOSEN ALIGNMENT TYPE IS NOT DEFINED')
  return loss

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow.compat.v2 as tf
#**********************************************************************
# including tf.v1 just to allow growth with limited GPU memory for now
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#**********************************************************************

from absl import app
from absl import flags
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

import algorithm
from utils import get_lr_opt_global_step
from utils import restore_ckpt
from utils import get_framewise_embeddings
from datasets import create_one_epoch_dataset
from config import CONFIG

flags.DEFINE_string('logdir', None, 'logdir name.')
flags.DEFINE_string('classname', None, 'class name.')
FLAGS = flags.FLAGS

def extract_embeddings(data_split, dataset_name, model):
    """ extract frame-wise embeddings"""
    # get dataset frames
    print(dataset_name)
    train_ds = create_one_epoch_dataset(dataset_name, split=data_split, path_to_tfrecords=CONFIG.PATH_TO_TFRECORDS)
    
    embs = []
    feat_maps = []
    steps = []
    videos = []
    names = []
    # loop through dataset
    for data in train_ds:
        step = data['chosen_steps']
        vid = np.squeeze(data['frames'])
        vid_name = data['name'][0].numpy().decode('utf8')
        print(vid_name)
        # extract embeddings
        emb, _, f_map = get_framewise_embeddings(model, data, 0, frames_per_batch=25, frame_labels=False)
        embs.append(emb)
        feat_maps.append(f_map[:,0,:,:,:])
        steps.append(step)
        videos.append(vid)
        names.append(vid_name)
        del emb, f_map
    
    return embs, steps, videos, names , feat_maps


def evaluate_KT(data, model_dir, metric, stride=5):
    """ function to evaluate framewise matches """
    
    # get training algorithm
    algo = algorithm.Algorithm()
    # restore the latest checkpoint of the trained model
    _, optimizer, _ = get_lr_opt_global_step()
    ckpt_manager, status, chekpoint = restore_ckpt(
        logdir=model_dir, optimizer=optimizer, **algo.model)
    if status.assert_existing_objects_matched():
        print(ckpt_manager.latest_checkpoint)
        #input('CHEKPOINT RESTORED')
    # get model
    model = algo.model
    
    # initialize
    taus_all = np.zeros((len(CONFIG.DATASETS),), dtype=np.float32)
    count = 0
    for dataset_name in CONFIG.DATASETS:
        # extract emneddings
        embs_list, _, _, _, _ = extract_embeddings(data, dataset_name, model)
        num_seqs = len(embs_list)
        # get kendall's tau for dataset_name
        print(num_seqs)
        taus = np.zeros((num_seqs * (num_seqs - 1)))
        taus_cosine = np.zeros((num_seqs * (num_seqs - 1)))
        idx = 0
        for i in range(num_seqs):
            query_feats = embs_list[i][::stride]
            print(query_feats.shape)
            for j in range(num_seqs):
                if i == j:
                     continue
                candidate_feats = embs_list[j][::stride]
                dists = cdist(query_feats, candidate_feats, metric) #CONFIG.EVAL.KENDALLS_TAU_DISTANCE)
                dists_cosine = cdist(query_feats, candidate_feats, 'cosine') #CONFIG.EVAL.KENDALLS_TAU_DISTANCE)
                nns = np.argmin(dists, axis=1)
                nns_cosine = np.argmin(dists_cosine, axis=1)
                taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
                taus_cosine[idx] = kendalltau(np.arange(len(nns)), nns_cosine).correlation
                print('tau=%0.5f/ %0.5f' % (taus[idx], taus_cosine[idx]))
                
                idx += 1
        taus = taus[~np.isnan(taus)]
        taus_cosine = taus_cosine[~np.isnan(taus_cosine)]
        print(dataset_name, taus.mean(), taus_cosine.mean())
        del embs_list
        taus_all[count] = taus.mean()
        count += 1
    # get kendall's tau for all visited classes
    tau = taus_all.mean()
    print(taus_all)
    print('datatset-wise tau=%0.5f' % tau)


            
def main(_):
    tf.enable_v2_behavior()
    tf.keras.backend.set_learning_phase(0)
    CONFIG.MODE = 'val'
    CONFIG.DATA.SAMPLING_STRATEGY = 'all'
    model_dir = FLAGS.logdir
    data_split = 'val'
    evaluate_KT(data_split, model_dir, metric='sqeuclidean')

  
if __name__ == '__main__':
    app.run(main)



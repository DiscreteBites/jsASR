# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 01:07:00 2019

@author: js2251
"""
from typing import cast

import numpy as np

import tensorflow as tf
from keras.models import Model

from .DataGenTimitTri import DataGeneratorTri

def predictPhonemeProbabilitiesNonCausalNN( filename_X: str, filename_Y: str, model_name = 'srA2_d_14', data_split_factor = 0.9):
    ''' use data_split_factor = 0 for running the model on all data or evaluating a test set, and 0.9for evaluating a train+dev set '''  

    X = np.load(filename_X)
    Y = np.load(filename_Y)
    Y = Y[50:]   # 50 timesteps in first causal NN

    # note that where Y == -1 no phoneme label is given "silence"
    # predict all including first after phoneme boundary
    idx = np.flatnonzero(Y > -1)
    
    batch_size   = 1024
    num_features = X.shape[1]
    num_timesteps = 305
    dim          = ( batch_size, num_timesteps, num_features )
    
    data_split = int( data_split_factor * len(idx))
    idx_val   = idx[data_split:]
    
    model = cast(Model, tf.keras.models.load_model( model_name + '.keras'))
    
    predict_generator = DataGeneratorTri(idx_val, X, Y, dim, reduce_factor = 1,non_causal_steps = int(num_timesteps/2), shuffle = False)
    evaluation = model.evaluate(predict_generator, verbose="1", return_dict=True)
    p = model.predict(predict_generator, verbose="1")
    
    p_prev = p[0]
    p_now = p[1]
    p_next = p[2]
    np.save( model_name + '_p_prev.npy',p_prev)
    np.save( model_name + '_p_now.npy',p_now)
    np.save( model_name + '_p_next.npy',p_next)
    
    idx_valed = np.array(idx_val)[np.array(range(len(p_now)))] + num_timesteps - int(num_timesteps/2)
    
    y_pred = np.argmax(p_now,axis=1)
    y_true = Y[idx_valed]
    pred_correct = sum( ( np.logical_or( y_pred[:-8] == y_true[:-8], y_pred[8:] == y_true[:-8]  ) ) ) / (len(y_pred)-8)
    np.save( model_name + '_Phonemes39_pred.npy', y_pred)
    np.save( model_name + '_Phonemes39_true.npy',y_true)
    
    print(evaluation[4:])
    print(pred_correct)
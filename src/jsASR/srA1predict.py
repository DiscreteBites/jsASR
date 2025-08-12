# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:29:18 2019

@author: js2251
"""
from typing import cast

import numpy as np

import tensorflow as tf
Model = tf.keras.Model

from .DataGenTimit import DataGenerator

def predictPhonemeProbabilitiesCausalNN(
    filename_X: str, filename_Y: str, model_name: str, 
    file_identifier_out: str, data_split_factor = 0.9 
):
    ''' data_split_factor = 0 to predict and store for all timesteps. set to 0.9 for evaluation of development set (and use X of train+dev set)
        see also srA1eval script for evaluation purposes '''
    X = np.load( filename_X )
    Y = np.load( filename_Y )
    
    # predict all including first after phoneme boundary
    idx = np.arange(len(Y))
    
    num_timesteps = 50
    batch_size    = 1024
    num_features  = X.shape[1]
    dim           = ( batch_size, num_timesteps, num_features )
    
    data_split = int( data_split_factor * len(idx))
    idx_val   = idx[data_split:]
    
    print('evaluating and predicting... this will take a while')
    model = cast(Model, tf.keras.models.load_model( model_name + '.keras'))

    predict_generator = DataGenerator(
        idx = idx_val, X=X, Y=Y, out_dim = dim, 
        reduce_factor = 1, shuffle = False
    )
    evaluation = model.evaluate(predict_generator, verbose="1", return_dict=True)
    p = model.predict(predict_generator, verbose="1")

    idx_valed = np.array(idx_val)[np.array(range(len(p)))] + num_timesteps
    
    # safe log
    eps = 1e-12
    logp = np.log(np.clip(p, eps, 1.0))
    np.save( file_identifier_out + '_logp.npy', logp)
    
    y_pred = np.argmax(p,axis=1)
    y_true = Y[idx_valed]
    pred_correct = sum( ( np.logical_or( y_pred[:-8] == y_true[:-8], y_pred[8:] == y_true[:-8]  ) ) ) / (len(y_pred)-8)
    np.save( file_identifier_out + '_Phonemes39_pred.npy', y_pred)
    np.save( file_identifier_out + '_Phonemes39_true.npy',y_true)
    
    print(evaluation)   
    print(pred_correct)
    
def combineModelCausalNN(
    filename_X1: str, filename_X2: str, filename_Y: str, 
    model_name_1 = 'srA1_k_1', model_name_2 = 'srA1_h_0', 
    file_identifier_out = 'srA1',
):
    ''' combine models based on level and cepstral coefficients (or any two models) by adding their log probabilities '''

    predictPhonemeProbabilitiesCausalNN(
        filename_X = filename_X1, filename_Y = filename_Y, model_name = model_name_1, 
        file_identifier_out=f'{file_identifier_out}_1', data_split_factor = 0
    )
    logp_1  = np.load( f'{file_identifier_out}_1' + '_logp.npy')
    
    predictPhonemeProbabilitiesCausalNN(
        filename_X = filename_X2, filename_Y = filename_Y, model_name = model_name_2,
        file_identifier_out=f'{file_identifier_out}_2', data_split_factor = 0
    )
    logp_2  = np.load( f'{file_identifier_out}_2' + '_logp.npy')
    
    logp = logp_1 + logp_2
    np.save( file_identifier_out + '_logp_combined_all.npy', logp)
    np.save( file_identifier_out + '_p_combined_all.npy', np.exp(logp))
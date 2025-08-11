# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:48:18 2018

@author: js
"""
from typing import cast

import pandas as pd
import numpy as np
from numpy import ndarray

import tensorflow as tf
from keras.models import Model
from keras.callbacks import History
from tqdm.keras import TqdmCallback

from sklearn.utils import compute_class_weight
from .DataGenTimit import DataGenerator
from .logging import print_histories

def trainCausalNN( 
    filename_X: str, filename_Y: str, filename_idx: str, 
    file_identifier_out = 'srA1_a', 
    epochs_to_save = 1, epochs_total = 100, batch_size = 1024, reduce_factor = 1, load_model = None 
):

    X: ndarray = np.load( filename_X )
    Y: ndarray = np.load( filename_Y )
    idx: ndarray = np.load( filename_idx )
    
    # Check for silent frames
    assert not np.any(Y < 0), "unlabeled phonemes present" 

    classes = np.array([int(c) for c in np.unique(Y)])
    weights = compute_class_weight(
        class_weight = 'balanced',
        classes = classes,
        y=Y
    )
    class_weight = dict(zip(classes, weights))
    idx = idx.reshape(idx.shape[0],)
    
    num_features  = X.shape[1]
    num_timesteps = 50
    dim           = ( batch_size, num_timesteps, num_features )
    
    data_split = int( 0.9 * len(idx))
    idx_train = idx[:data_split]
    idx_val   = idx[data_split:]
    
    inp = tf.keras.Input( shape = ( num_timesteps, num_features ) )
    
    x = tf.keras.layers.GRU( 
        64, 
        return_sequences=True, 
        kernel_regularizer=tf.keras.regularizers.l2(0.01) 
    )(inp)

    x = tf.keras.layers.GRU( 
        64, 
        kernel_regularizer=tf.keras.regularizers.l2(0.01) 
    )(x)
    
    out = tf.keras.layers.Dense( 40, activation='softmax' )(x)

    model: Model = tf.keras.Model(inputs=inp, outputs=out)

    if load_model is not None:
        model = cast(Model, tf.keras.models.load_model( load_model )) # start from a previously saved model, discard model that was just compiled
    
    #tf.keras.utils.plot_model(model, show_shapes=True, to_file='model_causal.png')

    model.compile(loss='sparse_categorical_crossentropy', # using the cross-entropy loss function update to -> sparse_categorical_crossentropy
                  optimizer='adam', # using the Adam optimiser
                  metrics=['sparse_categorical_accuracy']) # reporting the accuracy
        
    training_generator = DataGenerator(idx_train, X, Y, dim, reduce_factor = reduce_factor)
    validation_generator = DataGenerator(idx_val, X, Y, dim, reduce_factor = reduce_factor)
    
    best_val = -float("inf")
    best_epoch = -1
    global_epoch = 0
    val_key = "val_sparse_categorical_accuracy"
    all_histories = []

    # Train model on dataset
    for i in range(int(epochs_total / epochs_to_save)):
        # callback to save every epoch with global epoch number
        save_all = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{file_identifier_out}_epoch{{epoch:04d}}.keras",
            save_best_only=False
        )
        
        # callback to save best model separately
        save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{file_identifier_out}_best.keras",
            monitor=val_key,
            mode="max",
            save_best_only=True
        )
        
        h = cast(History, model.fit(
            x=training_generator,
            validation_data=validation_generator,
            class_weight=class_weight,
            epochs=global_epoch + epochs_to_save,  # end epoch number
            initial_epoch=global_epoch,            # start epoch number
            callbacks=[save_all, save_best, TqdmCallback(verbose=1)],
            verbose="0"
        ))
        
        # Track best val acc manually
        for j, v in enumerate(h.history[val_key]):
            epoch_num = global_epoch + j + 1
            if v > best_val:
                best_val = v
                best_epoch = epoch_num

        all_histories.append(pd.DataFrame(h.history))
        global_epoch += epochs_to_save

    print_histories(
        val_key, best_val,best_epoch,
        all_histories
    )
    ### misc

    #model=tf.keras.models.load_model('srA1_h_0.h5')
    #
    #last_lr = tf.keras.backend.get_value(model.optimizer.lr)
    #print(last_lr)
    #tf.keras.backend.set_value(model.optimizer.lr, last_lr/2)
    #print(tf.keras.backend.get_value(model.optimizer.lr))

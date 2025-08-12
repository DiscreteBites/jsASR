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
from .DataGenTimitTri import DataGeneratorTri
from .logging import print_histories

def trainNonCausalNN( 
    filename_X: str, filename_Y: str, 
    file_identifier_out = 'srA2_a', 
    epochs_to_save = 1, epochs_total = 100, batch_size = 1024, reduce_factor = 1, load_model = None 
): 

    X = np.load( filename_X )
    Y = np.load( filename_Y )
    Y = Y[50:]   # 50 timesteps in first causal NN
    idx = np.arange(len(X))   # predict all including first after phoneme border
    
    # Check for silent frames
    assert not np.any(Y < 0), "unlabeled phonemes present" 
    
    classes = np.array([int(c) for c in np.unique(Y)])
    weights = compute_class_weight(
        class_weight = 'balanced',
        classes = classes,
        y=Y
    )
    
    # Three head class heads
    cw_dict = dict(zip(classes, weights))
    class_weight = {
        "out_prev2": cw_dict, 
        "out_now2": cw_dict, 
        "out_next2": cw_dict
    }

    idx = idx.reshape(idx.shape[0],)
    
    num_features  = X.shape[1]
    num_timesteps = 305
    dim           = ( batch_size, num_timesteps, num_features )
    
    data_split = int( 0.9 * len(idx))
    idx_train = idx[:data_split]
    idx_val   = idx[data_split:]
    
    #model=tf.keras.models.load_model('srA2_c_9.h5')
    #weights = model.get_weights()
    
    inp = tf.keras.Input( shape = ( num_timesteps, num_features ) )
    
    x = tf.keras.layers.AveragePooling1D( pool_size = 5, name = "pool_10ms")(inp)
    
    x = tf.keras.layers.Bidirectional( tf.keras.layers.GRU( 128, return_sequences=True, trainable = True ), name = "gru1" )(x) # , kernel_regularizer=tf.keras.regularizers.l2(0.01)
    x = tf.keras.layers.Bidirectional( tf.keras.layers.GRU( 128  ), name = "gru2", trainable = True )(x) # , recurrent_dropout=0.2
    
    out1 = tf.keras.layers.Dense( 40, activation='softmax', name = "out_prev2"  )(x)
    out2 = tf.keras.layers.Dense( 40, activation='softmax', name = "out_now2"  )(x)
    out3 = tf.keras.layers.Dense( 40, activation='softmax', name = "out_next2"  )(x)
    
    model: Model = tf.keras.Model(inputs=inp, outputs=[out1,out2,out3])
    
    if load_model is not None:
            model = cast(Model, tf.keras.models.load_model( load_model )) # start from a previously saved model, discard model that was just compiled
        
    #tf.keras.utils.plot_model(model,show_shapes=True, to_file='model_bidirectional.png')
    
    model.compile(loss='sparse_categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam', # using the Adam optimiser
                  metrics=['sparse_categorical_accuracy']) # reporting the accuracy
    
    training_generator = DataGeneratorTri(idx_train, X, Y, dim, reduce_factor = reduce_factor,non_causal_steps = int(num_timesteps/2))
    validation_generator = DataGeneratorTri(idx_val, X, Y, dim, reduce_factor = reduce_factor,non_causal_steps = int(num_timesteps/2))
    
    best_val = -float("inf")
    best_epoch = -1
    global_epoch = 0
    val_key = "val_sparse_categorical_accuracy"
    all_histories = []

    # Train model on dataset
    for i in range( int(epochs_total / epochs_to_save) ):
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
            x = training_generator,
            validation_data = validation_generator,
            class_weight = class_weight,
            epochs=global_epoch + epochs_to_save,   # end epoch number (global)
            initial_epoch=global_epoch,             # start epoch number (global)
            callbacks=[save_all, save_best, TqdmCallback(verbose=1)],
            verbose="1"
        ))

        # Track best val metric manually
        if val_key in h.history:
            for j, v in enumerate(h.history[val_key]):
                epoch_num = global_epoch + j + 1
                if v > best_val:
                    best_val = v
                    best_epoch = epoch_num

        # Accumulate history for summary
        all_histories.append(pd.DataFrame(h.history))

        # advance global epoch window
        global_epoch += epochs_to_save
    
    # print combined summary (reuse your helper)
    print_histories(
        val_key, best_val, best_epoch, 
        all_histories
    )
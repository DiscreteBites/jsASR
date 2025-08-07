# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:48:18 2018

@author: js
"""
from typing import cast

import numpy as np
from numpy import ndarray

import tensorflow as tf
from keras.models import Model

from sklearn.utils import compute_class_weight
from .DataGenTimit import DataGenerator

def trainCausalNN( filename_X: str, filename_Y: str, filename_idx: str, file_identifier_out = 'srA1_a', epochs_to_save = 1, epochs_total = 100, batch_size = 1024, reduce_factor = 1, load_model = None ):

    X: ndarray = np.load( filename_X )
    Y: ndarray = np.load( filename_Y )
    idx: ndarray = np.load( filename_idx )
    
    # Sanitize silent frames
    Y_clean = Y[Y != -1]

    classes = [int(c) for c in np.unique(Y_clean)]
    weights = compute_class_weight(
        class_weight = 'balanced',
        classes = classes,
        y=Y_clean
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

    model: Model
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    #tf.keras.utils.plot_model(model, show_shapes=True, to_file='model_causal.png')
    
    model.compile(loss='sparse_categorical_crossentropy', # using the cross-entropy loss function update to -> sparse_categorical_crossentropy
                  optimizer='adam', # using the Adam optimiser
                  metrics=['categorical_accuracy']) # reporting the accuracy
    
    if load_model is not None:
        model = cast(Model, tf.keras.models.load_model( load_model )) # start from a previously saved model, discard model that was just compiled
    
    training_generator = DataGenerator(idx_train, X, Y, dim, reduce_factor = reduce_factor)
    validation_generator = DataGenerator(idx_val, X, Y, dim, reduce_factor = reduce_factor)
    
    # Train model on dataset
    for i in range(int(epochs_total / epochs_to_save)):
        model.fit(
            x=training_generator,
            validation_data=validation_generator,
            class_weight=class_weight,
            epochs=epochs_to_save
        )
        model.save(f"{file_identifier_out}_{i}.h5")
        
    ### misc
        
    #model=tf.keras.models.load_model('srA1_h_0.h5')
    #
    #last_lr = tf.keras.backend.get_value(model.optimizer.lr)
    #print(last_lr)
    #tf.keras.backend.set_value(model.optimizer.lr, last_lr/2)
    #print(tf.keras.backend.get_value(model.optimizer.lr))

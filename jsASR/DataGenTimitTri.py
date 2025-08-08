# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 10:45:16 2019

@author: js2251
"""

import numpy as np
import tensorflow as tf
from .DataGenTimit import DataGenerator
from .TimitData import getNextLabel, getPreviousLabel

class DataGeneratorTri(DataGenerator):
    ''' generate data from timit, indeces given separately to have possibility to ignore first n timesteps
        out_dim: 0-th element is batch size further elements input dim of NN
        reduce_factor to use only a fraction of the data per (randomly shuffled) batch'''
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.Y_prev = getPreviousLabel(self.Y,100)
        self.Y_next = getNextLabel(self.Y,100)

    def __len__(self):
        ''' batches per epoch '''
        return int(np.floor(len(self.idx) / self.batch_size / self.reduce_factor))

    def __getitem__(self, index):       
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] # Generate indexes of the batch        
        list_idx_temp = [self.idx[k] for k in indexes] # Find list of IDs
        X, Y_TRIPLE = self.__data_generation(list_idx_temp)
        return X, Y_TRIPLE

    def on_epoch_end(self):
        '''Update indexes after each epoch'''
        super().on_epoch_end()

    def __data_generation(self, list_idx_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)

        X = np.empty(self.out_dim)
        Y_now = np.empty((self.batch_size), dtype=int)
        Y_prev = np.empty((self.batch_size), dtype=int)
        Y_next = np.empty((self.batch_size), dtype=int)

        w = 0  # write pointer
        for ID in list_idx_temp:
            label_now = self.Y[ID - self.non_causal_steps]
            if label_now == -1:
                continue  # skip silent frames based on "now" label (to mirror causal)

            # slice input window
            X[w] = self.X[ID - self.num_time_steps + 1 : ID + 1].reshape(self.out_dim_nobatch)

            # gather labels
            Y_now[w]  = label_now
            Y_prev[w] = self.Y_prev[ID - self.non_causal_steps]
            Y_next[w] = self.Y_next[ID - self.non_causal_steps]
            w += 1

        if w == 0:
            raise ValueError("Batch has only silence. Need to resample indices.")

        # Trim to actual filled size
        X = X[:w]
        Y_prev = Y_prev[:w]
        Y_now  = Y_now[:w]
        Y_next = Y_next[:w]
        
        # Safety: no negatives remain (if prev/next can be -1 too, switch to assert on all three)
        assert not np.any(Y_now  < 0), "Negative 'now' labels found"
        assert not np.any(Y_prev < 0), "Negative 'prev' labels found"
        assert not np.any(Y_next < 0), "Negative 'next' labels found"

        return X, [Y_prev, Y_now, Y_next]
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 10:45:16 2019

@author: js2251
"""

import numpy as np

from .DataGenTimit import DataGenerator
from .TimitData import getNextLabel, getPreviousLabel

class DataGeneratorTri(DataGenerator):
    ''' generate data from timit, indeces given separately to have possibility to ignore first n timesteps
        out_dim: 0-th element is batch size further elements input dim of NN
        reduce_factor to use only a fraction of the data per (randomly shuffled) batch'''
    def __init__(self, cw_dict, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.Y_prev = getPreviousLabel(self.Y,100)
        self.Y_next = getNextLabel(self.Y,100)
        self.cw_dict  = cw_dict

    def __len__(self):
        ''' batches per epoch '''
        return int(np.floor(len(self.idx) / self.batch_size / self.reduce_factor))

    def __getitem__(self, index):       
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] # Generate indexes of the batch      
        list_idx_temp = [self.idx[k] for k in indexes] # Find list of IDs
        X, Y_TARGETS, Y_WEIGHTS = self.__data_generation(list_idx_temp)
        return X, Y_TARGETS, Y_WEIGHTS

    def on_epoch_end(self):
        '''Update indexes after each epoch'''
        super().on_epoch_end()

    def __data_generation(self, list_idx_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)

        X = np.empty(self.out_dim)
        Y_now = np.empty((self.batch_size), dtype=int)
        Y_prev = np.empty((self.batch_size), dtype=int)
        Y_next = np.empty((self.batch_size), dtype=int)
        
        for i, ID in enumerate(list_idx_temp):
            X[i,] = self.X[ID - self.num_time_steps + 1 : ID + 1].reshape(self.out_dim_nobatch)
            
            Y_now[i]  = self.Y[ID - self.non_causal_steps]
            Y_prev[i] = self.Y_prev[ID - self.non_causal_steps]
            Y_next[i] = self.Y_next[ID - self.non_causal_steps]
        
        # Safety: no negatives remain (if prev/next can be -1 too, switch to assert on all three)
        assert not np.any(Y_now  < 0), "unlabeled phonemes present in 'now'"
        assert not np.any(Y_prev < 0), "unlabeled phonemes present in 'prev'"
        assert not np.any(Y_next < 0), "unlabeled phonemes present in 'next'"
        
        targets = (Y_prev, Y_now, Y_next)

        # build per-example sample weights from cw_dict
        sw_prev = np.asarray([self.cw_dict[y] for y in Y_prev], dtype="float32")
        sw_now  = np.asarray([self.cw_dict[y] for y in Y_now],  dtype="float32")
        sw_next = np.asarray([self.cw_dict[y] for y in Y_next], dtype="float32")

        sample_weights = (sw_prev, sw_now, sw_next)

        return X, targets, sample_weights
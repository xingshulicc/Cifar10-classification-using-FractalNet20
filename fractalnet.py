# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Sep 11 15:26:04 2018

@author: xingshuli
"""

import numpy as np

from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.engine import Layer
from keras import backend as K

import tensorflow as tf

def tensorflow_categorical(count, seed):
    assert count > 0
    arr = [1.] + [.0 for _ in range(count - 1)]
    return tf.random_shuffle(arr, seed)

# return a random array in which only one element is 1 and the others are 0
# Ex: [0, 0, 1, 0, 0]
def rand_one_in_array(count, seed = None):
    if seed is None:
        seed = np.random.randint(1, 10e6)
        
    return tensorflow_categorical(count = count, seed = seed)

class JoinLayer(Layer):
    '''
    This layer will behave as Merge during testing but during training
    it will randomly select between using local or global drop-path and 
    apply the average of the paths alive after applying the drops.
        
    '''
    def __init__(self, drop_p, is_global, global_path, force_path, **kwargs):
        self.p = 1. - drop_p
        self.is_global = is_global
        self.global_path = global_path
        self.uses_learning_phase = True
        self.force_path = force_path
        super(JoinLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        #assume the number of columns for a join layer is 3 then:
        #the input_shape = [column_1_shape, column_2_shape, column_3_shape]
        #average_shape = [channels, feature_map_height, feature_map_width]
        
        self.average_shape = list(input_shape[0])[1:]
        
    def _random_arr(self, count, p):
        return K.random_binomial((count,), p = p)
    
    def _arr_with_one(self, count):
        return rand_one_in_array(count = count)
    
    def _gen_local_drops(self, count, p):
        arr = self._random_arr(count, p)
        drops = K.switch(K.any(arr), arr, self._arr_with_one(count))
        
        return drops
    
    def _gen_global_path(self, count):
        return self.global_path[:count]
    
    def _drop_path(self, inputs):
        
        #inputs = [[input_1], [input_2], [input_3], ...]
        #count is the number of columns for a join layer
        
        count = len(inputs)
        drops = K.switch(self.is_global, self._gen_global_path(count), 
                         self._gen_local_drops(count, self.p))
        
        ave = K.zeros(shape = self.average_shape)
        for i in range(0, count):
            ave += inputs[i] * drops[i]
        
        sum = K.sum(drops)
        ave = K.switch(K.not_equal(sum, 0.), ave/sum, ave)
        
        return ave
    
    def _ave(self, inputs):
        ave = inputs[0]
        for input in inputs[1:]:
            ave += input
        ave /= len(inputs)
        return ave
    
    def call(self, inputs, mask = None):
        if self.force_path:
            output = self._drop_path(inputs)
        else:
            output = K.in_train_phase(self._drop_path(inputs), self._ave(inputs))
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
class JoinLayerGen:
    
    '''
    JoinLayerGen will initialize seeds for both global_switch and global_path
    If global_switch = True, then generate global_path 
    Otherwise generate local_path    
    '''
    def __init__(self, width, global_p = 0.5, deepest = False):
        self.global_p = global_p
        self.width = width
        self.switch_seed = np.random.randint(1, 10e6)
        self.path_seed = np.random.randint(1, 10e6)
        self.deepest = deepest
        
        if deepest:
            self.is_global = K.variable(1.)
            self.path_array = K.variable([1.] + [.0 for _ in range(width - 1)])
        else:
            self.is_global = self._build_global_switch()
            self.path_array = self._build_global_path_arr()
            
    def _build_global_path_arr(self):
        return rand_one_in_array(seed = self.path_seed, count = self.width)
    
    def _build_global_switch(self):
        return K.equal(K.random_binomial((), p = self.global_p, seed = self.switch_seed), 1.)
        
    def get_join_layer(self, drop_p):
        global_switch = self.is_global
        global_path = self.path_array
        
        return JoinLayer(drop_p = drop_p, global_path = global_path, is_global = global_switch, force_path = self.deepest)
        
    
def fractal_conv(filter, nb_row, nb_col, dropout = None):
    '''
    The location of Dropout is not clear in the paper (arXiv:1605.07648), 
    in this code we refer the idea of arXiv:1801.05134 which puts Dropout after all BN layers
    '''
    def f(prev):
        conv = prev
        conv = Conv2D(filters = filter, kernel_size = (nb_row, nb_col), 
                      padding = 'same', kernel_initializer = 'he_normal')(conv)
    
        if dropout:
            conv = Dropout(dropout)(conv)
        conv = BatchNormalization(axis = -1)(conv)        
        conv = Activation('relu')(conv)
        
        return conv
        
    return f

def fractal_block(join_gen, c, filter, nb_col, nb_row, drop_p, dropout = None):
    def f(z):
        columns = [[z] for _ in range(c)]
#        last_row = 2**(c-1) - 1
        for row in range(2**(c-1)):
            t_row = []
            for col in range(c):
                prop = 2**(col)
                # Add blocks 
                if (row+1) % prop == 0:
                    t_col = columns[col]
                    t_col.append(fractal_conv(filter = filter,
                                              nb_col = nb_col,
                                              nb_row = nb_row,
                                              dropout = dropout)(t_col[-1]))
                    t_row.append(col)
            # Merge (if needed)
            if len(t_row) > 1:
                merging = [columns[x][-1] for x in t_row]
                merged  = join_gen.get_join_layer(drop_p = drop_p)(merging)
                for i in t_row:
                    columns[i].append(merged)
        return columns[0][-1]
    
    return f

def fractal_net(b, c, conv, drop_path, global_p = 0.5, dropout = None, deepest = False):
    '''
    b: the number of building blocks for entire network
    c: the number of columns for each building block
    conv: the information of 2D convolution for each building block
    deepest: if deepest is True, we should generate single column through network, 
    which has the most convolutional layers
    
    '''
    def f(z):
        output = z
        join_gen = JoinLayerGen(width = c, global_p = global_p, deepest = deepest)
        for i in range(b):
            (filter, nb_col, nb_row) = conv[i]
            dropout_i = dropout[i] if dropout else None
            output = fractal_block(join_gen = join_gen, 
                                   c = c, filter = filter, nb_col = nb_col, 
                                   nb_row = nb_row, drop_p = drop_path, 
                                   dropout = dropout_i)(output)
            
            output = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(output)
            
        return output
        
    return f
                                   










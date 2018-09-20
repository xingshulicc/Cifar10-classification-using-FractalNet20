# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Sep 18 11:24:35 2018

@author: xingshuli
"""
import os

from fractalnet import fractal_net
from learning_rate import choose

from keras.utils import np_utils
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout

from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
#from keras.optimizers import Adam

#from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

#GPU arrangement
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # '1' or '0' GPU

nb_classes = 10
epochs = 500
batch_size = 64
lr = 0.01

#use data augmentation to improve accuracy
data_augmentation = True

#load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)

#convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /=255
x_test /=255

#construct model and compile
def build_network(deepest = False):
    dropout = [0., 0., 0., 0., 0.]
    conv = [(64, 3, 3), (128, 3, 3), (256, 3, 3), (512, 3, 3), (512, 2, 2)]
    input = Input(shape = (32, 32, 3))
    output = fractal_net(c = 3, b = 5, conv = conv, drop_path = 0.15, 
                         dropout = dropout, deepest = deepest)(input)
    
    output = Flatten()(output)
    output = Dense(512, kernel_initializer = 'he_normal')(output)
    output = Activation('relu')(output)
    output = Dropout(0.25)(output)
    output = Dense(nb_classes, kernel_initializer = 'he_normal')(output)
    output = Activation('softmax')(output)
    
    model = Model(inputs = input, outputs = output)
    optimizer = SGD(lr = lr, momentum = 0.9, nesterov = True)
#    optimizer = Adam()
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics= ['accuracy'])
#    model.summary()    
    
    return model

#set callbacks in model
patience = 9
early_stop = EarlyStopping(monitor = 'val_loss', patience = patience, mode = 'auto')

#set learning rate schedule
lr_monitorable = True
lr_reduce = choose(lr_monitorable = lr_monitorable)

#set callbacks
callbacks = [early_stop, lr_reduce]

#train model with data augmentation or not
if not data_augmentation:
    print('training without data augmentation')
    net = build_network(deepest = False)
    net.fit(x_train, y_train, batch_size = batch_size, 
            epochs = epochs, validation_data = (x_test, y_test), 
            shuffle = True, callbacks = callbacks
            )
else:
    print('training with data augmentation')
    net = build_network(deepest = False)
    datagen = ImageDataGenerator(horizontal_flip = True, 
                                 width_shift_range = 0.1, 
                                 height_shift_range = 0.1)
    datagen.fit(x_train)
    
    hist = net.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), 
                             steps_per_epoch = x_train.shape[0]//batch_size, 
                             epochs = epochs, validation_data = (x_test, y_test), 
                             callbacks = callbacks
                             )
                             
    
    #print(hist.history['acc'])
    f = open('/home/xingshuli/Desktop/acc.txt','w')
    f.write(str(hist.history['acc']))
    f.close()
    #print(hist.history['val_acc'])
    f = open('/home/xingshuli/Desktop/val_acc.txt','w')
    f.write(str(hist.history['val_acc']))
    f.close()
    
#Evaluate model based on the patience of EarlyStopping
Er_patience = patience + 1
accur = []
with open('/home/xingshuli/Desktop/val_acc.txt','r') as f1:
    data1 = f1.readlines()
    for line in data1:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        accur.append(num_float)
f1.close()

y = sum(accur, [])
ave = sum(y[-Er_patience:]) / len(y[-Er_patience:])
print('Validation Accuracy = %.4f' % (ave))

#save model
save_dir = os.path.join(os.getcwd(), 'fractal_net')
model_name = 'keras_fractalnet_trained_model.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
net.save(model_path)
print('the model has been saved at %s' %model_path)




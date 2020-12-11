#!/usr/bin/env python

# History
# v01 : adaptation from the one given by Udacity to work
# v02 : adapt to commonFunctions_v10.py to use generator.
#       Start adding again everything from model_v12.py (image augmentation)
# v03 : migrate model from model_v12.py to generator_v03.py, tested on GPU ok.
#       Just need 6 or 7 epochs, not more.
# v04 : Add functionality to load different data collections
#       + add data Last Hard Turn

import os
import csv
import cv2
import numpy as np

from math import ceil
from sklearn.model_selection import train_test_split

from commonFunctions_v12 import get_lines_logfile 
from commonFunctions_v12 import generator
from commonFunctions_v12 import batch_len
from commonFunctions_v12 import print_info
from commonFunctions_v12 import visualize_loss_history
from commonFunctions_v12 import get_log_pathSampleData

path_last_hard_turn_data = "./simulationData/002_hardLeftTurn20201208_0220/"

# Reading CSV file FROM SAMPLE DATA, extracting lines.
samples = get_lines_logfile(get_log_pathSampleData())
# Reading CSV file from Last Hard Turn, extracting lines
# add them to samples lines.
samples.extend(get_lines_logfile(path_last_hard_turn_data))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set our batch size (*3 due to image center + left + right ....), then *2 due to flip of each images
batch_size=batch_len*3*2  #6*3*2 = 36 ....

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

print_info('Import Keras Models. Please wait ...')
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
print_info('Import Keras Models. Done.')

print_info('Build Model. Please wait ...')
model = Sequential()
model.add(Lambda(lambda x: ((x/255) - 0.5),input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Nvidia : strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel
# The input image is split into YUV planes and passed to the network.
model.add(Conv2D(filters=24,kernel_size=5,strides=2,padding="valid"))
model.add(Conv2D(filters=36,kernel_size=5,strides=2,padding="valid"))
model.add(Conv2D(filters=48,kernel_size=5,strides=2,padding="valid"))
# and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.
model.add(Conv2D(filters=64,kernel_size=3,strides=1,padding="valid"))
model.add(Conv2D(filters=64,kernel_size=3,strides=1,padding="valid"))
# follow the five convolutional layers with three fully connected layers, 
# leading to a final output control value which is the inverse-turning-radius. 
#model.add(Dropout(dropout_rate))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(dropout_rate))
model.add(Activation('relu'))
model.add(Dense(50))
#model.add(Dropout(dropout_rate))
model.add(Activation('relu'))
model.add(Dense(10))
#model.add(Dropout(dropout_rate))
model.add(Activation('relu'))
model.add(Dense(1))
print_info('Build Model. Done ...')

print_info('Model.compile(). Please wait ...')
model.compile(loss='mse', optimizer='adam')
print_info('Model.compile(). Done.')
# Callbacks to save best model and prevent overfit by early stopping 
checkpoint = ModelCheckpoint(filepath='bestModelFolder/model.{epoch:02d}-Loss{loss:.4f}-valLoss{val_loss:.4f}.h5', monitor='val_loss', save_best_only=False)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)
history_object = model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=6, verbose=1, callbacks=[checkpoint]) #, stopper])

print_info('Saving model parameters. Please wait ...')
model.save('model.h5')
print_info('Saving model parameters. Done.')
# save picture lossHistory.png
visualize_loss_history(history_object)
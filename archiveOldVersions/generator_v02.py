#!/usr/bin/env python

# History
# v01 : adaptation from the one given by Udacity to work
# v02 : adapt to commonFunctions_v10.py to use generator.
#       Start adding again everything from model_v12.py (image augmentation)

import os
import csv
import cv2
import numpy as np
import sklearn

from math import ceil
from random import shuffle
from sklearn.model_selection import train_test_split

from commonFunctions_v10 import get_lines_logfile 
from commonFunctions_v10 import get_info_from_lines
from commonFunctions_v10 import flip_horizontally

STEER_CORRECTION_FACTOR = 0.2 # to tune up for left and right images/measurements

# Set our batch size for fit generator
batch_len= 6

# Reading CSV file, extracting lines.
samples = get_lines_logfile()

train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)


def generator(samples, batch_size=batch_len):
    num_samples = len(samples)
    # print('num_samples : {}'.format(num_samples))
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # correction : should go only until min(num_samples,offset+batch_size)
            batch_samples = samples[offset: min(num_samples,offset+batch_size)]

            # here will get both center, left, right images + their measurements.
            # if batch_size = 32 --> 32*3 = 96 images ....
            images, angles = get_info_from_lines(batch_samples,STEER_CORRECTION_FACTOR,nb_images=None)
            # data augmentation flip horizontally image + inverse measurements
            augm_images, augm_measurements = flip_horizontally(images,angles)
            images.extend(augm_images)
            angles.extend(augm_measurements)
            
            # Nvidia : need to convert images in YUV ...
            images = RGB2YUV(images)
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size (*3 due to image center + left + right ....), then *2 due to flip of each images
batch_size=batch_len*3*2  #6*3*2 = 36 ....

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)
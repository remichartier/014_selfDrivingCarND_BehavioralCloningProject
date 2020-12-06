#!/usr/bin/env python
import numpy as np
import cv2

from commonFunctions_v03 import get_info_from_logfile
from commonFunctions_v03 import flip_horizontally

# History
# v01 : Start
# v02 : add nb_images to read parameter
# v03 : add normalization + mean centering data to 0
# v04 : data augmentation flip horizontally image + inverse measurements

# get images + steering angle measurements
images, measurements = get_info_from_logfile(nb_images=100)

# data augmentation flip horizontally image + inverse measurements
augm_images, augm_measurements = flip_horizontally(images,measurements)
images.extend(augm_images)
measurements.extend(augm_measurements)

X_train = np.array(images)
y_train = np.array(measurements)

#print(f'X_train shape : {X_train.shape}')
#print(f'images shape : {im.shape}')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.callbacks import ModelCheckpoint,EarlyStopping

model = Sequential()
model.add(Lambda(lambda x: ((x/255) - 0.5),input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# Callbacks to save best model and prevent overfit by early stopping 
checkpoint = ModelCheckpoint(filepath='bestModelFolder/model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)
# model.fit(callbacks=[checkpoint, stopper])
model.fit(X_train,y_train, validation_split=0.2, shuffle = True, epochs=10, callbacks=[checkpoint, stopper])

model.save('model.h5')
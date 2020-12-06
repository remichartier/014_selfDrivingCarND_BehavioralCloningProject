#!/usr/bin/env python
import numpy as np
import cv2

# History
# v01 : start
# v02 : try use only np.arrays, to skip conversion list/np.arrays

from commonFunctions_v02 import get_info_from_logfile

X_train, y_train = get_info_from_logfile()

# X_train = np.array(images)
# y_train = np.array(measurements)

print(f'X_train shape : {X_train.shape}, y_train shape : {y_train.shape()}')
#print(f'images shape : {im.shape}')

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# Callbacks to save best model and prevent overfit by early stopping 
checkpoint = ModelCheckpoint(filepath='bestModelFolder/model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)
# model.fit(callbacks=[checkpoint, stopper])
model.fit(X_train,y_train, validation_split=0.2, shuffle = True, epochs=10, callbacks=[checkpoint, stopper])

model.save('model.h5')
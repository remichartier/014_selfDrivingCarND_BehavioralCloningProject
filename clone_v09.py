#!/usr/bin/env python
import numpy as np
import cv2

from commonFunctions_v06 import get_info_from_logfile
from commonFunctions_v06 import flip_horizontally
from commonFunctions_v06 import visualize_loss_history

# History
# v01 : Start
# v02 : add nb_images to read parameter
# v03 : add normalization + mean centering data to 0
# v04 : data augmentation flip horizontally image + inverse measurements
# v05 : use left/right images + measurements with Steering error correction
# v06 : cropping images
# v07 : add a generator to load data and preprocess it on the fly, in batchsize portions 
#        to feed into your Behavioral Cloning model .
# v08 : Adding loss viusalization tool
# v09 : Re-start from v06 as fit_generator and need to add generator obsolete.
#       Latest Keras.Model.fit integrates a generator in itself.
#       ie v09 : Visualize loss history

STEER_CORRECTION_FACTOR = 0.2 # to tune up for left and right images/measurements

# Set our batch size for fit generator
batch_size=32

# get images + steering angle measurements
images, measurements = get_info_from_logfile(STEER_CORRECTION_FACTOR,nb_images=100)

# data augmentation flip horizontally image + inverse measurements
augm_images, augm_measurements = flip_horizontally(images,measurements)
images.extend(augm_images)
measurements.extend(augm_measurements)

X_train = np.array(images)
y_train = np.array(measurements)

#print(f'X_train shape : {X_train.shape}')
#print(f'images shape : {im.shape}')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

model = Sequential()
model.add(Lambda(lambda x: ((x/255) - 0.5),input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# Callbacks to save best model and prevent overfit by early stopping 
checkpoint = ModelCheckpoint(filepath='bestModelFolder/model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=10)
# model.fit(callbacks=[checkpoint, stopper])
history_object = model.fit(X_train,y_train, batch_size, validation_split=0.2, shuffle = True, epochs=10, callbacks=[checkpoint, stopper])

'''
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)
'''

model.save('model.h5')

# save picture lossHistory.png
visualize_loss_history(history_object)


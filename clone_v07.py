#!/usr/bin/env python
import numpy as np
import cv2

from commonFunctions_v04 import get_info_from_logfile
from commonFunctions_v04 import flip_horizontally

# History
# v01 : Start
# v02 : add nb_images to read parameter
# v03 : add normalization + mean centering data to 0
# v04 : data augmentation flip horizontally image + inverse measurements
# v05 : use left/right images + measurements with Steering error correction
# v06 : cropping images
# v07 : add a generator to load data and preprocess it on the fly, in batch size portions 
#       to feed into your Behavioral Cloning model .

STEER_CORRECTION_FACTOR = 0.2 # to tune up for left and right images/measurements

# Now done in generator function --> I comment lines 
# get images + steering angle measurements
# images, measurements = get_info_from_logfile(STEER_CORRECTION_FACTOR,nb_images=100,test_size=0.2)

lines = get_lines_logfile()

# need to insert Generator here. Samples = lines from driving_log_file = 'driving_log.csv'
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Set our batch size
batch_size=32

# data augmentation flip horizontally image + inverse measurements
# Now done in generator function --> I comment lines 
# augm_images, augm_measurements = flip_horizontally(images,measurements)
# images.extend(augm_images)
# measurements.extend(augm_measurements)

# conversion to np.arrays
# Now done in generator function --> I comment lines 
# X_train = np.array(images)
# y_train = np.array(measurements)

#print(f'X_train shape : {X_train.shape}')
#print(f'images shape : {im.shape}')

# compile and train the model using the generator function
train_generator = generator(train_samples,leftright_steer_corr=STEER_CORRECTION_FACTOR , batch_size=batch_size)
validation_generator = generator(validation_samples,leftright_steer_corr=STEER_CORRECTION_FACTOR , batch_size=batch_size)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

# test https://devblogs.nvidia.com/parrallelforall/deep-learning-self-driving-cars

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: ((x/255) - 0.5),input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# Callbacks to save best model and prevent overfit by early stopping 
checkpoint = ModelCheckpoint(filepath='bestModelFolder/model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)
# model.fit(callbacks=[checkpoint, stopper])
# model.fit(X_train,y_train, validation_split=0.2, shuffle = True, epochs=10, callbacks=[checkpoint, stopper])
model.fit_generator(train_generator, /
            steps_per_epoch=ceil(len(train_samples)/batch_size), /
            validation_data=validation_generator, /
            validation_steps=ceil(len(validation_samples)/batch_size), /
            epochs=5, verbose=1,callbacks=[checkpoint, stopper])
model.save('model.h5')
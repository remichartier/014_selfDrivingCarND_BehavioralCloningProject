#!/usr/bin/env python
import numpy as np
import cv2

from commonFunctions_v05 import get_lines_logfile
from commonFunctions_v05 import flip_horizontally
from commonFunctions_v05 import generator
from sklearn.model_selection import train_test_split
from math import ceil

# History
# v01 : Start
# v02 : add nb_images to read parameter
# v03 : add normalization + mean centering data to 0
# v04 : data augmentation flip horizontally image + inverse measurements
# v05 : use left/right images + measurements with Steering error correction
# v06 : cropping images
# v07 : add a generator to load data and preprocess it on the fly, in batch size portions 
#       to feed into your Behavioral Cloning model .
# v08 : Adding loss viusalization tool

STEER_CORRECTION_FACTOR = 0.2 # to tune up for left and right images/measurements

# Now done in generator function --> I comment lines 
# get images + steering angle measurements
# images, measurements = get_info_from_logfile(STEER_CORRECTION_FACTOR,nb_images=100,test_size=0.2)

lines = get_lines_logfile(nb_lines=1000)

# need to insert Generator here. Samples = lines from driving_log_file = 'driving_log.csv'
# note : remove first line (column titles) --> lines[1:]
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

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

# For Loss visualization :
from keras.models import Model
import matplotlib.pyplot as plt

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
# stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)
# model.fit(callbacks=[checkpoint, stopper])
# model.fit(X_train,y_train, validation_split=0.2, shuffle = True, epochs=10, callbacks=[checkpoint, stopper])
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples)/batch_size),
                    epochs=5, verbose=1,callbacks=[checkpoint])

from keras.models import Model
import matplotlib.pyplot as plt

'''
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)
'''

'''
./clone_v08.py:92: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  nb_epoch=5, verbose=1)
./clone_v08.py:92: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<generator..., validation_data=<generator..., verbose=1, steps_per_epoch=799, epochs=5, validation_steps=200)`
  nb_epoch=5, verbose=1)
'''

history_object = model.fit_generator(train_generator,validation_data =validation_generator,
                                     validation_steps = len(validation_samples),
                                     epochs=5, verbose=1, steps_per_epoch=len(train_samples))

model.save('model.h5')


### print the keys contained in the history object
print(history_object.history.keys())

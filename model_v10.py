#!/usr/bin/env python
import numpy as np
import cv2

from commonFunctions_v07 import get_info_from_logfile
from commonFunctions_v07 import flip_horizontally
from commonFunctions_v07 import visualize_loss_history
from commonFunctions_v07 import RGB2YUV

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
# v10 : choose better model for self driving cars and for this simulation.
#       Trying https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars

STEER_CORRECTION_FACTOR = 0.2 # to tune up for left and right images/measurements

# Set our batch size for fit generator
batch_size=32

# get images + steering angle measurements
images, measurements = get_info_from_logfile(STEER_CORRECTION_FACTOR,nb_images=100)

# data augmentation flip horizontally image + inverse measurements
augm_images, augm_measurements = flip_horizontally(images,measurements)
images.extend(augm_images)
measurements.extend(augm_measurements)

# Nvidia : need to convert images in YUV ...
images = RGB2YUV(images)

print('converting images to np arrays. Please wait ...')
X_train = np.array(images)
y_train = np.array(measurements)
print('converting images to np arrays. Done')

#print(f'X_train shape : {X_train.shape}')
#print(f'images shape : {im.shape}')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers.convolutional import Conv2D

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
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Activation('relu'))
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


import numpy as np
import cv2

from commonFunctions_v01 import get_info_from_logfile()

images, measurements = get_info_from_logfile()

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train, validation_split=0.2, shuffle = True)

model.save('model.h5')
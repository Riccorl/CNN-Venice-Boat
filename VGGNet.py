import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
# Conv layer 1
model.add(Conv2D(32, (3, 3), input_shape=(200, 60, 3)))
# ReLU layer 1
model.add(Activation('relu'))
# Conv layer 2
model.add(Conv2D(32, (3, 3), input_shape=(200, 60, 3)))
# ReLU layer 2
model.add(Activation('relu'))
# Pool layer 1
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv layer 3
model.add(Conv2D(32, (3, 3), input_shape=(200, 60, 3)))
# ReLU layer 3
model.add(Activation('relu'))
# Conv layer 4
model.add(Conv2D(32, (3, 3), input_shape=(200, 60, 3)))
# ReLU layer 4
model.add(Activation('relu'))
# Pool layer 2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv layer 5
model.add(Conv2D(32, (3, 3), input_shape=(200, 60, 3)))
# ReLU layer 5
model.add(Activation('relu'))
# Conv layer 6
model.add(Conv2D(32, (3, 3), input_shape=(200, 60, 3)))
# ReLU layer 6
model.add(Activation('relu'))
# Pool layer 3
model.add(MaxPooling2D(pool_size=(2, 2)))

# From 3D feature maps to 1D feature vectors
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(24))
model.add(Activation('softmax'))

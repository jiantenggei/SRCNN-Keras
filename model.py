import os
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.core import Activation


def built_model(input_shape=(33, 33, 1)):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=9,
                    padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(input_shape[2], 5, padding='same'))
    return model


if __name__ == '__main__':
    model = built_model()
    model.summary()

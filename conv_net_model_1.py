"""
Author: Will Nock, ISIS, Vanderbilt School of Engineering
Speech Disorder Diagnosing with Deep Learning

File: conv_net_model_1.py :
This file architects a neural net which categorizes spectrograms into diagnoses

------ MODEL 1 -------
Taken from : https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/

"""

from keras.models import Sequential
from keras.layers import InputLayer, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D


def makeModel():
    filters = 10
    filtersize = (5,5)
    input_shape = (256, 256, 3)
    
    model = Sequential()
    
    model.add(InputLayer(input_shape=input_shape))
    
    model.add(Conv2D(filters, filtersize, strides=(1, 1), padding='valid', data_format="channels_last", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    model.add(Dense(units=2, input_dim=50,activation='softmax'))
    
    return model


""" More Complex Model... maybe for later
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(1,256,256)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
    """

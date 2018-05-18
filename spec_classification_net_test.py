"""
Author: Will Nock, ISIS, Vanderbilt School of Engineering
Speech Disorder Diagnosing with Deep Learning

File: spec_classification_net_test.py :
This file tests a neural net which categorizes spectrograms into diagnoses
"""

"""
import importlib
importlib.import_module("import_data.py")
importlib.import_module("conv_net_model_1.py")
"""
from conv_net_model_1 import makeModel
import import_data

import numpy as np
import cv2
from keras.utils.np_utils import to_categorical


images, labels = import_data.getAllSamples()

for image in images:
    image = cv2.resize(image,(256,256))
    
images = np.array(images)
labels = np.array(labels)
labels = to_categorical(labels)


epochs =5
batchsize=128

model = makeModel()
model.summary()

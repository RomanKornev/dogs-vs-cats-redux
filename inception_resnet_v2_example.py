# -*- coding: utf-8 -*-
'''Inception-ResNet-v2 model for Keras.
'''
from __future__ import print_function

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

if __name__ == '__main__':
    model = InceptionResNetV2(include_top=True, weights='imagenet')

    img_path = 'data\\dogscats\\train\\cats\\cat.10013.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))  # ('n02123394', 'Persian_cat', 0.94211012)

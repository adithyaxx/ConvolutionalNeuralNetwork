from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import scipy
import numpy as np
import argparse
import cv2

IMG_SIZE = 50
LR = 1e-3

parser = argparse.ArgumentParser(description='Decide if an image is a picture of a bird')
parser.add_argument('image', type=str, help='The image image file to check')
args = parser.parse_args()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0, checkpoint_path='cat-dog-identifier.tfl.ckpt')

model.load('cat-dog-identifier.tfl.ckpt-383')

img = cv2.resize(cv2.imread(args.image, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))

# Predict
prediction = model.predict([img])

# Check the result.
is_dog = np.argmax(prediction[0]) == 1

if is_dog:
    print("That's a dog!")
else:
    print("That's a cat!")

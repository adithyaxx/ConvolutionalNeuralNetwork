from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import scipy
import numpy as np
import argparse
import cv2
import os

IMG_SIZE = 50
LR = 1e-3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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

model.load('./bestcheckpoint.tfl.ckpt7940')

img = cv2.resize(cv2.imread(args.image, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))

data = img.reshape(IMG_SIZE, IMG_SIZE, 1)
prediction = model.predict([data])[0]

print(f"cat: {prediction[0]}, dog: {prediction[1]}")

val = prediction[0] * 100

if (val >= 50):
    print(f"That's a cat ({val}%)!")
else:
    print(f"That's a dog! ({100-val}%)!")

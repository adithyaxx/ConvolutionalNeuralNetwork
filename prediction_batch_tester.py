from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse
import cv2
import os

IMG_SIZE = 32
TEST_DIR = 'test'

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.5)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0, best_val_accuracy=0.85, best_checkpoint_path='best')

model.load('model/ckpt8620')

for img in os.listdir(TEST_DIR):
	path = os.path.join(TEST_DIR, img)
	img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
	image = img.reshape(1,IMG_SIZE,IMG_SIZE,3)
	prediction = model.predict(image)[0]

	val = prediction[0] * 100

	if (val >= 50):
		print(f"That's a cat ({val}%)!\n")
	else:
		print(f"That's a dog! ({100-val}%)!\n")

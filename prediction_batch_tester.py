from __future__ import division, print_function, absolute_import

from PIL import Image
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import scipy
import numpy as np
import argparse
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

IMG_SIZE = 32
TEST_DIR = '/Users/adithya/Desktop/test'

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
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0, best_val_accuracy=0.85,
                    best_checkpoint_path='best')

model.load('model/ckpt8720')

fig = plt.figure(figsize=(16, 12))

for num in range(1,17):

    img_data = cv2.resize(cv2.imread(f'test/{num}.jpg'), (IMG_SIZE, IMG_SIZE))

    y = fig.add_subplot(4, 4, num)
    orig = cv2.resize(cv2.imread(f'test/{num}.jpg'), (128, 128))
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog: {:.4f}'.format(model_out[1])
    else:
        str_label = 'Cat: {:.4f}'.format(model_out[0])

    y.imshow(orig, interpolation='nearest')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

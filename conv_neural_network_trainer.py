import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import matplotlib.pyplot as plt

IMG_SIZE = 32
MODEL_NAME = 'cat-dog-identifier'
TEST_DIR = 'test'

def label_img(img):
    # dog.93.png
    word_label = img.split('.')[-3]

    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img, dtype=np.float64), np.array(label, dtype=np.float64)])
    shuffle(training_data)
    np.save('datasets/train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img, dtype=np.float64), img_num])

    shuffle(testing_data)
    np.save('datasets/test_data.npy', testing_data)
    return testing_data


#train_data = create_train_data()
#test_data = create_test_data()

train_data = np.load('datasets/train_data.npy')
test_data = np.load('datasets/test_data.npy')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train])
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test])
test_y = [i[1] for i in test]

X, Y = shuffle(X, Y)

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], data_preprocessing=img_prep, data_augmentation=img_aug, name='input')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.5)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0, best_val_accuracy=0.85, best_checkpoint_path='model/ckpt')

model.load('model/cat-dog-identifier.tfl')
print('Existing Model loaded!')

model.fit({'input': X}, {'targets': Y}, n_epoch=100,
          validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, shuffle=True, show_metric=True, batch_size=96, run_id=MODEL_NAME, snapshot_epoch=True)

model.save("model/cat-dog-identifier.tfl")
print("Model saved successfully!")
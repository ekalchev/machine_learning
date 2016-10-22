from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tensorflow as tf
from os import listdir
from os.path import isfile, join

# Data loading and preprocessing
#import tflearn.datasets.mnist as mnist
#X, Y, testX, testY = mnist.load_data(one_hot=True)
#X = X.reshape([-1, 28, 28, 1])
#testX = testX.reshape([-1, 28, 28, 1])

image_path = "/mnt/c/temp/resizedRaspberies/"

#list all filenames in image_path directory
filenames = [f for f in listdir(image_path) if isfile(join(image_path, f))]

#prefix the filenames with image_path
filenames_full_path = [image_path + filename for filename in filenames]

#read images to a tensor
image_queue = tf.train.string_input_producer(filenames_full_path)
reader = tf.WholeFileReader()
key, value = reader.read(image_queue)
X = tf.image.decode_jpeg(value, channels=3)
Y = tf.constant(1.0, shape=[1, 217])

# Building convolutional network
network = input_data(shape=[None, 80, 80, 3], name='input')
network = conv_2d(network, 100, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 200, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 300, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 400, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 1, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': X}, {'target': Y}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

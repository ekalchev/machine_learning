from __future__ import division, print_function, absolute_import

import tflearn
import fruit_recognition_model

from tflearn.data_utils import build_hdf5_image_dataset
import h5py
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

root_directory = '/home/emooo/PycharmProjects/machine_learning/'
image_directory = root_directory + 'dataset/train/'
dataset_file = root_directory + 'trained_data/train_dataset.h5'

imageWidth = 80
imageHeight = 80
numClasses = 2

print("Fruit Recognition. Emil Kalchev 2016")
print("Data set file: ",dataset_file)
print("Image directory: ",image_directory)

build_hdf5_image_dataset(image_directory, image_shape=(imageWidth, imageHeight), mode='folder', output_path=dataset_file, categorical_labels=True, normalize=True)


h5f = h5py.File(dataset_file, 'r')
X = h5f['X']
Y = h5f['Y']

network = fruit_recognition_model.getEkalchevNet(imageWidth, imageHeight, numClasses)

# Training
model = tflearn.DNN(network, checkpoint_path='fruit_recognition',
                    max_checkpoints=1, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=1000, validation_set=0.2, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=50,
          snapshot_epoch=False, run_id='fruit_recognition_convnet')

model.save(root_directory + "fruit_recognition_model.tfl")
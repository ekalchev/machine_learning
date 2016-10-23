from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from os import listdir
from os.path import isfile, join
from tflearn.layers.normalization import local_response_normalization
import ntpath

def load_images(image_path):
    #list all filenames in image_path directory
    filenames = [f for f in listdir(image_path) if isfile(join(image_path, f))]

    num_images = len(filenames)
    #prefix the filenames with image_path
    filenames_full_path = [image_path + filename for filename in filenames]
    image_queue = tf.train.string_input_producer(filenames_full_path)
    reader = tf.WholeFileReader()
    key, value = reader.read(image_queue)

    images = tf.image.decode_jpeg(value, channels=3)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    imageList = []
    labels = []

    for _ in range(num_images): #length of your filename list
      path = sess.run(key)
      print(path)
      firstLetterOfImageFile = ntpath.basename(path)[:1]
      if(firstLetterOfImageFile == b'd'):
            labels.append([0, 1])
      else:
            labels.append([1, 0])

      currentImage = sess.run(images)
      imageList.append(currentImage)

    X = sess.run(tf.pack(imageList));
    print(X.shape)

    Y = sess.run(tf.pack(labels));

    coord.request_stop()
    coord.join(threads)

    del images
    del imageList
    del labels

    sess.close()
    return X,Y, num_images

X,Y,_ = load_images("/home/emooo/PycharmProjects/machine_learning/alexnet/train/")
Xtest,Ytest,_ = load_images("/home/emooo/PycharmProjects/machine_learning/alexnet/test/")

print(Y)
print(Ytest)

network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)


model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')

print(model.predict(Xtest))
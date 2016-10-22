import tensorflow as tf
from os import listdir
from os.path import isfile, join

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

model = tf.initialize_all_variables()

sess = tf.Session()

sess.run(model)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

images_tensor = sess.run(X)
print(sess.run(key))

coord.request_stop()
coord.join(threads)

sess.close()


    
#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#print(sess.run(hello))
#a = tf.constant(10)
#b = tf.constant(32)
#print(sess.run(a + b))
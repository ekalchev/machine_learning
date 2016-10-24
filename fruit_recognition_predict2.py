import fruit_recognition_model
import tflearn
from tflearn.data_utils import build_hdf5_image_dataset
import h5py
import time

root_directory = '/home/emooo/PycharmProjects/machine_learning/'
image_directory = root_directory + 'dataset/test/'
dataset_file = root_directory + 'trained_data/predict_dataset.h5'

imageWidth = 80
imageHeight = 80
numClasses = 2

network = fruit_recognition_model.getEkalchevNet(imageWidth, imageHeight, numClasses)
model = tflearn.DNN(network)

model.load("_fruit_recognition_model.tfl")
print("Predicted values:")

build_hdf5_image_dataset(image_directory, image_shape=(imageWidth, imageHeight), mode='folder', output_path=dataset_file, categorical_labels=True, normalize=True)


h5f = h5py.File(dataset_file, 'r')
X = h5f['X']

print("Predict started")
start = time.perf_counter()
print(model.predict(X))
end = time.perf_counter()
print("Predict completed", end - start, "sec")

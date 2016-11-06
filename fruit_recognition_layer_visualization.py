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

#plot hidden layers

from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

import numpy.ma as ma
def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic



W_visu = model.get_weights(conv1.W) # Here share the weights you want, for example your layer conv1
W_visu = np.squeeze(W_visu)
pl.figure(figsize=(15, 15))
pl.title('conv1 weights')
nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary)

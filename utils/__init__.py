import matplotlib.pyplot as plt
import numpy as np
import keras
import cv2 as cv

from theano import function, config, shared, tensor
import time, os

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def print_env():
    vlen = 10 * 30 * 768
    iters = 1000
    rng = np.random.RandomState(22)
    x = shared(np.asarray(rng.rand(vlen), config.floatX))
    f = function([], tensor.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print('Looping %d times took %f seconds' % (iters, t1 - t0))
    print('Result is %s' % (r, ))
    if np.any([isinstance(x.op, tensor.Elemwise) and
                       ('Gpu' not in type(x.op).__name__)
               for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')

def showImages(image, mask=None, predict=None):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.figure()
    if mask is not None:
        plt.imshow(mask, cmap=plt.cm.gray)
        plt.title('the ground truth')
    if predict is not None:
        plt.figure()
        plt.imshow(predict, cmap=plt.cm.gray)
        plt.title('the predicted')

def norm_images(images):
    images = images.astype('float32')
    for i, image in enumerate(images):
        mean = np.mean(image)  # mean for data centering
        std = np.std(image)  # std for data normalization

        image -= mean
        image /= std
        images[i] = image

    return images

def threshold(images, min_value, max_value):
    images[images < min_value] = 0
    images[images > max_value] = 0
    images = (images - min_value) / (max_value - min_value)
    return images

def equalizeHist(images):
    for i, image in enumerate(images):
        image = image * 255
        image = image.astype(np.ubyte)
        images[i, :, :, 0] = cv.equalizeHist(image[:, :, 0])
    return images

def to_category(mask):
    mask[np.logical_and(0 <= mask, mask < 0.5)] = 0
    mask[np.logical_and(0.5 <= mask, mask < 1.5)] = 1
    mask[np.logical_and(1.5 <= mask, mask < 2)] = 2
    return mask

def back2noscale(boxs, liver_mask):
    output = np.zeros_like(liver_mask)
    for i, box in enumerate(boxs):
        min_row, min_col, max_row, max_col = box
        if max_row > 0 and max_col > 0:
            mask = cv.resize(liver_mask[i, :, :, 0], (max_col-min_col, max_row-min_row))
            output[i, min_row: max_row, min_col:max_col, 0] = to_category(mask)
    return output


def remove_bg(images, masks):
    ob_index = []
    for i, (image, mask) in enumerate(zip(images, masks)):
        if np.sum(mask) <= 0: continue
        ob_index.append(i)
    total = len(ob_index)
    images_new = np.zeros((total, images.shape[1], images.shape[2], 1), dtype=images.dtype)
    masks_new = np.zeros((total, images.shape[1], images.shape[2], 1), dtype=images.dtype)
    for i, index in enumerate(ob_index):
        images_new[i] = images[index]
        masks_new[i] = masks[index]
    return images_new, masks_new

def cvt1dTo3d(X_train):
    dims = X_train.shape
    X_train_tmp = np.zeros([dims[0], dims[1], dims[2], 3])
    X_train_tmp[:, :, :, 0] = X_train[:, :, :, 0]
    X_train_tmp[:, :, :, 1] = X_train[:, :, :, 0]
    X_train_tmp[:, :, :, 2] = X_train[:, :, :, 0]
    return X_train_tmp

def cvtSecond2HMS(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '%02d:%02d:%02d' % (h, m, s)

def add_backend_name(images_npy, masks_npy):
    images_npy_parts = images_npy.split('.')
    masks_npy_parts = masks_npy.split('.')
    if keras.backend.image_data_format() == 'channels_last':
        images_npy = images_npy_parts[0] + '_tf.' + images_npy_parts[1]
        masks_npy = masks_npy_parts[0] + '_tf.' + masks_npy_parts[1]
    if keras.backend.image_data_format() == 'channels_first':
        images_npy = images_npy_parts[0] + '_th.' + images_npy_parts[1]
        masks_npy = masks_npy_parts[0] + '_th.' + masks_npy_parts[1]
    return (images_npy, masks_npy)

def mkdirInCache(file_path):
    parentdir = os.path.abspath(os.path.join(file_path, os.pardir))
    grand_parent_dir = os.path.abspath(os.path.join(parentdir, os.pardir))
    if not os.path.exists(grand_parent_dir):
        os.makedirs(grand_parent_dir)
    if not os.path.exists(parentdir):
        os.makedirs(parentdir)

def elastic_transform_keras(image):
    alpha, sigma = 34, 4
    random_state = np.random.RandomState(None)

    shape = image.shape[:-1]
    dx = gaussian_filter(random_state.rand(*shape) * 2 - 1, sigma) * alpha
    dy = gaussian_filter(random_state.rand(*shape) * 2 - 1, sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    res_x = map_coordinates(image[:, :, 0], indices, order=1, mode='reflect').reshape(image.shape)

    return res_x

# def elastic_transform(image, mask, alpha, sigma, alpha_affine=None, random_state=None):
#     def f(image):
#         if random_state is None:
#             random_state = np.random.RandomState(None)
#
#         shape = image.shape
#
#         dx = gaussian_filter(random_state.rand(*shape) * 2 - 1, sigma) * alpha
#         dy = gaussian_filter(random_state.rand(*shape) * 2 - 1, sigma) * alpha
#
#         x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
#         indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
#
#         res_x = map_coordinates(image, indices, order=1, mode='reflect').shape(shape)
#         res_y = map_coordinates(mask, indices, order=1, mode='reflect').shape(shape)
#         return res_x, res_y
#     return f





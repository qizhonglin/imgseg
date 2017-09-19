from utils import add_backend_name, norm_images, mkdirInCache
from sklearn.cross_validation import train_test_split

import keras
import numpy as np
import cv2

class DataSetImage(object):
    def __init__(self,
                 image_row, image_col, dtype,
                 images_file, masks_file,
                 images_npy, masks_npy
                 ):
        self.image_row, self.image_col = image_row, image_col
        self.dtype = dtype

        self.images_file, self.masks_file = images_file, masks_file

        self.images_npy, self.masks_npy = images_npy, masks_npy
        self.__add_size_name()
        self.images_npy, self.masks_npy = add_backend_name(self.images_npy, self.masks_npy)
        mkdirInCache(self.images_npy)

        self.images = None
        self.masks = None

    def load(self):
        print('-'*30)
        print('Loading images and mask...')
        print('-'*30)
        self.__load_from_files()

        print('-' * 30)
        print('Preprocessing images and mask...')
        print('-' * 30)
        self.images = norm_images(self.images)
        self.masks = self.masks.astype('float32')
        self.masks /= 255.  # scale masks to [0, 1]
        print('Done: {0} images'.format(self.images.shape[0]))

        self.save_as_npy()

        return self.images, self.masks

    def _read_image(self, filename):
        img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        return img_gray

    def save_as_npy(self):
        np.save(self.images_npy, self.images)
        np.save(self.masks_npy, self.masks)
        print('Saving to {0} and {1} files done.'.format(self.images_npy, self.masks_npy))

    @staticmethod
    def load_from_npy(images_npy, masks_npy, test_size=.2):
        images = np.load(images_npy)
        masks = np.load(masks_npy)
        X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=test_size, random_state=1)
        return X_train, X_test, y_train, y_test

    def __load_from_files(self):
        total = len(self.images_file)
        if keras.backend.image_data_format() == 'channels_last':
            self.images = np.ndarray((total, self.image_row, self.image_col, 1), dtype=self.dtype)
            self.masks = np.ndarray((total, self.image_row, self.image_col, 1), dtype=self.dtype)
        if keras.backend.image_data_format() == 'channels_first':
            self.images = np.ndarray((total, 1, self.image_row, self.image_col), dtype=self.dtype)
            self.masks = np.ndarray((total, 1, self.image_row, self.image_col), dtype=self.dtype)

        for i, image_name in enumerate(self.images_file):
            print(image_name)
            if keras.backend.image_data_format() == 'channels_last':
                image = self._read_image(image_name)
                mask = self._read_image(self.masks_file[i])

                self.images[i, :, :, 0] = self.__resize(image)
                self.masks[i, :, :, 0] = self.__resize(mask)
            if keras.backend.image_data_format() == 'channels_first':
                image = self._read_image(image_name)
                mask = self._read_image(self.masks_file[i])

                self.images[i, 0, :, :] = self.__resize(image)
                self.masks[i, 0, :, :] = self.__resize(mask)
        print('Done: {0} cases with {1} images'.format(i + 1, self.images.shape[0]))

    def __resize(self, img_gray):
        return cv2.resize(img_gray, (self.image_col, self.image_row))

    def __add_size_name(self):
        size_name = '_' + str(self.image_row) + '_' + str(self.image_col)
        images_npy_parts = self.images_npy.split('.')
        masks_npy_parts = self.masks_npy.split('.')
        self.images_npy = images_npy_parts[0] + size_name + '.' + images_npy_parts[1]
        self.masks_npy = masks_npy_parts[0] + size_name + '.' + masks_npy_parts[1]
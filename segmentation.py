from __future__ import print_function

from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras

import matplotlib.pyplot as plt

from pprint import pprint
import time
import random

from keras.preprocessing.image import ImageDataGenerator
from itertools import izip

from sklearn.utils import class_weight

from utils.ImageModel import *
from utils import showImages, print_env, cvtSecond2HMS, mkdirInCache, elastic_transform_keras, back2noscale
from utils.metrics import Metrics
from SkinData import SkinData
from DataSetVolumn import DataSetVolumn, TumorVolumn
from BreastData import BreastData

from utils import to_category
import cv2 as cv
from viewer import viewSequence

seed = 9001
random.seed(seed)

class Segmentation(object):
    def __init__(self, model_name='unet', modelcheckpoint='cache/skin/model/unet.hdf5'):
        self.model_name = model_name
        self.model = None

        mkdirInCache(modelcheckpoint)
        self.modelcheckpoint = Segmentation.absolute_path(modelcheckpoint)

    def train(self, images, mask, validation_data, batch_size=32, nb_epoch=500, is_datagen=False, weights_path=None):
        print('-' * 30)
        print('Creating model...')
        print('-' * 30)
        model = globals()[self.model_name](images.shape[1:]).resume_model(weights_path, self.modelcheckpoint)

        print('-' * 30)
        print('compiling model...')
        print('-' * 30)
        model = ImageModel.compile(model)

        callbacks = [
            ModelCheckpoint(self.modelcheckpoint, monitor='val_loss', verbose=1, save_best_only=True),
            EarlyStopping(monitor='val_loss', patience=50, verbose=1)
        ]

        print('-' * 30)
        print('Fitting model...')
        print('-' * 30)
        if not is_datagen:
            model.fit(images, mask, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                  validation_data=validation_data, callbacks=callbacks)
        else:
            data_gen_args = dict(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 rotation_range=90.0,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 preprocessing_function=elastic_transform_keras)
            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)
            image_datagen.fit(images, augment=True, seed=seed)
            mask_datagen.fit(mask, augment=True, seed=seed)

            image_generator = image_datagen.flow(images, batch_size=batch_size, seed=seed)
            mask_generator = mask_datagen.flow(mask, batch_size=batch_size, seed=seed)
            train_generator = izip(image_generator, mask_generator)

            model.fit_generator(train_generator, steps_per_epoch=len(images)/batch_size, epochs=nb_epoch,
                  validation_data=validation_data, callbacks=callbacks)

        self.model = model

    def predict(self, images, batch_size=32):
        model = globals()[self.model_name](images.shape[1:]).resume_model(None, self.modelcheckpoint)
        self.model = ImageModel.compile(model)
        return self.model.predict(images, verbose=1, batch_size=batch_size)

    @staticmethod
    def absolute_path(model_pretrain):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.expanduser(os.path.join(current_dir, model_pretrain)) if model_pretrain else None
        return weights_path

    @staticmethod
    def run_skin(istrain, model_name, modelcheckpoint, model_pretrain=None,
            batch_size=32, nb_epoch=500, is_datagen=False):
        print('-' * 30)
        print('Loading data...')
        print('-' * 30)
        X_train, X_test, y_train, y_test = SkinData.load_from_npy(
            images_npy='cache/skin/datasets/images_224_224_tf.npy',
            masks_npy='cache/skin/datasets/masks_224_224_tf.npy')

        seg = Segmentation(model_name, modelcheckpoint)

        if istrain:
            seg.train(X_train, y_train, (X_test, y_test),
                      weights_path=Segmentation.absolute_path(model_pretrain),
                      batch_size=batch_size, nb_epoch=nb_epoch, is_datagen=is_datagen)

        predicts = seg.predict(X_test, batch_size=batch_size)
        pprint(Metrics.all(y_test, predicts))

        return (X_test, y_test, predicts)

    @staticmethod
    def run_breast(istrain, model_name, modelcheckpoint, model_pretrain=None,
            batch_size=32, nb_epoch=500, is_datagen=False,
            images_npy='cache/breast/datasets/images_crop_224_224_tf.npy',
            masks_npy='cache/breast/datasets/masks_crop_224_224_tf.npy'):
        print('-' * 30)
        print('Loading data...')
        print('-' * 30)
        X_train, X_test, y_train, y_test = BreastData.load_from_npy(images_npy, masks_npy, test_size=.1)
        print('Done Loading train with shape {0} and test with shape {1}'.format(X_train.shape, X_test.shape))

        # shape = y_train.shape
        # y_flat = np.reshape(y_train, shape[0]*shape[1]*shape[2]*shape[3]).astype('uint8')
        # maxvalue, minvalue = np.max(y_flat), np.min(y_flat)
        # cw = class_weight.compute_class_weight('balanced', np.unique(y_flat), y_flat)

        seg = Segmentation(model_name, modelcheckpoint)

        if istrain:
            seg.train(X_train, y_train, (X_test, y_test),
                      weights_path=Segmentation.absolute_path(model_pretrain),
                      batch_size=batch_size, nb_epoch=nb_epoch, is_datagen=is_datagen)

        predicts = seg.predict(X_test, batch_size=batch_size)
        pprint(Metrics.all(y_test, predicts))

        return (X_test, y_test, predicts)

    @staticmethod
    def run_liver(istrain, model_name, modelcheckpoint, model_pretrain=None,
            batch_size=1, nb_epoch=500, is_datagen=False, isliver=True):

        reader = DataSetVolumn() if isliver else TumorVolumn()
        seg = Segmentation(model_name, modelcheckpoint)

        if istrain:
            X_train, y_train = reader.load_traindata()
            seg.train(X_train, y_train, reader.validation_data(),
                      weights_path=Segmentation.absolute_path(model_pretrain),
                      batch_size=batch_size, nb_epoch=nb_epoch, is_datagen=is_datagen)

        testdata = reader.load_testdata()

        metrics_testdata = []
        for imagefile, data in testdata.iteritems():
            X_test, y_test = data
            predicts = seg.predict(X_test, batch_size=batch_size)

            def save(liver):
                volumefile = 'volumn-' + imagefile.split('-')[1] + '.npy'
                maskfile = 'segmentation-' + imagefile.split('-')[1] + '.npy'
                predictfile = 'predict-' + imagefile.split('-')[1] + '.npy'
                np.save(os.path.join('cache/{0}/result'.format(liver), volumefile), X_test)
                np.save(os.path.join('cache/{0}/result'.format(liver), maskfile), y_test)
                np.save(os.path.join('cache/{0}/result'.format(liver), predictfile), predicts)
            # save('liver') if isliver else save('tumor')

            def tumor2noscale(imagefile, predicts):
                if not isliver:
                    boxs, images, masks = reader.get_data(imagefile)
                    masks[masks < 2] = 0
                    masks /= 2
                    imagesize = (512, 512)
                    predicts_new = np.zeros((predicts.shape[0], imagesize[0], imagesize[1], 1), dtype=predicts.dtype)
                    for i, predict in enumerate(predicts):
                        predicts_new[i, :, :, 0] = cv.resize(predict[:, :, 0], imagesize)
                    predicts_new = to_category(predicts_new)
                    predicts_new = back2noscale(boxs, predicts_new)
                    # viewSequence(masks)
                    # viewSequence(predicts_new)
                    y_test, predicts = masks, predicts_new
            tumor2noscale(imagefile, predicts)
            pprint(Metrics.all(y_test, predicts))
            metrics_testdata.append((imagefile, Metrics.all(y_test, predicts)))

        result = {
            'acc': sum([metrics['acc'] for imagefile, metrics in metrics_testdata])/len(metrics_testdata),
            'dice': sum([metrics['dice'] for imagefile, metrics in metrics_testdata]) / len(metrics_testdata),
            'jacc': sum([metrics['jacc'] for imagefile, metrics in metrics_testdata]) / len(metrics_testdata),
            'sensitivity': sum([metrics['sensitivity'] for imagefile, metrics in metrics_testdata]) / len(metrics_testdata),
            'specificity': sum([metrics['specificity'] for imagefile, metrics in metrics_testdata]) / len(metrics_testdata)
        }
        print('the average metrics case by case')
        pprint(result)

        return (X_test, y_test, predicts)




class SegmentationBatch(Segmentation):
    def __init__(self, model_name, modelcheckpoint):
        super(SegmentationBatch, self).__init__(model_name, modelcheckpoint)

    def train(self, data, batch_size=32, nb_epoch=500, is_datagen=False, weights_path=None, channel=5):
        image_size = (512, 512, channel)
        print('-' * 30)
        print('Creating model...')
        print('-' * 30)
        model = globals()[self.model_name](image_size).resume_model(weights_path, self.modelcheckpoint)

        print('-' * 30)
        print('compiling model...')
        print('-' * 30)
        model = ImageModel.compile(model)

        callbacks = [
            ModelCheckpoint(self.modelcheckpoint, monitor='val_loss', verbose=1, save_best_only=True),
            EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        ]

        print('-' * 30)
        print('Fitting model...')
        print('-' * 30)
        data = DataSetVolumn()
        validation_data = data.validation_data_channel(channel)
        _, file_dict = data.flow_all_memory_num(0.7, channel)
        for loop in range(nb_epoch):
            print('*'*30 + 'The loop/nb_epoch is {0}/{1}'.format(loop, nb_epoch) + '*'*30)
            print('*' * 70)
            for i in range(len(file_dict)):
                images, masks = data.flow_all_memory(i, file_dict, channel)
                model.fit(images, masks, batch_size=batch_size, nb_epoch=1, verbose=1, shuffle=True,
                      validation_data=validation_data, callbacks=callbacks)

        self.model = model

    @staticmethod
    def run_liver(
            istrain=False,
            model_name='unet',
            modelcheckpoint='cache/liver/model/unet.hdf5',
            model_pretrain='cache/liver/model/weight_unet_gen_tf.h5',
            batch_size=1, nb_epoch=500, is_datagen=False, channel=5):

        reader = DataSetVolumn()
        seg = SegmentationBatch(model_name, modelcheckpoint)

        if istrain:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            weights_path = os.path.expanduser(os.path.join(current_dir, model_pretrain)) if model_pretrain else None
            seg.train(reader, weights_path=weights_path,
                      batch_size=batch_size, nb_epoch=nb_epoch, is_datagen=is_datagen, channel=channel)

        testdata = reader.load_testdata_channel(channel)

        metrics_testdata = []
        for imagefile, data in testdata.iteritems():
            X_test, y_test = data
            predicts = seg.predict(X_test, batch_size=batch_size)
            pprint(Metrics.all(y_test, predicts))
            metrics_testdata.append((imagefile, Metrics.all(y_test, predicts)))

        result = {
            'acc': sum([metrics['acc'] for imagefile, metrics in metrics_testdata]) / len(metrics_testdata),
            'dice': sum([metrics['dice'] for imagefile, metrics in metrics_testdata]) / len(metrics_testdata),
            'jacc': sum([metrics['jacc'] for imagefile, metrics in metrics_testdata]) / len(metrics_testdata),
            'sensitivity': sum([metrics['sensitivity'] for imagefile, metrics in metrics_testdata]) / len(metrics_testdata),
            'specificity': sum([metrics['specificity'] for imagefile, metrics in metrics_testdata]) / len(metrics_testdata)
        }
        print('the average metrics case by case')
        pprint(result)

        return (X_test, y_test, predicts)


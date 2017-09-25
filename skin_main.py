from __future__ import print_function


from utils import showImages, cvtSecond2HMS
import time
import random
import matplotlib.pyplot as plt

from segmentation import Segmentation

seed = 9001
random.seed(seed)


def run_unet(istrain=False):
    return Segmentation.run_skin(istrain=istrain,
            model_name='unet',
            modelcheckpoint='cache/skin/model/unet.hdf5',
            batch_size=32)

def run_FCN(istrain=False):
    return Segmentation.run_skin(istrain=istrain,
                            model_name='FCN',
                            modelcheckpoint='cache/skin/model/fcn.hdf5',
                            batch_size=32)

def run_ResNet50(istrain=False):
    return Segmentation.run_skin(istrain=istrain,
                            model_name='ResNet50',
                            modelcheckpoint='cache/skin/model/resnet50.hdf5',
                            batch_size=32)

def run_DenseNet(istrain=False):
    return Segmentation.run_skin(istrain=istrain,
                            model_name='DenseNet',
                            modelcheckpoint='cache/skin/model/densenet.hdf5',
                            batch_size=4)

def run_unet_gen(istrain=False):
    return Segmentation.run_skin(istrain=istrain,
                            model_name='unet',
                            modelcheckpoint='cache/skin/model/unet_gen.hdf5',
                            is_datagen=True)

def run_unet_standard_gen(istrain=False):
    return Segmentation.run_skin(istrain=istrain,
                            model_name='unet_standard',
                            modelcheckpoint='cache/skin/model/unet_standard_gen.hdf5',
                            batch_size=8, nb_epoch=1000, is_datagen=True)


if __name__ == '__main__':
    # print_env()

    ts = time.clock()

    (X_test, y_test, predicts) = run_unet(istrain=True)                             # dice = 84.9       85.8
    # (X_test, y_test, predicts) = run_FCN(istrain=True)                  # dice = 71.4
    # (X_test, y_test, predicts) = run_ResNet50(istrain=False)            # dice = 80
    # (X_test, y_test, predicts) = run_DenseNet(istrain=True)               # dice = 82.3

    # unet + data augment
    # (X_test, y_test, predicts) = run_unet_gen(istrain=True)           # dice= 86.8

    # (X_test, y_test, predicts) = run_unet_standard_gen(istrain=True)    # dice = 87.4

    print("total process time: %s" % cvtSecond2HMS(time.clock() - ts))

    # for i in range(0, 1):
    #     showImages(X_test[i, :, :, 0], y_test[i, :, :, 0], predicts[i, :, :, 0])
    #
    # # for (image, mask, predict) in zip(X_test, y_test, predicts):
    # #     seg.show(image[0, :, :], mask[0, :, :], predict[0, :, :])
    #
    # plt.show()


from __future__ import print_function


from utils import showImages, cvtSecond2HMS
import time
import random
import matplotlib.pyplot as plt

from segmentation import Segmentation

seed = 9001
random.seed(seed)


def run_unet_gen(istrain=False):
    return Segmentation.run_breast(istrain=istrain,
                            model_name='unet',
                            modelcheckpoint='cache/breast/model/unet_gen.hdf5',
                            is_datagen=True)

def run_unet_gen_448_448(istrain=False):
    return Segmentation.run_breast(istrain=istrain,
                            model_name='unet5',
                            modelcheckpoint='cache/breast/model/unet_gen_448_448.hdf5',
                            batch_size=16, nb_epoch=500, is_datagen=True,
                            images_npy='cache/breast/datasets/images_pad_crop_448_448_tf.npy',
                            masks_npy='cache/breast/datasets/masks_pad_crop_448_448_tf.npy')

def run_unet_gen_448_448_echo(istrain=False):
    return Segmentation.run_breast(istrain=istrain,
                            model_name='unet5',
                            modelcheckpoint='cache/breast/model/unet_gen_448_448_echo.hdf5',
                            batch_size=16,
                            is_datagen=True,
                            images_npy='cache/breast/datasets/images_echo_448_448_tf.npy',
                            masks_npy='cache/breast/datasets/masks_echo_448_448_tf.npy')

def run_unet_gen_448_448_padecho(istrain=False):
    return Segmentation.run_breast(istrain=istrain,
                            model_name='unet5',
                            modelcheckpoint='cache/breast/model/unet_gen_448_448_padecho.hdf5',
                            batch_size=16,
                            is_datagen=True,
                            images_npy='cache/breast/datasets/images_pad_echo_448_448_tf.npy',
                            masks_npy='cache/breast/datasets/masks_pad_echo_448_448_tf.npy')

if __name__ == '__main__':
    # print_env()

    ts = time.clock()

    # (X_test, y_test, predicts) = run_unet_gen(istrain=False)  # dice = 86.6

    (X_test, y_test, predicts) = run_unet_gen_448_448(istrain=True)  # dice = 84.7

    # (X_test, y_test, predicts) = run_unet_gen_448_448_echo(istrain=False)  # dice = 80.25

    # (X_test, y_test, predicts) = run_unet_gen_448_448_padecho(istrain=False)  # dice = 80.1

    print("total process time: %s" % cvtSecond2HMS(time.clock() - ts))
    #
    # for i in range(0, 1):
    #     showImages(X_test[i, :, :, 0], y_test[i, :, :, 0], predicts[i, :, :, 0])
    #
    # # for (image, mask, predict) in zip(X_test, y_test, predicts):
    # #     seg.show(image[0, :, :], mask[0, :, :], predict[0, :, :])
    #
    # plt.show()

import random, time

from segmentation import Segmentation, SegmentationBatch
from utils import cvtSecond2HMS

seed = 9001
random.seed(seed)


def run_unet(istrain=False):
    return Segmentation.run_liver(istrain=istrain,
                                  model_name='unet',
                                  modelcheckpoint='cache/liver/model/unet.hdf5',
                                  batch_size=8,
                                  nb_epoch=20)

def run_unet_25D(istrain=False):
    return SegmentationBatch.run_liver(istrain=istrain,
                                       model_name='unet',
                                       modelcheckpoint='cache/liver/model/unet_25D.hdf5',
                                       batch_size=8,
                                       nb_epoch=20,
                                       channel=5)

if __name__ == '__main__':
    # print_env()


    ts = time.clock()

    # (X_test, y_test, predicts) = run_unet(istrain=False)       # dice = 96.1
    (X_test, y_test, predicts) = run_unet_25D(istrain=True)  # dice = 90.5

    print("total process time: ", cvtSecond2HMS(time.clock() - ts))
    #
    # for i in range(0, 1):
    #     if keras.backend.image_data_format() == 'channels_last':
    #         showImages(X_test[i, :, :, 0], y_test[i, :, :, 0], predicts[i, :, :, 0])
    #     if keras.backend.image_data_format() == 'channels_first':
    #         showImages(X_test[i, 0, :, :], y_test[i, 0, :, :], predicts[i, 0, :, :])
    # # for (image, mask, predict) in zip(X_test, y_test, predicts):
    # #     seg.show(image[0, :, :], mask[0, :, :], predict[0, :, :])
    #
    # plt.show()
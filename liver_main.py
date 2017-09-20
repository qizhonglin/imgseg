import random, time

from segmentation import Segmentation, SegmentationBatch
from utils import cvtSecond2HMS

seed = 9001
random.seed(seed)


def run_unet(istrain=False, tissue='liver'):
    return Segmentation.run_liver(istrain=istrain,
                                  model_name='unet',
                                  modelcheckpoint='cache/{0}/model/unet.hdf5'.format(tissue),
                                  batch_size=8,
                                  nb_epoch=20,
                                  isliver=True if tissue=='liver' else False)

def run_unet_25D(istrain=False):
    return SegmentationBatch.run_liver(istrain=istrain,
                                       model_name='unet',
                                       modelcheckpoint='cache/liver/model/unet_25D.hdf5',
                                       batch_size=8,
                                       nb_epoch=20,
                                       channel=5)

def run_unet_reg_25D(istrain=False):
    return SegmentationBatch.run_liver(istrain=istrain,
                                       model_name='unet_reg',
                                       modelcheckpoint='cache/liver/model/unet_reg_25D.hdf5',
                                       batch_size=4,
                                       nb_epoch=20,
                                       channel=5)

def run_unet_standard_25D(istrain=False):
    return SegmentationBatch.run_liver(istrain=istrain,
                                       model_name='unet_standard',
                                       modelcheckpoint='cache/liver/model/unet_standard_25D.hdf5',
                                       batch_size=4,
                                       nb_epoch=20,
                                       channel=5)

if __name__ == '__main__':
    # print_env()


    ts = time.clock()

    # (X_test, y_test, predicts) = run_unet(istrain=True)       # dice = 96.1
    # (X_test, y_test, predicts) = run_unet_25D(istrain=True)  # dice = 90.5
    # (X_test, y_test, predicts) = run_unet_reg_25D(istrain=True)  # dice = 89.6
    # (X_test, y_test, predicts) = run_unet_standard_25D(istrain=False)  # dice = 88.7

    (X_test, y_test, predicts) = run_unet(istrain=True, tissue='tumor')  # dice =

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

import os

from DataSetImage import DataSetImage


'''
SkinData is to load images and masks from images dir and masks dir
and save to images_H_W_tf.npy as numpy.array with shape(sample_num, H, W, 1)
 
example:
# read *.jpg within dir and save to npy
ds = SkinData(image_row=224, image_col=224,
             images_npy='cache/skin/datasets/images.npy', 
             masks_npy='cache/skin/datasets/masks.npy')
ds.load()

# load data from npy
X_train, X_test, y_train, y_test = SkinData.load_from_npy(
            images_npy='cache/skin/datasets/images.npy', 
             masks_npy='cache/skin/datasets/masks.npy')
'''
class SkinData(DataSetImage):
    def __init__(self, image_row=420, image_col=580, dtype='uint8',
                 images_npy='cache/skin/datasets/images.npy', masks_npy='cache/skin/datasets/masks.npy',
                 data_dir='/media/philips/New Volume/AI/datasets/ISIC_2017/Part1/ISIC-2017_Training_Data',
                 mask_dir='/media/philips/New Volume/AI/datasets/ISIC_2017/Part1/ISIC-2017_Training_Part1_GroundTruth'):
        images_file = [image_file for image_file in os.listdir(data_dir) if
                       'aug' not in image_file and '.jpg' in image_file]
        masks_file = [image_name[:-4] + '_segmentation.png' for image_name in images_file]
        # images_file = images_file[:2]
        # masks_file = masks_file[:2]

        images_file = [os.path.join(data_dir, filename) for filename in images_file]
        masks_file = [os.path.join(mask_dir, filename) for filename in masks_file]

        super(SkinData, self).__init__(image_row, image_col, dtype, images_file, masks_file, images_npy, masks_npy)

if __name__ == '__main__':
    # ds = SkinData(image_row=224, image_col=224,
    #                   images_npy='cache/skin/datasets/images1.npy',
    #                   masks_npy='cache/skin/datasets/masks1.npy')
    # images, masks = ds.load()
    # print(images.shape)
    X_train, X_test, y_train, y_test = SkinData.load_from_npy(
                        images_npy='cache/skin/datasets/images_224_224_tf.npy',
                        masks_npy='cache/skin/datasets/masks_224_224_tf.npy')
    print(X_train.shape)

from glob import glob
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

from DataSetImage import DataSetImage

class BreastData(DataSetImage):
    def __init__(self, image_row=224, image_col=224, dtype='float32',
                 images_npy='cache/breast/datasets/images_crop.npy', masks_npy='cache/breast/datasets/masks_crop.npy',
                 data_dir=r'/home/philips/.keras/datasets/breast/crop-images'):
        all_file = glob(data_dir + '/*.png')
        markers_file = [image_file for image_file in all_file if 'Marker' in image_file]
        masks_file = [image_file for image_file in all_file if 'Mask' in image_file]
        inpaints_file = [image_file for image_file in all_file if 'inpaint' in image_file]
        images_file = list(set(all_file) - set(markers_file) - set(masks_file) - set(inpaints_file))
        images_file.sort()
        masks_file.sort()
        markers_file.sort()
        inpaints_file.sort()

        self.inpaints_file = inpaints_file
        super(BreastData, self).__init__(image_row, image_col, dtype, inpaints_file, masks_file, images_npy, masks_npy)

    def describe(self):
        images_size = {}
        for i, image_name in enumerate(self.images_file):
            print(image_name)
            image = self._read_image(image_name)
            images_size[image_name] = image.shape
        pprint(images_size)

        row = [image_size[0] for filename, image_size in images_size.iteritems()]
        col = [image_size[1] for filename, image_size in images_size.iteritems()]
        ratio = [image_size[0]/float(image_size[1]) for filename, image_size in images_size.iteritems()]
        row = np.array(row)
        col = np.array(col)
        ratio = np.array(ratio)

        print('the mean of (Height, Width, Ratio) = ({0}, {1}, {2})'.format(np.mean(row), np.mean(col), np.mean(ratio)))
        print('the median of (Height, Width, Ratio) = ({0}, {1}, {2})'.format(np.median(row), np.median(col),
                                                                              np.median(ratio)))

        plt.plot(row)
        plt.title('image height')
        plt.figure()
        plt.plot(col)
        plt.title('image width')
        plt.figure()
        plt.plot(ratio)
        plt.title('image height vs width')

    def pad_images(self, dst_dir=r'/home/philips/.keras/datasets/breast/pad-crop-images'):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        self.__pad_images(self.masks_file, dst_dir)
        self.__pad_images(self.inpaints_file, dst_dir)

    def __pad_images(self, images_file, dst_dir):
        for i, image_name in enumerate(images_file):
            print(image_name)
            image = self._read_image(image_name)
            r, c = image.shape
            r_new = max(max(r, c), 448)
            image_pad = np.zeros((r_new, r_new), dtype='uint8')
            r_s = r_new / 2 - r / 2
            c_s = r_new / 2 - c / 2
            image_pad[r_s:r_s + r, c_s:c_s+c] = image

            filename = os.path.split(image_name)[-1]
            dst_file = os.path.join(dst_dir, filename)

            cv.imwrite(dst_file, image_pad)

    @staticmethod
    def inpaint(image_file, marker_file, is_preprocess):
        img_gray = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        marker = cv.imread(marker_file, cv.IMREAD_GRAYSCALE)

        if is_preprocess:
            marker_d = cv.dilate(marker, np.ones((7, 7)), iterations=1)
            marker_e = cv.erode(marker_d, np.ones((5, 5)), iterations=1)

        res = cv.inpaint(img_gray, marker_e, 3, cv.INPAINT_TELEA)

        # titles = ['src', 'marker', 'dilate', 'erode', 'inpaint']
        # images = [img_gray, marker, marker_d, marker_e, res]
        # showImagesOneByOne(titles, images)
        # plt.show()

        return img_gray, res

    @staticmethod
    def inpaint_all(breast, is_preprocess=False):
        all_file = glob(breast)
        markers_file = [image_file for image_file in all_file if 'Marker' in image_file]
        masks_file = [image_file for image_file in all_file if 'Mask' in image_file]
        inpaints_file = [image_file for image_file in all_file if 'inpaint' in image_file]
        images_file = list(set(all_file) - set(markers_file) - set(masks_file) - set(inpaints_file))
        images_file.sort()
        masks_file.sort()
        markers_file.sort()
        # assert len(images_file) == len(markers_file) == len(masks_file) == len(inpaints_file), 'impossible'

        for image_file, marker_file in zip(images_file, markers_file):
            img_gray, res = BreastData.inpaint(image_file, marker_file, is_preprocess)
            titles = ['src', 'inpaint']
            images = [img_gray, res]
            # showImagesOneByOne(titles, images)
            image_name = image_file.split('.png')[0]
            inpaint_file = image_name + '-inpaint.png'
            cv.imwrite(inpaint_file, res)



def run(image_row=224, image_col=224,
        images_npy='cache/breast/datasets/images_crop.npy',
        masks_npy='cache/breast/datasets/masks_crop.npy',
        data_dir=r'/home/philips/.keras/datasets/breast/crop-images'):
    ds = BreastData(image_row=image_row, image_col=image_col,
                    images_npy=images_npy, masks_npy=masks_npy,
                    data_dir=data_dir)
    images, masks = ds.load()
    print('all has {0}'.format(images.shape))

    def add_size_backend(images_npy):
        filename = os.path.split(images_npy)[-1]
        filename_new = filename.split('.')[0] + '_' + str(image_row) + '_' + str(image_col) + '_tf.npy'
        dst_file = os.path.join(os.path.split(images_npy)[0], filename_new)
        return dst_file

    X_train, X_test, y_train, y_test = BreastData.load_from_npy(
                        images_npy=add_size_backend(images_npy),
                        masks_npy=add_size_backend(masks_npy),
                        test_size=.1)
    print('train has {0}'.format(X_train.shape))

if __name__ == '__main__':
    # run()

    # BreastData().describe()

    # dst_dir = r'/home/philips/.keras/datasets/breast/pad-crop-images'
    # BreastData(data_dir=r'/home/philips/.keras/datasets/breast/crop-images').pad_images(dst_dir)
    # run(image_row=448, image_col=448,
    #     images_npy='cache/breast/datasets/images_pad_crop.npy',
    #     masks_npy='cache/breast/datasets/masks_pad_crop.npy',
    #     data_dir=dst_dir)

    # # BreastData.inpaint_all(breast=r'/home/philips/.keras/datasets/breast/echo-images/*.png', is_preprocess=True)
    # dst_dir = r'/home/philips/.keras/datasets/breast/pad-echo-images'
    # # BreastData(data_dir=r'/home/philips/.keras/datasets/breast/echo-images').pad_images(dst_dir)
    # run(image_row=448, image_col=448,
    #     images_npy='cache/breast/datasets/images_pad_echo.npy',
    #     masks_npy='cache/breast/datasets/masks_pad_echo.npy',
    #     data_dir=dst_dir)

    dst_dir = r'/home/philips/.keras/datasets/breast/echo-images'
    run(image_row=448, image_col=448,
        images_npy='cache/breast/datasets/images_echo.npy',
        masks_npy='cache/breast/datasets/masks_echo.npy',
        data_dir=dst_dir)

    plt.show()
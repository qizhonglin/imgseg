
import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import cv2 as cv

from DataSetVolumn import DataSetVolumn
from viewer import viewSequence, numpy2vtk
from viewer.MrViewer import MrViewer
import matplotlib.pyplot as plt
from skimage.measure import regionprops

class DataSetVolumnNii(DataSetVolumn):
    def __init__(self, dtype='float32',
                 data_dir='/home/qzlin/.keras/datasets/LiTS/raw/train'):
        super(DataSetVolumnNii, self).__init__(dtype, data_dir)

    def _get_image_file(self):
        self.images_file = [image_file for image_file in os.listdir(self.data_dir) if 'volume' in image_file]
        self.masks_file = [image_file for image_file in os.listdir(self.data_dir) if 'segmentation' in image_file]
        self.images_file.sort()
        self.masks_file.sort()
        print('image file is ', self.images_file)
        print('mask file is ', self.masks_file)

    def _load_image(self, image_path):
        data = sitk.ReadImage(image_path)
        nda = sitk.GetArrayFromImage(data)
        return nda

class VolumnCrop(object):
    def __init__(self, volumn_file, segmentation_file):
        self.volumn_file = volumn_file
        self.segmentation_file = segmentation_file

    def load(self):
        volumn = np.load(self.volumn_file)
        segmentation = np.load(self.segmentation_file)
        volumn = np.swapaxes(volumn, 1, 3)
        segmentation = np.swapaxes(segmentation, 1, 3)

        return volumn, segmentation

    def connected_component(self, segmentation):
        predict = np.copy(segmentation)
        thresh = .5
        predict[predict < thresh] = 0
        predict[predict >= thresh] = 1
        labeled_array, num_features = ndimage.label(predict)
        labeled_array = np.reshape(labeled_array, labeled_array.shape[:-1])
        progs = regionprops(labeled_array)
        max_index = np.argmax([prog['area'] for prog in progs])
        labeled_array[labeled_array == progs[max_index]['label']] = 1
        labeled_array[np.logical_not(labeled_array == progs[max_index]['label'])] = 0
        return labeled_array

    def bbox(self, labeled_array, margin=10):
        boxs = np.zeros((len(labeled_array), 4), dtype=np.int)
        for i, slice in enumerate(labeled_array):
            if np.sum(slice) > 0:
                prog = regionprops(slice)
                min_row, min_col, max_row, max_col = prog[0]['bbox']
                if max_row > 0 and max_col > 0:
                    min_row, min_col, max_row, max_col = min_row-margin, min_col-2*margin, max_row+margin, max_col+2*margin
                    boxs[i] = min_row, min_col, max_row, max_col
        return boxs

    def __crop_liver(self, boxs, volumn_preprocess, segmentation, new_size=(512, 512)):
        liver = np.zeros_like(volumn_preprocess)
        liver_mask = np.zeros_like(segmentation)
        for i, box in enumerate(boxs):
            min_row, min_col, max_row, max_col = box
            if max_row > 0 and max_col > 0:
                liver[i, :, :, 0] = cv.resize(volumn_preprocess[i, min_row: max_row, min_col:max_col, 0], new_size)
                mask = cv.resize(segmentation[i, min_row: max_row, min_col:max_col, 0], new_size)
                liver_mask[i, :, :, 0] = self.to_category(mask)
        return liver, liver_mask

    @staticmethod
    def to_category(mask):
        mask[np.logical_and(0 <= mask, mask < 0.5)] = 0
        mask[np.logical_and(0.5 <= mask, mask < 1.5)] = 1
        mask[np.logical_and(1.5 <= mask, mask < 2)] = 2
        return mask

    def back2noscale(self, boxs, liver_mask):
        output = np.zeros_like(liver_mask)
        for i, box in enumerate(boxs):
            min_row, min_col, max_row, max_col = box
            if max_row > 0 and max_col > 0:
                mask = cv.resize(liver_mask[i, :, :, 0], (max_col-min_col, max_row-min_row))
                output[i, min_row: max_row, min_col:max_col, 0] = self.to_category(mask)
        return output

    def crop_liver(self):
        volumn, segmentation = self.load()
        labeled_array = self.connected_component(segmentation)
        boxs = self.bbox(labeled_array)
        liver, liver_mask = self.__crop_liver(boxs, volumn, segmentation)
        # liver_range = (np.min(liver), np.max(liver))
        # liver_mask_range = (np.min(liver_mask), np.max(liver_mask))
        # VolumnCropTest(None).view(np.reshape(volumn, volumn.shape[:-1]),
        #                           np.reshape(segmentation, segmentation.shape[:-1]),
        #                           np.reshape(liver, liver.shape[:-1]),
        #                           np.reshape(liver_mask, liver_mask.shape[:-1]))
        return liver, liver_mask, boxs


class VolumnCropTest(object):
    def __init__(self, file):
        self.file = file

    def load(self):
        volumn, segmentation, volumn_preprocess, predict = self.file
        volumn = np.load(volumn)
        segmentation = np.load(segmentation)
        volumn = np.swapaxes(volumn, 1, 3)
        segmentation = np.swapaxes(segmentation, 1, 3)
        volumn_preprocess = np.load(volumn_preprocess)
        predict = np.load(predict)

        return volumn, volumn_preprocess, segmentation, predict

    def view(self, volumn, volumn_preprocess, segmentation, predict):
        datas = [numpy2vtk(volumn),
                 numpy2vtk(volumn_preprocess),
                 numpy2vtk(segmentation),
                 numpy2vtk(predict)]
        MrViewer(datas=datas).viewSlice()

    def connected_component(self, predict):
        thresh = .5
        predict[predict < thresh] = 0
        predict[predict >= thresh] = 1
        labeled_array, num_features = ndimage.label(predict)
        labeled_array = np.reshape(labeled_array, labeled_array.shape[:-1])
        progs = regionprops(labeled_array)
        max_index = np.argmax([prog['area'] for prog in progs])
        labeled_array[labeled_array == progs[max_index]['label']] = 1
        labeled_array[np.logical_not(labeled_array == progs[max_index]['label'])] = 0
        return labeled_array

    def bbox(self, labeled_array, margin=10):
        boxs = np.zeros((len(labeled_array), 4), dtype=np.int)
        for i, slice in enumerate(labeled_array):
            if np.sum(slice) > 0:
                prog = regionprops(slice)
                min_row, min_col, max_row, max_col = prog[0]['bbox']
                if max_row > 0 and max_col > 0:
                    min_row, min_col, max_row, max_col = min_row-margin, min_col-2*margin, max_row+margin, max_col+2*margin
                    boxs[i] = min_row, min_col, max_row, max_col
        return boxs

    def __crop_liver(self, boxs, volumn_preprocess, segmentation, new_size=(512, 512)):
        liver = np.zeros_like(volumn_preprocess)
        liver_mask = np.zeros_like(segmentation)
        for i, box in enumerate(boxs):
            min_row, min_col, max_row, max_col = box
            if max_row > 0 and max_col > 0:
                liver[i, :, :, 0] = cv.resize(volumn_preprocess[i, min_row: max_row, min_col:max_col, 0], new_size)
                mask = cv.resize(segmentation[i, min_row: max_row, min_col:max_col, 0], new_size)
                liver_mask[i, :, :, 0] = self.to_category(mask)
        return liver, liver_mask

    @staticmethod
    def to_category(mask):
        mask[np.logical_and(0 <= mask, mask < 0.5)] = 0
        mask[np.logical_and(0.5 <= mask, mask < 1.5)] = 1
        mask[np.logical_and(1.5 <= mask, mask < 2)] = 2
        return mask

    def back2noscale(self, boxs, liver_mask):
        output = np.zeros_like(liver_mask)
        for i, box in enumerate(boxs):
            min_row, min_col, max_row, max_col = box
            if max_row > 0 and max_col > 0:
                mask = cv.resize(liver_mask[i, :, :, 0], (max_col-min_col, max_row-min_row))
                output[i, min_row: max_row, min_col:max_col, 0] = self.to_category(mask)
        return output

    def crop_liver(self):
        volumn, volumn_preprocess, segmentation, predict = self.load()
        labeled_array = self.connected_component(predict)
        boxs = self.bbox(labeled_array)
        liver, liver_mask = self.__crop_liver(boxs, volumn_preprocess, segmentation)
        liver_range = (np.min(liver), np.max(liver))
        liver_mask_range = (np.min(liver_mask), np.max(liver_mask))
        self.view(volumn_preprocess, segmentation, liver, liver_mask)
        return liver, liver_mask, boxs

class Result(object):
    def __init__(self,
                 src_dir='/home/qzlin/.keras/datasets/LiTS/liver',
                 dst_dir='cache/liver/result'):
        self.src_dir = src_dir
        self.dst_dir = dst_dir

        self._get_image_file()
        self.__get_indexs()

    def _get_image_file(self):
        self.images_file = [image_file for image_file in os.listdir(self.dst_dir) if 'volumn' in image_file]
        self.masks_file = [image_file for image_file in os.listdir(self.dst_dir) if 'segmentation' in image_file]
        self.predicts_file = [image_file for image_file in os.listdir(self.dst_dir) if 'predict' in image_file]
        self.images_file.sort()
        self.masks_file.sort()
        self.predicts_file.sort()
        print('image file is ', self.images_file)
        print('mask file is ', self.masks_file)
        print('predict file is ', self.predicts_file)

    def __get_indexs(self):
        self.indexs = [imagefile.split('-')[1] for imagefile in self.images_file]
        self.indexs = [imagefile.split('.')[0] for imagefile in self.indexs]

    def __create_files(self):
        volumns = ['volume-' + index + '-mask.npy' for index in self.indexs]
        segmentations = ['segmentation-' + index + '-mask.npy' for index in self.indexs]
        volumn_preprocess = ['volumn-' + index + '.npy' for index in self.indexs]
        predict = ['predict-' + index + '.npy' for index in self.indexs]

        volumns = [os.path.join(self.src_dir, tmp) for tmp in volumns]
        segmentations = [os.path.join(self.src_dir, tmp) for tmp in segmentations]
        volumn_preprocess = [os.path.join(self.dst_dir, tmp) for tmp in volumn_preprocess]
        predict = [os.path.join(self.dst_dir, tmp) for tmp in predict]
        return volumns, segmentations, volumn_preprocess, predict

    def view(self, index):
        file = self.get_file(index)
        volumncrop = VolumnCropTest(file)
        volumn, volumn_preprocess, segmentation, predict = volumncrop.load()
        volumncrop.view(volumn, volumn_preprocess, segmentation, predict)

    def get_file(self, index):
        files = self.__create_files()
        index = index % len(files[0])
        file = (files[0][index], files[1][index], files[2][index], files[3][index])
        return file

    def view_connected_component(self, index):
        file = self.get_file(index)
        volumncrop = VolumnCropTest(file)
        liver, liver_mask, boxs = volumncrop.crop_liver()
        output = volumncrop.back2noscale(boxs, liver_mask)

class Liver2Tumor(Result):
    def __init__(self,
                 src_dir='/home/qzlin/.keras/datasets/LiTS/liver',
                 dst_dir='/home/qzlin/.keras/datasets/LiTS/tumor'):
        self.src_dir = src_dir
        self.dst_dir = dst_dir

        self._get_image_file()
        self.__get_indexs()

    def _get_image_file(self):
        self.images_file = [image_file for image_file in os.listdir(self.src_dir) if 'volume' in image_file]
        self.masks_file = [image_file for image_file in os.listdir(self.src_dir) if 'segmentation' in image_file]
        self.images_file.sort()
        self.masks_file.sort()
        print('image file is ', self.images_file)
        print('mask file is ', self.masks_file)

    def __get_indexs(self):
        self.indexs = [imagefile.split('-')[1] for imagefile in self.images_file]
        self.indexs = [imagefile.split('.')[0] for imagefile in self.indexs]

    def crop_liver(self):
        for i, index in enumerate(self.indexs):
            if i < 15: continue
            imagefile = os.path.join(self.src_dir, self.images_file[i])
            maskfile = os.path.join(self.src_dir, self.masks_file[i])
            liverfile = os.path.join(self.dst_dir, 'volumn-{0}.npy'.format(index))
            livermaskfile = os.path.join(self.dst_dir, 'segmentation-{0}.npy'.format(index))
            boxfile = os.path.join(self.dst_dir, 'box-{0}.npy'.format(index))

            volumncrop = VolumnCrop(imagefile, maskfile)
            liver, liver_mask, boxs = volumncrop.crop_liver()
            # liver_range = (np.min(liver), np.max(liver))
            # liver_mask_range = (np.min(liver_mask), np.max(liver_mask))
            # VolumnCropTest(None).view(np.reshape(liver, liver.shape[:-1]),
            #           np.reshape(liver_mask, liver_mask.shape[:-1]),
            #           np.reshape(liver, liver.shape[:-1]),
            #           np.reshape(liver_mask, liver_mask.shape[:-1]))
            np.save(liverfile, liver)
            np.save(livermaskfile, liver_mask)
            np.save(boxfile, boxs)






if __name__ == '__main__':
    Liver2Tumor().crop_liver()
    exit()
    # Result().view(0)
    Result().view_connected_component(0)
    exit()
    # label_ex()
    data = DataSetVolumnNii()
    channel = 1
    batch_size, file_dict = data.flow_all_memory_num(0.7, channel)
    print(file_dict)
    total_slice = 0
    for i in range(len(file_dict)):
        images, masks = data.flow_all_memory(i, file_dict, channel)
        # viewSequence(images)
        # viewSequence(masks)
        total_slice = total_slice + images.shape[0]
        print('Loading Dataset {0} with {1}'.format(i+1, images.shape))
    print('Load {0} slices, = {1} = {2} batch'.format(total_slice, np.sum(data.nums_slice), i))
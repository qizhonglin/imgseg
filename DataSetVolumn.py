import os, time
import numpy as np
from sklearn.cross_validation import train_test_split
import keras
import cv2 as cv

from utils import cvtSecond2HMS, threshold, mkdirInCache, to_category, norm_images, remove_bg
from viewer import viewSequence


class DataSetVolumn(object):
    def __init__(self, dtype='float32',
                 data_dir='/home/philips/.keras/datasets/LiTS/liver', image_size=(256, 256)):
        self.image_size = image_size
        self.dtype = dtype

        self.data_dir = data_dir
        self._get_image_file()

        self.__split_datasets()

        print('-' * 30)
        print('counting total slices of all volumn...')
        print('-' * 30)
        self.nums_slice = self.__get_numslices()

    def _get_image_file(self):
        self.images_file = [image_file for image_file in os.listdir(self.data_dir) if
                            'volumn' in image_file]
        self.masks_file = [image_file for image_file in os.listdir(self.data_dir) if
                            'segmentation' in image_file]
        self.images_file.sort()
        self.masks_file.sort()

    def __split_datasets(self):
        self.images_file_train, self.images_file_test, self.masks_file_train, self.masks_file_test = \
            train_test_split(self.images_file, self.masks_file, test_size=.1, random_state=1)
        # train1, test1, train2, test2 = \
        #     train_test_split(self.images_file, self.masks_file, test_size=.1, random_state=1)
        # assert self.images_file_train == train1
        # assert self.images_file_test == test1
        # assert self.masks_file_train == train2
        # assert self.masks_file_test == test2

    def __get_numslices(self):
        ts = time.clock()

        category = self.data_dir.split('/')[-1]
        info_file = 'cache/{0}/datasets/info.npy'.format(category)
        mkdirInCache(info_file)

        if os.path.isfile(info_file):
            self.images_file_train, nums_slice, self.images_file_test = np.load(info_file)
        else:
            nums_slice = np.zeros(len(self.images_file_train), dtype='int')
            for i, image_file in enumerate(self.images_file_train):
                image_path = os.path.join(self.data_dir, image_file)
                images = self._load_image(image_path)
                nums_slice[i] = len(images)
            np.save(info_file, (self.images_file_train, nums_slice, self.images_file_test))
        # print(nums_slice)
        # print(np.sum(nums_slice))

        # home
        # nums_slice = [41,96,276,44,177,168,183,167,186,133,189,90,226,64,181,89,241,59,37,276,110,123,119,111,250, \
        #               77,191,194,104,215,98,170,75,277,186,76,229,99,219,173,69,239,36,260,280,112,299,258,59,73, \
        #               189,29,176,116,189,233,214,179,89,92,205,118,232,56,67,170,113,169,189,132,192,139,114,194, \
        #               113,59,232,179,83,113,64,266,58,292,193,46,116,175,98,122,234,187,251,85,79,115,61,120,113, \
        #               118,215,50,37,263,56,259,215,241,248,91,149,201,79,29,198,227,112]
        # nums_slice = [41, 96, 276, 44, 177, 168, 183, 167, 186, 133, 189, 90]

        # office
        # nums_slice = [112, 36, 292, 132, 36, 241, 194, 59, 110, 56, 241, 194]
        # nums_slice = [112, 36]

        case_num = len(nums_slice)
        self.images_file_train, self.images_file_test, self.masks_file_train, self.masks_file_test = \
            self.images_file_train[:case_num], self.images_file_test[:case_num], self.masks_file_train[
                                                                                 :case_num], self.masks_file_test[
                                                                                             :case_num]
        print("run time of importing {0} data is {1}".format(np.sum(nums_slice), cvtSecond2HMS(time.clock() - ts)))
        return nums_slice

    def preprocess(self, images, masks):
        images = threshold(images, -100, 400)
        # images = norm_images(images)
        masks[masks > 0] = 1
        return images, masks

#-------------------load all data in memory---------------
    def load_traindata(self):
        num = 3#len(self.nums_slice)
        total = np.sum(self.nums_slice[:num])
        self.images = np.zeros((total, self.image_size[0], self.image_size[1], 1), dtype=self.dtype)
        self.masks = np.zeros((total, self.image_size[0], self.image_size[1], 1), dtype=self.dtype)

        print('-' * 30)
        print('Loading images and mask...')
        print('-' * 30)
        index = 0
        for i, image_file in enumerate(self.images_file_train):
            if i >= num: break
            print('loading case {0} which is {1}'.format(i, image_file))
            image, mask = self.__read_case(
                self.images_file_train[i], self.masks_file_train[i])
            self.images[index:index + image.shape[0]] = image
            self.masks[index:index + image.shape[0]] = mask
            index = index + image.shape[0]
        print('Done: {0} - {1} images'.format(i + 1, self.images.shape[0]))

        # self.images, self.masks = remove_bg(self.images, self.masks)

        return self.images, self.masks

    def load_traindata_nchannel(self, channel=5):
        total = np.sum(self.nums_slice)

        print('-' * 30)
        print('Loading images and mask...')
        print('-' * 30)
        self.images, self.masks = self.__read_multi_cases(self.images_file_train, self.masks_file_train, total, channel)

        self.assert_channel()

        return self.images, self.masks

    def assert_channel(self):
        assert np.sum(self.images[2, :, :, 0]) == np.sum(self.images[0, :, :, 2]), 'no impossible'
        assert np.sum(self.images[2, :, :, 1]) == np.sum(self.images[1, :, :, 2]), 'no impossible'

        assert np.sum(self.images[2, :, :, 2]) == np.sum(self.images[0, :, :, 4]), 'no impossible'
        assert np.sum(self.images[2, :, :, 2]) == np.sum(self.images[1, :, :, 3]), 'no impossible'
        assert np.sum(self.images[2, :, :, 2]) == np.sum(self.images[3, :, :, 1]), 'no impossible'
        assert np.sum(self.images[2, :, :, 2]) == np.sum(self.images[4, :, :, 0]), 'no impossible'

        assert np.sum(self.images[2, :, :, 3]) == np.sum(self.images[3, :, :, 2]), 'no impossible'
        assert np.sum(self.images[2, :, :, 4]) == np.sum(self.images[4, :, :, 2]), 'no impossible'

    def __read_multi_cases(self, images_file, masks_file, total_slices, channel):
        images = np.zeros((total_slices, self.image_size[0], self.image_size[1], channel), dtype=self.dtype)
        masks = np.zeros((total_slices, self.image_size[0], self.image_size[1], 1), dtype=self.dtype)

        index = 0
        cr = channel / 2
        for i, image_file in enumerate(images_file):
            print('loading case {0} which is {1}'.format(i, image_file))
            images_onecase, masks_onecase = self.__read_case(images_file[i], masks_file[i])
            num = len(images_onecase)
            for j in range(num):
                for k in range(-cr, cr + 1):
                    if 0 < j + k and j + k < num:
                        images[index + j, :, :, k + cr] = images_onecase[j + k, :, :, 0] if len(images_onecase.shape)==4 else images_onecase[j + k, :, :]
            # if len(masks_onecase.shape) == 4:
            masks[index:index + len(images_onecase), :, :, :] = masks_onecase
            # else:
            #     masks[index:index + len(images_onecase), :, :, 0] = masks_onecase
            index = index + len(images_onecase)
        return images, masks

    def __read_case(self, image_file, mask_file):
        image_path = os.path.join(self.data_dir, image_file)
        mask_path = os.path.join(self.data_dir, mask_file)
        images = self._load_image(image_path)
        masks = self._load_image(mask_path)
        if self.image_size is not (512, 512):
            images = self.__resize(images)
            masks = to_category(self.__resize(masks))
        images, masks = self.preprocess(images, masks)
        return images, masks

    def __resize(self, images):
        images_new = np.zeros((images.shape[0], self.image_size[0], self.image_size[1], 1), dtype=images.dtype)
        for i, image in enumerate(images):
            images_new[i, :, :, 0] = cv.resize(image[:, :, 0], self.image_size)
        return images_new

    def load_testdata(self):
        # self.images_file_test, self.masks_file_test = self.images_file_test[:2], self.masks_file_test[:2]

        test_data = {}
        print('-' * 30)
        print('Loading and preprocessing validation cases...')
        print('-' * 30)
        for image_file, mask_file in zip(self.images_file_test, self.masks_file_test):
            print('loading case {0}'.format(image_file))
            images_onecase, masks_onecase = self.__read_case(image_file, mask_file)
            test_data[image_file] = (images_onecase, masks_onecase)
        return test_data

    def validation_data(self):
        images_onecase, masks_onecase = self.__read_case(self.images_file_test[0], self.masks_file_test[0])
        return images_onecase, masks_onecase
        # test_data = self.load_testdata()
        # count = 0
        # for image_file, (image, mask) in test_data.iteritems():
        #     count = count + image.shape[0]
        # images = np.zeros((count, image.shape[1], image.shape[2], image.shape[3]), dtype=np.float32)
        # masks = np.zeros_like(images)
        # i = 0
        # for image_file, (image, mask) in test_data.iteritems():
        #     images[i:i+image.shape[0]] = image
        #     masks[i:i+mask.shape[0]] = mask
        #     i = i + image.shape[0]
        # return images, masks

#-------------------load batch by batch with all memory---------
    def flow_all_memory(self, index, file_dict, channel=5):
        index = index % len(file_dict)
        file_list = list(file_dict)
        batch_file = file_dict[file_list[index]]
        images_file = batch_file[0]
        masks_file = batch_file[1]
        dataset_size = batch_file[2]
        images, masks = self.__read_multi_cases(images_file, masks_file, dataset_size, channel)

        return images, masks

    def flow_all_memory_num(self, ratio=0.7, channel=5):
        free_memory = 12 * ratio     # xx G
        dataset_size = int(free_memory * 1024 / channel)             #one slice requires 512*512*channel*4byte = channel M
        # dataset_size = 1000
        print('idea dataset_size is {0} slices'.format(dataset_size))

        # group volumns, the size of which is no more than dataset_size
        nums_slice_cum = np.cumsum(self.nums_slice)
        nums_slice = np.array(self.nums_slice)
        file_split = [0]
        for i in range(len(nums_slice_cum)):
            index = np.searchsorted(nums_slice_cum, dataset_size)
            file_split.append(index)
            nums_slice[:index] = 0
            nums_slice_cum = np.cumsum(nums_slice)
            if index == len(nums_slice_cum):
                break

        file_dict = {}
        for i in range(len(file_split)-1):
            s, e = file_split[i], file_split[i+1]
            file_dict[(s, e)] = self.images_file_train[s:e], self.masks_file_train[s:e], np.sum(self.nums_slice[s:e])
        return dataset_size, file_dict

    def load_testdata_channel(self, channel=5):
        test_data = {}
        print('-' * 30)
        print('Loading and preprocessing validation cases...')
        print('-' * 30)
        for image_file, mask_file in zip(self.images_file_test, self.masks_file_test):
            print('loading case {0}'.format(image_file))
            image, mask = self.__read_case(image_file, mask_file)
            images, masks = self.__read_multi_cases([image_file], [mask_file], image.shape[0], channel)
            test_data[image_file] = (images, masks)
        return test_data

    def validation_data_channel(self, channel=5):
        image, mask = self.__read_case(self.images_file_test[0], self.masks_file_test[0])
        images, masks = self.__read_multi_cases([self.images_file_test[0]], [self.masks_file_test[0]], image.shape[0], channel)
        return images, masks

    def _load_image(self, image_path):
        return np.load(image_path)

class TumorVolumn(DataSetVolumn):
    def __init__(self, dtype='float32',
                 data_dir='/home/philips/.keras/datasets/LiTS/tumor', image_size=(256, 256)):
        super(TumorVolumn, self).__init__(dtype, data_dir, image_size)

    def preprocess(self, images, masks):
        # images = threshold(images, np.min(images), np.max(images))
        images = threshold(images, -200, 200)
        # images = norm_images(images)
        # images = equalizeHist(images)
        # images = threshold(images, np.min(images), np.max(images))
        # print((np.min(masks), np.max(masks)))
        masks[masks < 2] = 0
        masks /= 2
        # print((np.min(masks), np.max(masks)))
        # assert np.sum(masks[masks>0]) == np.sum(masks>0)
        return images, masks

    def get_data(self, imagefile):
        index = imagefile.split('-')[1]
        index = index.split('.')[0]

        boxs = np.load(os.path.join(self.data_dir, 'box-{0}.npy'.format(index)))
        images = np.load(os.path.join(self.data_dir, 'volumn-{0}.npy'.format(index)))
        masks = np.load(os.path.join(self.data_dir, 'segmentation-{0}.npy'.format(index)))
        return boxs, images, masks




class VolumnViewer(DataSetVolumn):
    def preprocess(self, images, masks):
        images = threshold(images, np.min(images), np.max(images))
        return images, masks



from viewer import show_image_mask

if __name__ == '__main__':
    # test_data = DataSetVolumn().load_testdata()
    # for image_file, data in test_data.iteritems():
    #     print('{0} has {1} slices and {2} masks'.format(image_file, data[0].shape, data[1].shape))

    # images_onecase, masks_onecase = DataSetVolumn().validation_data()
    images_onecase, masks_onecase = TumorVolumn().validation_data()
    images_onecase, masks_onecase = remove_bg(images_onecase, masks_onecase)

    # show_image_mask(images_onecase, masks_onecase)
    viewSequence(images_onecase)
    viewSequence(masks_onecase)
    exit()

    # out of memory
    # images, masks = DataSetVolumn().load_traindata_nchannel(channel=5)
    # print('Loading Dataset {0}'.format(images.shape))

    # data = DataSetVolumn()
    data = TumorVolumn()
    # channel = 5
    # batch_size, file_dict = data.flow_all_memory_num(0.7, channel)
    # total_slice = 0
    # for i in range(len(file_dict)):
    #     images, masks = data.flow_all_memory(i, file_dict, channel)
    #     total_slice = total_slice + images.shape[0]
    #     print('Loading Dataset {0} with {1}'.format(i+1, images.shape))
    # print('Load {0} slices, = {1}'.format(total_slice, np.sum(data.nums_slice)))

    # image, mask = data.validation_data_channel()
    # print(image.shape)
    test_data = data.load_testdata()







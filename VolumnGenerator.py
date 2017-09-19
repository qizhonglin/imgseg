
import os, time
import numpy as np
from sklearn.cross_validation import train_test_split
from itertools import izip
from keras.utils.data_utils import Sequence
import keras

from utils import cvtSecond2HMS, threshold
import cv2
from psutil import virtual_memory


class VolumnGenerator(Sequence):
    def __init__(self, data_dir='/home/qzlin/.keras/datasets/LiTS', batch_size=32, image_size=None, min_value=-100, max_value=400, typename='liver'):
        self.data_dir = data_dir

        self.images_file = [image_file for image_file in os.listdir(self.data_dir) if
                       'volume' in image_file]
        files_num = [image_file.split('-')[1] for image_file in self.images_file]
        self.masks_file = ['segmentation-{0}-mask.npy'.format(filename) for filename in files_num]
        # print('volume: ', self.images_file)
        # print('segmentation', self.masks_file)
        # print('total volume: ', len(self.images_file), len(self.masks_file))

        self.images_file_train, self.images_file_test, self.masks_file_train, self.masks_file_test = \
            train_test_split(self.images_file, self.masks_file, test_size=.1, random_state=1)
        self.nums_slice = self.__get_numslices()

        self.batch_size = batch_size
        self.image_size = image_size

        self.min_value = min_value
        self.max_value = max_value
        self.typename = typename

    def get_batch_size(self):
        return self.batch_size

    @property
    def image_shape(self):
        if self.image_size:
            return (self.image_size[0], self.image_size[1], 1)
        return (512, 512, 1)

    @property
    def steps_per_epoch(self):
        return np.sum(self.nums_slice) / self.batch_size + 1

    def __len__(self):
        return np.sum(self.nums_slice)

    def __getitem__(self, item):
        batch_size = self.batch_size
        image_size = self.image_size

        item = item % self.steps_per_epoch
        num_slices = self.nums_slice
        num_slices_cum = np.cumsum(num_slices)

        image_batch, mask_batch = None, None

        s = item * batch_size
        e = (item + 1) * batch_size
        s_index = np.searchsorted(num_slices_cum, s)
        e_index = np.searchsorted(num_slices_cum, e)
        if s_index < len(num_slices) and e_index < len(num_slices):
            # print('{0} in {1}'.format(s, s_index))
            # print('{0} in {1}'.format(e, e_index))

            if s_index == e_index:
                images, masks = self.read_case(self.images_file_train[s_index],
                                               self.masks_file_train[s_index])
                if s_index == 0:
                    slice_index = s
                else:
                    slice_index = s - num_slices_cum[s_index - 1]
                image_batch = images[slice_index:slice_index + batch_size]
                mask_batch = masks[slice_index:slice_index + batch_size]
            else:
                images_s, masks_s = self.read_case(self.images_file_train[s_index],
                                                   self.masks_file_train[s_index])
                images_e, masks_e = self.read_case(self.images_file_train[e_index],
                                                   self.masks_file_train[e_index])

                image_batch = VolumnGenerator.__get_image_batch(images_s, s, num_slices_cum, s_index, images_e, e,
                                                                     e_index)
                mask_batch = VolumnGenerator.__get_image_batch(masks_s, s, num_slices_cum, s_index, masks_e, e,
                                                                    e_index)
            # print('batch_index = {0} from {1} to {2} in volumn index {3} and {4}, batch_size = {5}'.
            #       format(item, s, e, s_index, e_index, image_batch.shape))
        elif item * batch_size < num_slices_cum[-1]:
            images, masks = self.read_case(self.images_file_train[s_index],
                                           self.masks_file_train[s_index])
            rem_s = num_slices[-1] + item * batch_size - num_slices_cum[-1]
            image_batch = images[rem_s:]
            mask_batch = masks[rem_s:]
            # print('batch_index = {0} from {1} to {2} in volumn index {3} and {4}, batch_size = {5}'.
            #       format(item, s, num_slices_cum[-1], s_index, s_index, image_batch.shape))
        if image_batch is not None:
            image_batch = np.swapaxes(image_batch, 1, 3)
            mask_batch = np.swapaxes(mask_batch, 1, 3)
            if image_size is not None:
                image_batch = np.array([cv2.resize(image, image_size) for image in image_batch])
                mask_batch = np.array([cv2.resize(image, image_size) for image in mask_batch])
                dims = image_batch.shape
                image_batch, mask_batch = image_batch.reshape(dims + (1,)), mask_batch.reshape(dims + (1,))
            image_batch = threshold(image_batch, self.min_value, self.max_value)
            if self.typename == 'liver':
                mask_batch[mask_batch > 0] = 1
        return image_batch, mask_batch

    def get_index(self, item=0, batch_size=32, image_size=None):
        item = item % self.steps_per_epoch
        num_slices = self.nums_slice
        num_slices_cum = np.cumsum(num_slices)

        image_batch, mask_batch = None, None

        s = item * batch_size
        e = (item + 1) * batch_size
        s_index = np.searchsorted(num_slices_cum, s)
        e_index = np.searchsorted(num_slices_cum, e)
        if s_index < len(num_slices) and e_index < len(num_slices):
            # print('{0} in {1}'.format(s, s_index))
            # print('{0} in {1}'.format(e, e_index))

            if s_index == e_index:
                images, masks = self.read_case(self.images_file_train[s_index],
                                               self.masks_file_train[s_index])
                if s_index == 0:
                    slice_index = s
                else:
                    slice_index = s - num_slices_cum[s_index-1]
                image_batch = images[slice_index:slice_index+batch_size]
                mask_batch = masks[slice_index:slice_index+batch_size]
            else:
                images_s, masks_s = self.read_case(self.images_file_train[s_index],
                                                   self.masks_file_train[s_index])
                images_e, masks_e = self.read_case(self.images_file_train[e_index],
                                                   self.masks_file_train[e_index])
                # image_batch_s = images_s[s-num_slices_cum[s_index-1]:]
                # image_batch_e = images_e[:e-num_slices_cum[e_index - 1]]
                # image_batch = np.vstack((image_batch_s, image_batch_e))
                image_batch = VolumnGenerator.__get_image_batch(images_s, s, num_slices_cum, s_index, images_e, e, e_index)
                mask_batch = VolumnGenerator.__get_image_batch(masks_s, s, num_slices_cum, s_index, masks_e, e, e_index)
            print('batch_index = {0} from {1} to {2} in volumn index {3} and {4}, batch_size = {5}'.format(item, s, e, s_index, e_index, image_batch.shape))
            # return image_batch, mask_batch
        elif item*batch_size < num_slices_cum[-1]:
            images, masks = self.read_case(self.images_file_train[s_index],
                                           self.masks_file_train[s_index])
            rem_s = num_slices[-1] + item*batch_size-num_slices_cum[-1]
            image_batch = images[rem_s:]
            mask_batch = masks[rem_s:]
            print('batch_index = {0} from {1} to {2} in volumn index {3} and {4}, batch_size = {5}'.format(item, s, num_slices_cum[-1],
                                                                                                           s_index,
                                                                                                           s_index,
                                                                                                           image_batch.shape))
            # return image_batch, mask_batch
        if image_batch is not None:
            image_batch = np.swapaxes(image_batch, 1, 3)
            mask_batch = np.swapaxes(mask_batch, 1, 3)
            if image_size is not None:
                image_batch = np.array([cv2.resize(image, (image_size[0], image_size[1], 1)) for image in image_batch])
                mask_batch = np.array([cv2.resize(image, (image_size[0], image_size[1], 1)) for image in mask_batch])
                dims = image_batch.shape
                image_batch, mask_batch = image_batch.reshape(dims + (1,)), mask_batch.reshape(dims + (1,))
            image_batch = threshold(image_batch, self.min_value, self.max_value)
            if self.typename == 'liver':
                mask_batch[mask_batch > 0] = 1
        return image_batch, mask_batch

    def flow(self):
        images_rem = None
        masks_rem = None
        for (image_file, mask_file) in izip(self.images_file_train, self.masks_file_train):
            images, masks = self.read_case(image_file, mask_file)

            if images_rem is not None and len(images_rem):
                images = np.vstack((images, images_rem))
                masks = np.vstack((masks, masks_rem))

            count = len(images) / self.batch_size
            images_rem = images[count*self.batch_size:]
            masks_rem = masks[count*self.batch_size:]
            for i in xrange(count):
                images_batch = images[i*self.batch_size:(i+1)*self.batch_size]
                masks_batch = masks[i*self.batch_size:(i+1)*self.batch_size]
                yield (images_batch, masks_batch)
        if images_rem is not None and len(images_rem):
            yield (images_rem, masks_rem)

    def flow_all_memory(self, ratio=0.7):
        # mem = virtual_memory()
        # print('total memory is %dG', float(mem.total) / 1024 ** 3)
        # used_memory = ratio * float(mem.total) / 1024 ** 3
        # dataset_size = int(used_memory * 1024)
        # nums_slice_cum = np.cumsum(self.nums_slice)
        #
        # count = nums_slice_cum[-1] / dataset_size
        dataset_size, count = self.flow_all_memory_num(ratio)
        nums_slice_cum = np.cumsum(self.nums_slice)
        for i in range(count):
            s_index = np.searchsorted(nums_slice_cum, dataset_size*i)
            e_index = np.searchsorted(nums_slice_cum, dataset_size*(i+1))
            total = sum(self.nums_slice[s_index:e_index])
            images_file_mem = self.images_file_train[s_index:e_index]
            masks_file_mem = self.masks_file_train[s_index:e_index]
            print('-' * 30)
            print('Loading images and masks {0} with all memory'.format(i+1))
            print('-' * 30)
            images, masks = self.read_multi_cases(images_file_mem, masks_file_mem, total)
            yield (images, masks)
        total = sum(self.nums_slice[e_index:])
        images_file_mem = self.images_file_train[e_index:]
        masks_file_mem = self.masks_file_train[e_index:]
        print('-' * 30)
        print('Loading images and masks {0} with all memory'.format(count+1))
        print('-' * 30)
        images, masks = self.read_multi_cases(images_file_mem, masks_file_mem, total)
        yield (images, masks)
    def flow_all_memory_num(self, ratio=0.7):
        mem = virtual_memory()
        print('total memory is %dG', float(mem.total) / 1024 ** 3)
        used_memory = ratio * float(mem.total) / 1024 ** 3  #XX G
        dataset_size = int(used_memory * 1024)              #one slice requires 512*512*4byte = 1M
        nums_slice_cum = np.cumsum(self.nums_slice)

        count = nums_slice_cum[-1] / dataset_size
        return dataset_size, count+1

    def read_multi_cases(self, images_file, masks_file, total_slices):
        w = 512 if self.image_size is None else self.image_size[0]
        h = 512 if self.image_size is None else self.image_size[1]
        images = np.ndarray((total_slices, w, h, 1), dtype='int16')
        masks = np.ndarray((total_slices, w, h, 1), dtype='int16')

        index = 0
        for i in range(len(images_file)):
            print('loading case {0} which is {1}'.format(i, images_file[i]))
            images_onecase, masks_onecase = self.read_case(
                images_file[i], masks_file[i])
            images[index:index + len(images_onecase), :, :, :] = np.swapaxes(images_onecase, 1, 3)
            masks[index:index + len(images_onecase), :, :, :] = np.swapaxes(masks_onecase, 1, 3)
            index = index + len(images_onecase)
        print('Done: {0} cases have {1} images'.format(i + 1, images.shape[0]))
        return images, masks

    def read_case(self, image_file, mask_file):
        image_path = os.path.join(self.data_dir, image_file)
        mask_path = os.path.join(self.data_dir, mask_file)
        images = np.load(image_path)
        masks = np.load(mask_path)
        return images, masks

    def __get_numslices(self):
        ts = time.clock()
        # nums_slice = np.zeros(len(self.images_file_train), dtype='int')
        # for i, image_file in enumerate(self.images_file_train):
        #     image_path = os.path.join(self.data_dir, image_file)
        #     images = np.load(image_path)
        #     nums_slice[i] = len(images)
        # print(nums_slice)
        # print(np.sum(nums_slice))

        nums_slice = [41,96,276,44,177,168,183,167,186,133,189,90,226,64,181,89,241,59,37,276,110,123,119,111,250, \
                      77,191,194,104,215,98,170,75,277,186,76,229,99,219,173,69,239,36,260,280,112,299,258,59,73, \
                      189,29,176,116,189,233,214,179,89,92,205,118,232,56,67,170,113,169,189,132,192,139,114,194, \
                      113,59,232,179,83,113,64,266,58,292,193,46,116,175,98,122,234,187,251,85,79,115,61,120,113, \
                      118,215,50,37,263,56,259,215,241,248,91,149,201,79,29,198,227,112]
        # nums_slice = [41, 96, 276, 44, 177, 168, 183, 167, 186, 133, 189, 90]
        print("run time of importing all data: ", cvtSecond2HMS(time.clock() - ts))
        return nums_slice

    @staticmethod
    def __get_image_batch(images_s, s, num_slices_cum, s_index, images_e, e, e_index):
        image_batch_s = images_s[s - num_slices_cum[0 if s_index == 0 else s_index - 1]:]
        image_batch_e = images_e[:e - num_slices_cum[0 if e_index == 0 else e_index - 1]]
        image_batch = np.vstack((image_batch_s, image_batch_e))
        return image_batch

from DataSetVolumn import viewSequence
if __name__ == '__main__':
    ts = time.clock()

    reader = VolumnGenerator()
    print('num_slices = {0}, total slices = {1}'.format(reader.nums_slice, np.sum(reader.nums_slice)))


    # for item in range(1):
    #     image_batch, mask_batch = reader.get_index(item)
    #     viewSequence(image_batch)
    #     viewSequence(mask_batch)

    total_batch = 0
    for (images_batch, masks_batch) in reader.flow_all_memory():
        print(images_batch.shape)
        total_batch = total_batch + 1
    print('total batch is ', total_batch)


    print("total process time: ", cvtSecond2HMS(time.clock() - ts))





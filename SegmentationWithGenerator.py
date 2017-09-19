from sklearn.cross_validation import train_test_split
from utils.ImageModel import *
from utils import showImages, print_env, cvtSecond2HMS, add_backend_name, mkdirInCache
from utils.metrics import Metrics
from pprint import pprint
import time
import random

from VolumnGenerator import VolumnGenerator

seed = 9001
random.seed(seed)

class SegmentationWithGenerator(object):
    def __init__(self, model_file='cache/model.json', weights_file="cache/weight.h5", modelcheckpoint='cache/unet.hdf5'):
        self.model = None

        (self.model_file, self.weights_file) = add_backend_name(model_file, weights_file)
        mkdirInCache(self.model_file)
        self.modelcheckpoint = modelcheckpoint

    def train(self, generator, model_name='unet', nb_epoch=500, weights_path=None):
        print('-' * 30)
        print('Creating model...')
        print('-' * 30)
        model = globals()[model_name](generator.image_shape, self.model_file, self.weights_file).get_model_pretrain(weights_path)

        print('-' * 30)
        print('compiling model...')
        print('-' * 30)
        model = ImageModel.compile(model)

        print('-' * 30)
        print('Fitting model...')
        print('-' * 30)
        model.fit_generator(generator, steps_per_epoch=generator.steps_per_epoch, epochs=nb_epoch)

        print('-' * 30)
        print('saving model to {0} and {1}'.format(self.model_file, self.weights_file))
        print('-' * 30)
        ImageModel(self.model_file, self.weights_file).save_model(model)

        self.model = model

    @staticmethod
    def run(
            istrain=False,
            model_name='unet',
            model_file='cache/liver/model/model_unet.json',
            weights_file="cache/liver/model/weight_unet.h5",
            modelcheckpoint='cache/liver/model/unet.hdf5',
            model_pretrain='cache/liver/model/weight_unet_gen_tf.h5',
            nb_epoch=500):

        seg = SegmentationWithGenerator(model_file, weights_file, modelcheckpoint)

        if istrain:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            weights_path = os.path.expanduser(os.path.join(current_dir, model_pretrain)) if model_pretrain else None
            seg.train(VolumnGenerator(batch_size=1), model_name,
                      weights_path=weights_path,
                      nb_epoch=nb_epoch)


if __name__ == '__main__':
    SegmentationWithGenerator.run(istrain=True)
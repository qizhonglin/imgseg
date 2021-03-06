from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.layers.advanced_activations import PReLU

from BilinearUpSampling import BilinearUpSampling2D

from keras_contrib.applications import densenet

import os
import sys
import numpy as np
from keras.applications.vgg16 import *

from resnet_helpers import *
from densenet_helpers import create_fcn_densenet

class ImageModel(object):
    def __init__(self):
        self.model = None

    # @staticmethod
    # def dice_coef(y_true, y_pred, smooth=1.0):
    #     w = 256
    #     y_true = K.reshape(y_true, [-1, w*w])
    #     y_pred = K.reshape(y_pred, [-1, w*w])
    #     intersection = K.sum(y_true * y_pred, axis=-1)
    #     return (2. * intersection + smooth) / (K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) + smooth)
    #
    # @staticmethod
    # def dice_coef_loss(y_true, y_pred):
    #     cw = [0.534, 7.828]
    #     w = 256
    #     cw = [1, 2.2]
    #     # loss = cw[0] * y_true * K.log(y_pred+K.epsilon()) + cw[1] * (1-y_true) * K.log(1-y_pred+K.epsilon())
    #     y_true = K.reshape(y_true, [-1, w * w])
    #     y_pred = K.reshape(y_pred, [-1, w * w])
    #     loss = cw[0] * y_true * K.log(y_pred+K.epsilon()) + cw[1]*(1-y_true)*K.log(1-y_pred+K.epsilon())
    #     return -K.mean(loss, axis=-1)

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1.0):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    def dice_coef_loss(y_true, y_pred):
        return 1 - ImageModel.dice_coef(y_true, y_pred)

    @staticmethod
    def compile(model):
        model.compile(optimizer=Adam(lr=1e-5), loss=ImageModel.dice_coef_loss, metrics=[ImageModel.dice_coef])
        return model

    # def save_model(self, model):
    #     # serialize model to JSON
    #     model_json = model.to_json()
    #     with open(self.model_file, "w") as json_file:
    #         json_file.write(model_json)
    #
    #     # serialize weights to HDF5
    #     model.save_weights(self.weights_file)
    #     print("Saved model to disk in {0} and {1}".format(self.model_file, self.weights_file))
    #
    #     self.model = model
    #
    # def load_model(self):
    #     # load json and create model
    #     json_file = open(self.model_file, 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     # loaded_model = model_from_json(loaded_model_json)
    #     loaded_model = model_from_json(loaded_model_json, custom_objects={"BilinearUpSampling2D": BilinearUpSampling2D})
    #
    #     # load weights into new model
    #     loaded_model.load_weights(self.weights_file)
    #     print("Loaded model from disk in {0} and {1}".format(self.model_file, self.weights_file))
    #
    #     self.model = loaded_model
    #     return self.model

    def merge_feature_maps(self, conv5, conv4):
        if K.image_data_format() == 'channels_last':
            up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
        if K.image_data_format() == 'channels_first':
            up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
        return up6

    def get_model(self):
        return self.model

    def get_model_pretrain(self, weights_path):
        model = self.get_model()

        if weights_path and os.path.isfile(weights_path):
            model.load_weights(weights_path, by_name=False)
        return model

    def resume_model(self, weights_path,  modelcheckpoint):
        model = self.get_model_pretrain(weights_path)
        if modelcheckpoint and os.path.isfile(modelcheckpoint):
            model.load_weights(modelcheckpoint)
        return model

    def create_unet(self, inputs, nb_filters=[32, 64, 128, 256, 512], weight_decay=1e-4):
        skips = []
        x = inputs
        # downsampling way
        for i, nb_filter in enumerate(nb_filters[:-1]):
            for j in range(2):
                name = 'block{0}_conv{1}'.format(i, j)
                x = Convolution2D(nb_filter, 3, 3, activation='relu', border_mode='same', name=name, W_regularizer=l2(weight_decay))(x)
            skips.append(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # bottleneck
        for j in range(2):
            name = 'block{0}_conv{1}'.format(i+1, j)
            x = Convolution2D(nb_filters[-1], 3, 3, activation='relu', border_mode='same', name=name, W_regularizer=l2(weight_decay))(x)

        # upsampling way
        skips = skips[::-1]
        nb_filters_up = nb_filters[:-1][::-1]
        for i, nb_filter in enumerate(nb_filters_up):
            x = self.merge_feature_maps(x, skips[i])
            for j in range(2):
                name = 'block{0}_conv{1}'.format(i+len(nb_filters), j)
                x = Convolution2D(nb_filter, 3, 3, activation='relu', border_mode='same', name=name, W_regularizer=l2(weight_decay))(x)

        # final layer
        output = Convolution2D(1, 1, 1, activation='sigmoid')(x)
        return output

    def create_unetresnet(self, inputs, nb_filters=[64, 128, 256, 512, 512], repeats=[2, 2, 3, 3, 3], weight_decay=1e-4):
        skips = []
        x = inputs

        # initial layer
        for j in range(repeats[0]):
            name = 'block{0}_conv{1}'.format(0, j)
            x = Convolution2D(nb_filters[0], 3, 3, activation='relu', border_mode='same', name=name, W_regularizer=l2(weight_decay))(x)
        skips.append(x)

        # downsampling way
        for i, (nb_filter, repeat) in enumerate(zip(nb_filters[1:], repeats[1:])):
            x = MaxPooling2D(pool_size=(2, 2))(x)
            shortcut = Convolution2D(nb_filter, 1, 1, name='block{0}_conv{1}'.format(i+1, 0), W_regularizer=l2(weight_decay))(x)
            for j in range(repeat):
                name = 'block{0}_conv{1}'.format(i+1, j+1)
                x = Convolution2D(nb_filter, 3, 3, activation='relu', border_mode='same', name=name, W_regularizer=l2(weight_decay))(x)
            x = merge([x, shortcut], mode='sum')
            skips.append(x)
        skips = skips[:-1]

        # upsampling way
        skips = skips[::-1]
        nb_filters_up = nb_filters[:-1][::-1]
        repeats_up = repeats[:-1][::-1]
        for i, (nb_filter, repeat) in enumerate(zip(nb_filters_up, repeats_up)):
            x = self.merge_feature_maps(x, skips[i])
            shortcut = Convolution2D(nb_filter, 1, 1, name='block{0}_conv{1}'.format(i + len(nb_filters), 0), W_regularizer=l2(weight_decay))(x)
            for j in range(repeat):
                name = 'block{0}_conv{1}'.format(i + len(nb_filters), j+1)
                x = Convolution2D(nb_filter, 3, 3, activation='relu', border_mode='same', name=name, W_regularizer=l2(weight_decay))(x)
            x = merge([x, shortcut], mode='sum')

        # final layer
        output = Convolution2D(1, 1, 1, activation='sigmoid')(x)
        return output

class unet(ImageModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(unet, self).__init__()

    def get_model(self):
        inputs = Input(self.input_shape)

        outputs = self.create_unet(inputs)

        model = Model(input=inputs, output=outputs)

        model.summary()
        return model

class unet5(ImageModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(unet5, self).__init__()

    def get_model(self):
        inputs = Input(self.input_shape)

        outputs = self.create_unet(inputs, nb_filters=[16, 32, 64, 128, 256, 512])

        model = Model(input=inputs, output=outputs)

        model.summary()
        return model

class unet_standard(ImageModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(unet_standard, self).__init__()

    def get_model(self):
        inputs = Input(self.input_shape)

        outputs = self.create_unet(inputs, nb_filters=[64, 128, 256, 512, 1024])

        # weight_decay = 1e-4
        #
        # conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', W_regularizer=l2(weight_decay))(inputs)
        # conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2', W_regularizer=l2(weight_decay))(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)
        # # pool1 = BatchNormalization()(pool1)
        #
        # conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1', W_regularizer=l2(weight_decay))(pool1)
        # conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2', W_regularizer=l2(weight_decay))(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)
        # # pool2 = BatchNormalization()(pool2)
        #
        # conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1', W_regularizer=l2(weight_decay))(pool2)
        # conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2', W_regularizer=l2(weight_decay))(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)
        # # pool3 = BatchNormalization()(pool3)
        #
        # conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1', W_regularizer=l2(weight_decay))(pool3)
        # conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2', W_regularizer=l2(weight_decay))(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(conv4)
        # # pool4 = BatchNormalization()(pool4)
        #
        # conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='block5_conv1', W_regularizer=l2(weight_decay))(pool4)
        #
        # up6 = self.merge_feature_maps(conv5, conv4)
        # conv6 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='block6_conv1', W_regularizer=l2(weight_decay))(up6)
        # conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block6_conv2', W_regularizer=l2(weight_decay))(conv6)
        #
        # up7 = self.merge_feature_maps(conv6, conv3)
        # # up7 = BatchNormalization()(up7)
        # conv7 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block7_conv1', W_regularizer=l2(weight_decay))(up7)
        # conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block7_conv2', W_regularizer=l2(weight_decay))(conv7)
        #
        # up8 = self.merge_feature_maps(conv7, conv2)
        # # up8 = BatchNormalization()(up8)
        # conv8 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block8_conv1', W_regularizer=l2(weight_decay))(up8)
        # conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block8_conv2', W_regularizer=l2(weight_decay))(conv8)
        #
        # up9 = self.merge_feature_maps(conv8, conv1)
        # # up9 = BatchNormalization()(up9)
        # conv9 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block9_conv1', W_regularizer=l2(weight_decay))(up9)
        # conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block9_conv2', W_regularizer=l2(weight_decay))(conv9)
        # conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block9_conv3', W_regularizer=l2(weight_decay))(conv9)
        #
        # # conv9 = BatchNormalization()(conv9)
        # conv10 = Convolution2D(1, 1, 1, activation='sigmoid', name='block10_conv1')(conv9)

        model = Model(input=inputs, output=outputs)

        model.summary()
        return model

class FCN(ImageModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(FCN, self).__init__()

    @staticmethod
    def transfer_FCN_Vgg16():
        input_shape = (224, 224, 1)
        img_input = Input(shape=input_shape)
        # Block 1
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='no1')(img_input)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='no2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # Convolutional layers transfered from fully-connected layers
        x = Convolution2D(4096, 7, 7, activation='relu', border_mode='same', name='fc1')(x)
        x = Convolution2D(4096, 1, 1, activation='relu', border_mode='same', name='fc2')(x)
        x = Convolution2D(1, 1, 1, activation='softmax', name='predictions_1')(x)
        # x = Reshape((7,7))(x)

        # Create model
        model = Model(img_input, x)
        weights_path = os.path.expanduser(
            os.path.join('~', '.keras/models/skin_fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))

        # transfer if weights have not been created
        if os.path.isfile(weights_path) == False:
            flattened_layers = model.layers
            index = {}
            for layer in flattened_layers:
                if layer.name:
                    index[layer.name] = layer
            vgg16 = VGG16()
            for layer in vgg16.layers:
                weights = layer.get_weights()
                if layer.name == 'fc1':
                    weights[0] = np.reshape(weights[0], (7, 7, 512, 4096))
                elif layer.name == 'fc2':
                    weights[0] = np.reshape(weights[0], (1, 1, 4096, 4096))
                # elif layer.name == 'predictions':
                #     layer.name = 'predictions_1000'
                #     weights[0] = np.reshape(weights[0], (1, 1, 4096, 1000))
                if index.has_key(layer.name):
                    index[layer.name].set_weights(weights)
            model.save_weights(weights_path)
            print('Successfully transformed!')
        # else load weights
        else:
            model.load_weights(weights_path, by_name=True)
            print('Already transformed!')

    def get_model(self):
        inputs = Input(self.input_shape)

        weight_decay = 1e-4

        # Block 1
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', W_regularizer=l2(weight_decay))(inputs)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2', W_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1', W_regularizer=l2(weight_decay))(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2', W_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1', W_regularizer=l2(weight_decay))(x)
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2', W_regularizer=l2(weight_decay))(x)
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3', W_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1', W_regularizer=l2(weight_decay))(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2', W_regularizer=l2(weight_decay))(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3', W_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1', W_regularizer=l2(weight_decay))(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2', W_regularizer=l2(weight_decay))(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3', W_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # Convolutional layers transfered from fully-connected layers
        x = Convolution2D(4096, 7, 7, activation='relu', border_mode='same', name='fc1', W_regularizer=l2(weight_decay))(x)
        x = Dropout(0.5)(x)
        x = Convolution2D(4096, 1, 1, activation='relu', border_mode='same', name='fc2', W_regularizer=l2(weight_decay))(x)
        x = Dropout(0.5)(x)
        # classifying layer
        x = Convolution2D(1, 1, 1, init='he_normal', activation='sigmoid', border_mode='valid', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)

        x = BilinearUpSampling2D(size=(32, 32))(x)

        model = Model(inputs, x)

        # weights_path = os.path.expanduser(
        #     os.path.join('~', '.keras/models/skin_fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
        # model.load_weights(weights_path, by_name=True)

        model.summary()
        return model

class ResNet50(ImageModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(ResNet50, self).__init__()

    def get_model(self):
        inputs = Input(self.input_shape)

        weight_decay = 1e-4
        batch_momentum = 0.9

        bn_axis = 3

        x = Convolution2D(64, 7, 7, subsample=(2, 2), border_mode='same', name='conv1', W_regularizer=l2(weight_decay))(inputs)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(3, [64, 64, 256], stage=2, block='a')(x)
        x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
        x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)

        x = conv_block(3, [128, 128, 512], stage=3, block='a', strides=(2, 2))(x)
        x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
        x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
        x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)

        x = conv_block(3, [256, 256, 1024], stage=4, block='a', strides=(2, 2))(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

        x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', atrous_rate=(2, 2))(x)
        x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', atrous_rate=(2, 2))(x)
        x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', atrous_rate=(2, 2))(x)

        x = Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)
        x = BilinearUpSampling2D(target_size=tuple(self.input_shape[0:2]))(x)

        model = Model(inputs, x)

        model.summary()
        return model

class UnetResNet(ImageModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(UnetResNet, self).__init__()

    def get_model(self):
        inputs = Input(self.input_shape)

        weight_decay = 1e-4

        x1 = Convolution2D(64, 7, 7, border_mode='same', name='conv1', W_regularizer=l2(weight_decay))(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        p1 = MaxPooling2D(strides=(2, 2))(x1)

        x2 = conv_block(3, [64, 64, 64], stage=1, block='a')(p1)
        x2 = identity_block(3, [64, 64, 64], stage=1, block='b')(x2)
        p2 = MaxPooling2D(pool_size=(2, 2))(x2)

        x3 = conv_block(3, [128, 128, 128], stage=2, block='a')(p2)
        x3 = identity_block(3, [128, 128, 128], stage=2, block='b')(x3)
        p3 = MaxPooling2D(pool_size=(2, 2))(x3)

        x4 = conv_block(3, [256, 256, 256], stage=3, block='a')(p3)
        x4 = identity_block(3, [256, 256, 256], stage=3, block='b')(x4)
        p4 = MaxPooling2D(pool_size=(2, 2))(x4)

        x5 = conv_block(3, [512, 512, 512], stage=4, block='a')(p4)
        x5 = identity_block(3, [512, 512, 512], stage=4, block='b')(x5)

        x6 = self.merge_feature_maps(x5, x4)
        x6 = conv_block(3, [256, 256, 256], stage=6, block='a')(x6)
        x6 = identity_block(3, [256, 256, 256], stage=6, block='b')(x6)

        x7 = self.merge_feature_maps(x6, x3)
        x7 = conv_block(3, [128, 128, 128], stage=7, block='a')(x7)
        x7 = identity_block(3, [128, 128, 128], stage=7, block='b')(x7)

        x8 = self.merge_feature_maps(x7, x2)
        x8 = conv_block(3, [64, 64, 64], stage=8, block='a')(x8)
        x8 = identity_block(3, [64, 64, 64], stage=8, block='b')(x8)

        x9 = self.merge_feature_maps(x8, x1)
        x9 = conv_block(3, [32, 32, 32], stage=9, block='a')(x9)
        x9 = identity_block(3, [32, 32, 32], stage=9, block='b')(x9)

        x10 = Convolution2D(1, 1, 1, activation='sigmoid')(x9)

        model = Model(input=inputs, output=x10)

        model.summary()
        return model

class DenseNet(ImageModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(DenseNet, self).__init__()

    def get_model(self):
        inputs = Input(self.input_shape)

        x = create_fcn_densenet(inputs, nb_layers_per_block=[4, 5, 7, 10, 12, 15])

        model = Model(inputs, x)

        model.summary()
        return model

class UnetSimpleResNet(ImageModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(UnetSimpleResNet, self).__init__()

    def get_model(self):
        inputs = Input(self.input_shape)

        outputs = self.create_unetresnet(inputs)

        model = Model(inputs, outputs)

        model.summary()
        return model

if __name__ == '__main__':
    # FCN.transfer_FCN_Vgg16()
    # unet((224, 224, 1)).transfer_pretrain()
    # ResNet50((32, 32, 1)).get_model()
    # DenseNet((32, 32, 1)).get_model()
    # unet5((448, 448, 1)).get_model()
    # unet_standard((224, 224, 1)).get_model()
    UnetSimpleResNet((224, 224, 1)).get_model()
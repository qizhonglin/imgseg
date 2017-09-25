from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

def create_fcn_densenet(img_input, nb_dense_block=5, growth_rate=16,
                nb_layers_per_block=4, reduction=0.0, dropout_rate=0.0,
                weight_decay=1E-4, init_conv_filters=48, classes=1,
                activation='sigmoid', upsampling_conv=128, upsampling_type='upsampling'):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), \
            'If list, nb_layer is used as provided. ' \
            'Note that list size must be (nb_dense_block + 1)'

        bottleneck_nb_layers = nb_layers[-1]
        rev_layers = nb_layers[::-1]
        nb_layers.extend(rev_layers[1:])
    else:
        bottleneck_nb_layers = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Conv2D(init_conv_filters, (3, 3), kernel_initializer="he_uniform",
               padding="same", name="initial_conv2D", use_bias=False,
               kernel_regularizer=l2(weight_decay))(img_input)

    nb_filter = init_conv_filters

    skip_list = []

    # Add dense blocks and transition down block
    for block_idx in range(nb_dense_block):
        x, nb_filter = __dense_block(x, nb_layers[block_idx],
                                     nb_filter, growth_rate,
                                     dropout_rate=dropout_rate,
                                     weight_decay=weight_decay)

        # Skip connection
        skip_list.append(x)

        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression,
                               dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

        # this is calculated inside transition_down_block
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_down_block
    # return the concatenated feature maps without the concatenation of the
    # input
    _, nb_filter, concat_list = __dense_block(x, bottleneck_nb_layers,
                                              nb_filter,
                                              growth_rate,
                                              dropout_rate=dropout_rate,
                                              weight_decay=weight_decay,
                                              return_concat_list=True)

    skip_list = skip_list[::-1]  # reverse the skip list

    # Add dense blocks and transition up block
    for block_idx in range(nb_dense_block):
        n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]

        # upsampling block must upsample only the
        # feature maps (concat_list[1:]),
        # not the concatenation of the input with the
        # feature maps (concat_list[0]).
        l = concatenate(concat_list[1:], axis=concat_axis)

        t = __transition_up_block(l, nb_filters=n_filters_keep,
                                  type=upsampling_type)

        # concatenate the skip connection with the transition block
        x = concatenate([t, skip_list[block_idx]], axis=concat_axis)

        # Dont allow the feature map size to grow in upsampling dense blocks
        _, nb_filter, concat_list = \
            __dense_block(x,
                          nb_layers[nb_dense_block + block_idx + 1],
                          nb_filter=growth_rate, growth_rate=growth_rate,
                          dropout_rate=dropout_rate,
                          weight_decay=weight_decay,
                          return_concat_list=True, grow_nb_filters=False)

    x = concatenate(concat_list[1:], axis=concat_axis)
    x = Conv2D(classes, (1, 1), activation=activation,
               padding='same', kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)
    return x

def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False,
                  dropout_rate=None, weight_decay=1E-4,
                  grow_nb_filters=True, return_concat_list=False):
    """ Build a dense_block where each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the
            actual output
    Returns: keras tensor with nb_layers of conv_block appended
    """

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        x = __conv_block(x, growth_rate, bottleneck,
                         dropout_rate, weight_decay)
        x_list.append(x)

        x = concatenate(x_list, axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter

def __conv_block(ip, nb_filter, bottleneck=False,
                 dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added
             (optional bottleneck)
    """

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)

    if bottleneck:
        # Obtained from
        # https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
        inter_channel = nb_filter * 4

        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_uniform',
                   padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = BatchNormalization(axis=concat_axis,
                               gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_uniform',
               padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __transition_block(ip, nb_filter, compression=1.0, dropout_rate=None,
                       weight_decay=1E-4, dilation_rate=1, pooling='max',
                       kernel_size=(1, 1)):
    """ Apply BatchNorm, Relu 1x1, Conv2D, compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of
            feature maps in the transition block, is optional.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        dilation_rate: an integer or tuple/list of 2 integers, specifying the
          dilation rate to use for dilated, or atrous convolution.
          Can be a single integer to specify the same value for all
          spatial dimensions.
        pooling: Data pooling to reduce resolution,
            one of "avg", "max", or None.
    Returns:

        keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    """

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter * compression), kernel_size,
               kernel_initializer='he_uniform', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               dilation_rate=dilation_rate)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    if pooling == 'avg':
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif pooling == 'max':
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return x

def __transition_up_block(ip, nb_filters, type='upsampling',
                          weight_decay=1E-4):
    """ SubpixelConvolutional Upscaling (factor = 2)
    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', or 'deconv'. Determines type of
            upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    """

    if type == 'upsampling':
        x = UpSampling2D()(ip)
    # elif type == 'subpixel':
    #     x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same',
    #                kernel_regularizer=l2(weight_decay),
    #                use_bias=False, kernel_initializer='he_uniform')(ip)
    #     x = SubPixelUpscaling(scale_factor=2)(x)
    #     x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same',
    #                kernel_regularizer=l2(weight_decay),
    #                use_bias=False, kernel_initializer='he_uniform')(x)
    # else:
    #     x = Conv2DTranspose(nb_filters, (3, 3), output_shape,
    #                         activation='relu', padding='same',
    #                         subsample=(2, 2),
    #                         kernel_initializer='he_uniform')(ip)

    return x

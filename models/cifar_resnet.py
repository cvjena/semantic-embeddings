# -*- coding: utf-8 -*-
"""ResNet model for CIFAR.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras import layers, regularizers
from keras import backend as K
from keras.layers import Input
from keras.layers import Dense, Activation, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.models import Model
from keras.engine import Layer, InputSpec
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils, conv_utils
from keras.utils.data_utils import get_file
try:
    from keras.utils.conv_utils import normalize_data_format
except ImportError:
    from keras.backend import normalize_data_format


class ChannelPadding(Layer):
    """Zero-padding on channel axis.
    # Arguments
        padding: int, or tuple of int (length 2)
            - If int:
            How many zeros to add at the beginning and end of
            the padding dimension (axis 1).
            - If tuple of int (length 2):
            How many zeros to add at the beginning and at the end of
            the padding dimension (`(left_pad, right_pad)`).
    """

    def __init__(self, padding=1, data_format=None, **kwargs):
        super(ChannelPadding, self).__init__(**kwargs)
        self.padding = conv_utils.normalize_tuple(padding, 2, 'padding')
        self.data_format = normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        axis = 1 if self.data_format == 'channels_first' else -1
        if input_shape[axis] is None:
            return input_shape
        else:
            length = input_shape[axis] + self.padding[0] + self.padding[1]
            if axis == 1:
                return input_shape[:1] + (length,) + input_shape[2:]
            else:
                return input_shape[:-1] + (length,)

    def call(self, inputs):
        pattern = [[0,0] for i in range(len(inputs.shape))]
        axis = 1 if self.data_format == 'channels_first' else -1
        pattern[axis] = self.padding
        return K.tf.pad(inputs, pattern)

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(ChannelPadding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def simple_block(input_tensor, filters, prefix, kernel_size = 3, stride = 1,
                 regularizer = None, activation = 'relu', conv_shortcut = False, bn = True):
    """A block with shortcut connection.

    # Arguments
        input_tensor: input tensor
        filters: tuple with number of input and output channels
        prefix: prefix of layer names
        kernel_size: default 3, the kernel size of conv layers
        stride: stride of first conv layer in the block
        regularizer: kernel regularizer
        activation: name of the activation function to be used
        conv_shortcut: boolean, specifying whether to use padding (False) or
            convolution (True) at the shortcut
        bn: boolean specifying whether to include BatchNormalization layers

    # Returns
        Output tensor for the block.
    """
    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + prefix
    bn_name_base = 'bn' + prefix

    x = Conv2D(filters[1], kernel_size, padding='same', strides=(stride, stride),
               kernel_regularizer = regularizer,
               name=conv_name_base + 'x')(input_tensor)
    if bn:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'x')(x)
    x = Activation(activation)(x)

    x = Conv2D(filters[1], kernel_size, padding='same',
               kernel_regularizer = regularizer,
               name=conv_name_base + 'y')(x)
    if bn:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'y')(x)

    shortcut = input_tensor
    if (filters[0] != filters[1]) and conv_shortcut:
        shortcut = Conv2D(filters[1], (1, 1), strides=(stride, stride),
                          kernel_regularizer = regularizer,
                          name=conv_name_base + 'z')(shortcut)
        if bn:
            shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + 'z')(shortcut)
    else:
        if stride > 1:
            shortcut = AveragePooling2D((stride, stride), name='avg'+prefix)(shortcut)
        if filters[0] < filters[1]:
            shortcut = ChannelPadding(((filters[1] - filters[0]) // 2, filters[1] - filters[0] - (filters[1] - filters[0]) // 2),
                                     name = 'pad'+prefix)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation(activation)(x)
    return x


def unit(input_tensor, filters, n, prefix, kernel_size = 3, stride = 1, **kwargs):
    """A stack of blocks.

    # Arguments
        input_tensor: input tensor
        filters: tuple with number of input and output channels
        n: number of blocks in the unit
        prefix: prefix of layer names
        kernel_size: default 3, the kernel size of conv layers
        stride: stride of first conv layer in the unit

    # Returns
        Output tensor for the block.
    """
    
    x = simple_block(input_tensor, filters, prefix + '1', kernel_size=kernel_size, stride=stride, **kwargs)
    for i in range(1, n):
        x = simple_block(x, [filters[1], filters[1]], prefix + str(i+1), kernel_size=kernel_size, **kwargs)
    return x


def SmallResNet(n = 9, filters = [16, 32, 64],
                include_top=True, weights=None,
                input_tensor=None, input_shape=None,
                pooling='avg', regularizer=regularizers.l2(0.0002), activation = 'relu',
                top_activation='softmax',
                conv_shortcut=False, bn=True,
                classes=100, name=None):
    """Instantiates the CIFAR ResNet architecture described in section 4.2 of the paper.

    # Arguments
        n: number of blocks in each unit
        filters: list of number of filters in each unit
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization)
            or path to weights file.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(32, 32, 3)` (with `channels_last` data format)
            or `(3, 32, 32)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        regularizer: weight of kernel regularizer.
        activation: name of the activation function to be used.
        top_activation: name of the activation function to be used for the top layer.
        conv_shortcut: boolean, specifying whether to use padding (False) or
            convolution (True) at the shortcuts.
        bn: boolean specifying whether to include BatchNormalization layers.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    
    # Determine proper input shape
    if input_shape is None:
        if K.image_data_format() == 'channels_first':
            input_shape = (3, 32, 32) if include_top and pooling is None else (3, None, None)
        else:
            input_shape = (32, 32, 3) if include_top and pooling is None else (None, None, 3)

    # Build network
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(filters[0], (3, 3), padding='same', name='conv0', kernel_regularizer = regularizer)(img_input)
    if bn:
        x = BatchNormalization(axis=bn_axis, name='bn0')(x)
    x = Activation(activation)(x)
    
    x = unit(x, [filters[0], filters[0]], n, '1-', kernel_size = 3, stride = 1, regularizer=regularizer, activation=activation, conv_shortcut=conv_shortcut, bn=bn)
    for i in range(1, len(filters)):
        x = unit(x, [filters[i-1], filters[i]], n, str(i+1)+'-', kernel_size = 3, stride = 2, regularizer=regularizer, activation=activation, conv_shortcut=conv_shortcut, bn=bn)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='max_pool')(x)

    if include_top:
        x = Dense(classes, activation=top_activation, name = 'embedding' if top_activation is None else 'prob', kernel_regularizer = regularizer)(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cifar-resnet{}'.format(2*len(filters)*n) if name is None else name)

    # load weights
    if weights is not None:
        model.load_weights(weights)

    if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
        warnings.warn('You are using the TensorFlow backend, yet you '
                      'are using the Theano '
                      'image data format convention '
                      '(`image_data_format="channels_first"`). '
                      'For best performance, set '
                      '`image_data_format="channels_last"` in '
                      'your Keras config '
                      'at ~/.keras/keras.json.')
    return model

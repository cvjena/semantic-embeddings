# -*- coding: utf-8 -*-
"""PyramidNet model for CIFAR.

# Reference:

- [Deep Pyramidal Residual Networks](https://arxiv.org/abs/1610.02915)

# Reference implementation:

- https://github.com/jhkim89/PyramidNet/blob/master/addpyramidnet.lua
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras import layers, regularizers
from keras import backend as K
from keras.layers import Input
from keras.layers import Dense, Activation, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.engine import Layer, InputSpec
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils, conv_utils
from keras.utils.data_utils import get_file

from .cifar_resnet import ChannelPadding



def PyramidNet(depth, alpha, bottleneck = True,
               include_top=True, weights=None,
               input_tensor=None, input_shape=None,
               pooling='avg', regularizer=regularizers.l2(0.0002),
               activation = 'relu', top_activation='softmax',
               classes=100, name=None):
    """Instantiates the PyramidNet architecture.

    # Arguments
        depth: depth of the network.
        alpha: total number of channels to be distributed across the layers.
        bottleneck: boolean specifying whether to use bottleneck blocks.
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
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        name: name of the network.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    
    
    def shortcut(x, n, stride):
        if stride > 1:
            x = AveragePooling2D(stride)(x)
        input_channels = int(x.shape[1 if K.image_data_format() == 'channels_first' else -1])
        if input_channels < n:
            x = ChannelPadding((0, n - input_channels))(x)
        return x
    
    
    def basic_block(x, n, stride):
        s = BatchNormalization()(x)
        s = Conv2D(n, (3, 3), strides = stride, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = regularizer)(s)
        s = BatchNormalization()(s)
        s = Activation(activation)(s)
        s = Conv2D(n, (3, 3), padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = regularizer)(s)
        s = BatchNormalization()(s)
        return layers.add([s, shortcut(x, n, stride)])
    
    
    def bottleneck_block(x, n, stride):
        s = BatchNormalization()(x)
        s = Conv2D(n, (1, 1), kernel_initializer = 'glorot_normal', kernel_regularizer = regularizer)(s)
        s = BatchNormalization()(s)
        s = Activation(activation)(s)
        s = Conv2D(n, (3, 3), strides = stride, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = regularizer)(s)
        s = BatchNormalization()(s)
        s = Activation(activation)(s)
        s = Conv2D(n*4, (1, 1), kernel_initializer = 'glorot_normal', kernel_regularizer = regularizer)(s)
        s = BatchNormalization()(s)
        return layers.add([s, shortcut(x, n * 4, stride)])
    
    
    def unit(x, features, count, stride):
        block = bottleneck_block if bottleneck else basic_block
        for i in range(count):
            x = block(x, features, stride)
        return x
    
    
    # Derived parameters
    n = (depth - 2) // 9 if bottleneck else (depth - 2) // 6
    channels = 16
    start_channel = 16
    add_channel = float(alpha) / (3*n)
    
    # Determine proper input shape
    if input_shape is None:
        if K.image_data_format() == 'channels_first':
            input_shape = (3, 32, 32) if include_top else (3, None, None)
        else:
            input_shape = (32, 32, 3) if include_top else (None, None, 3)

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

    x = Conv2D(start_channel, (3, 3), padding='same', name='conv0', kernel_initializer = 'glorot_normal', kernel_regularizer = regularizer)(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn0')(x)
    
    for b in range(3):
        start_channel += add_channel
        x = unit(x, round(start_channel), 1, 1 if b == 0 else 2)
        for i in range(1, n):
            start_channel += add_channel
            x = unit(x, round(start_channel), 1, 1)
    
    x = BatchNormalization(axis=bn_axis, name='bn4')(x)
    x = Activation(activation, name='act4')(x)

    # Final pooling
    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='max_pool')(x)

    # Top layer
    if include_top:
        x = Dense(classes, activation=top_activation, name = 'embedding' if top_activation is None else 'prob', kernel_regularizer = regularizer)(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='pyramidnet-{}-{}'.format(depth, alpha) if name is None else name)

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

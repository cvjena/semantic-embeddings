""" Modified VGG16 model for Keras (+ batchnorm, avg pooling instead of max pooling, + global avg pooling). """

from __future__ import print_function
from __future__ import absolute_import

import os
import warnings

from keras.models import Model
from keras.layers import Flatten, BatchNormalization, Dense, Input, Conv2D, AvgPool2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K



def VGG16(include_top=True, final_activation=None, input_tensor=None, classes=1000):
    """Instantiates the VGG16 architecture.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        final_activation: activation function of last layer.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if input_tensor is None:
        img_input = Input(shape=(None, None, 3))
    elif not K.is_keras_tensor(input_tensor):
        img_input = Input(tensor=input_tensor, shape=(None, None, 3))
    else:
        img_input = input_tensor
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_bn1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = AvgPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = BatchNormalization(name='block2_bn1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization(name='block2_bn2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = AvgPool2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = BatchNormalization(name='block3_bn1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization(name='block3_bn2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization(name='block3_bn3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = AvgPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = BatchNormalization(name='block4_bn1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = BatchNormalization(name='block4_bn2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization(name='block4_bn3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = AvgPool2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = BatchNormalization(name='block5_bn1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = BatchNormalization(name='block5_bn2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = BatchNormalization(name='block5_bn3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    
    x = GlobalAveragePooling2D(name='avg_pool')(x)

    if include_top:
        # Classification block
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation=final_activation, name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    return model

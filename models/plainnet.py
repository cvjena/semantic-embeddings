import keras
from keras import backend as K


def PlainNet(output_dim,
             filters = [64, 64, 'ap', 128, 128, 128, 'ap', 256, 256, 256, 'ap', 512, 'gap', 'fc512'],
             activation = 'relu',
             regularizer = keras.regularizers.l2(0.0005),
             final_activation = None,
             input_shape = (32, 32, 3),
             pool_size = (2,2),
             name = None):
    """ Creates the Plain-11 network architecture.

    # Reference:

    Björn Barz and Joachim Denzler:
    "Deep Learning is not a Matter of Depth but of Good Training."
    International Conference on Pattern Recognition and Artificial Intelligence (ICPRAI), pp. 683-687, 2018.
    http://hera.inf-cv.uni-jena.de:6680/pdf/Barz18:GoodTraining

    # Arguments:

    - output_dim: Number of output units of the final layer.

    - filters: List specifying the individual layers constituting the network architecture.
               BatchNormalization will be inserted automatically.
               Possible types of values for the items of this list are:
                   - int: a 3x3 Conv2D layer with this number of channels.
                   - 'fc'+int: fully-connected layer with the specified number of units.
                   - 'ap': average pooling.
                   - 'mp': maximum pooling.
                   - 'gap': global average pooling.

    - activation: Activation function of Conv2D and Dense layers.

    - regularizer: Kernel regularizer of Conv2D and Dense layers.

    - final_activation: Activation function of the final layer.

    - input_shape: 3-tuple specifying the shape of the input tensor.

    - pool_size: 2-tuple specifying the kernel size of average pooling and maximum pooling layers.

    - name: Name of the network.
    """
    
    prefix = '' if name is None else name + '_'
    
    flattened = False
    layers = [
        keras.layers.Conv2D(filters[0], (3, 3), padding = 'same', activation = activation, kernel_regularizer = regularizer, input_shape = input_shape, name = prefix + 'conv1'),
        keras.layers.BatchNormalization(name = prefix + 'bn1')
    ]
    for i, f in enumerate(filters[1:], start = 2):
        if f == 'mp':
            layers.append(keras.layers.MaxPooling2D(pool_size = pool_size, name = '{}mp{}'.format(prefix, i)))
        elif f == 'ap':
            layers.append(keras.layers.AveragePooling2D(pool_size = pool_size, name = '{}ap{}'.format(prefix, i)))
        elif f == 'gap':
            layers.append(keras.layers.GlobalAvgPool2D(name = prefix + 'avg_pool'))
            flattened = True
        elif isinstance(f, str) and f.startswith('fc'):
            if not flattened:
                layers.append(keras.layers.Flatten(name = prefix + 'flatten'))
                flattened = True
            layers.append(keras.layers.Dense(int(f[2:]), activation = activation, kernel_regularizer = regularizer, name = '{}fc{}'.format(prefix, i)))
            layers.append(keras.layers.BatchNormalization(name = '{}bn{}'.format(prefix, i)))
        else:
            layers.append(keras.layers.Conv2D(f, (3, 3), padding = 'same', activation = activation, kernel_regularizer = regularizer, name = '{}conv{}'.format(prefix, i)))
            layers.append(keras.layers.BatchNormalization(name = '{}bn{}'.format(prefix, i)))
    
    if not flattened:
        layers.append(keras.layers.Flatten(name = prefix + 'flatten'))
        flattened = True
    layers.append(keras.layers.Dense(output_dim, activation = final_activation, name = prefix + ('prob' if final_activation == 'softmax' else 'embedding')))
    
    return keras.models.Sequential(layers, name = name)
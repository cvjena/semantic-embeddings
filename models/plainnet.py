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
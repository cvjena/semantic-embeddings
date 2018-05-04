import sys, os.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'DenseNet'))

import numpy as np

import keras
from keras import backend as K

import warnings

from models import cifar_resnet, cifar_pyramidnet, vgg16, wide_residual_network as wrn
import densenet  # pylint: disable=import-error
from clr_callback import CyclicLR
from sgdr_callback import SGDR



ARCHITECTURES = ['simple', 'simple-mp', 'resnet-32', 'resnet-110', 'resnet-110-fc', 'wrn-28-10', 'densenet-100-12', 'pyramidnet-272-200', 'pyramidnet-110-270', 'vgg16', 'resnet-50']

LR_SCHEDULES = ['SGD', 'SGDR', 'CLR', 'ResNet-Schedule']



def squared_distance(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)


def mean_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def nn_accuracy(embedding, dot_prod_sim = False):

    def nn_accuracy(y_true, y_pred):

        centroids = K.constant(embedding.T)
        centroids_norm = K.constant((embedding.T ** 2).sum(axis = 0, keepdims = True))
        pred_norm = K.sum(K.square(y_pred), axis = 1, keepdims = True)
        dist = pred_norm + centroids_norm - 2 * K.dot(y_pred, centroids)

        true_dist = K.sum(K.square(y_pred - y_true), axis = -1)

        return K.cast(K.less(K.abs(true_dist - K.min(dist, axis = -1)), 1e-6), K.floatx())
    
    def max_sim_acc(y_true, y_pred):

        centroids = K.constant(embedding.T)
        sim = K.dot(y_pred, centroids)
        true_sim = K.sum(y_pred * y_true, axis = -1)
        return K.cast(K.less(K.abs(K.max(sim, axis = -1) - true_sim), 1e-6), K.floatx())
    
    return max_sim_acc if dot_prod_sim else nn_accuracy


def devise_ranking_loss(embedding, margin = 0.1):
    
    def _loss(y_true, y_pred):
        embedding_t = K.constant(embedding.T)
        true_sim = K.sum(y_true * y_pred, axis = -1)
        other_sim = K.dot(y_pred, embedding_t)
        return K.sum(K.relu(margin - true_sim[:,None] + other_sim), axis = -1) - margin
    
    return _loss


def build_simplenet(output_dim, filters, activation = 'relu', regularizer = keras.regularizers.l2(0.0005), final_activation = None, name = None):
    
    prefix = '' if name is None else name + '_'
    
    flattened = False
    layers = [
        keras.layers.Conv2D(filters[0], (3, 3), padding = 'same', activation = activation, kernel_regularizer = regularizer, input_shape = (32, 32, 3), name = prefix + 'conv1'),
        keras.layers.BatchNormalization(name = prefix + 'bn1')
    ]
    for i, f in enumerate(filters[1:], start = 2):
        if f == 'mp':
            layers.append(keras.layers.MaxPooling2D(pool_size = (2,2), name = '{}mp{}'.format(prefix, i)))
        elif f == 'ap':
            layers.append(keras.layers.AveragePooling2D(pool_size = (2,2), name = '{}ap{}'.format(prefix, i)))
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


def build_network(num_outputs, architecture, classification = False, name = None):
    """ Constructs a CNN.
    
    # Arguments:
    
    - num_outputs: number of final output units.
    
    - architecture: name of the architecture. See ARCHITECTURES for a list of possible values.
    
    - classification: If `True`, the final layer will have a softmax activation, otherwise no activation at all.
    
    - name: The name of the network.
    
    # Returns:
        keras.models.Model
    """
    
    # CIFAR-100 architectures
    
    if architecture == 'resnet-32':
        
        return cifar_resnet.SmallResNet(5, filters = [16, 32, 64] if classification else [32, 64, num_outputs],
                                        include_top = classification, classes = num_outputs, name = name)
        
    elif architecture == 'resnet-110':
        
        return cifar_resnet.SmallResNet(18, filters = [16, 32, 64] if classification else [32, 64, num_outputs],
                                        include_top = classification, classes = num_outputs, name = name)
    
    elif architecture == 'resnet-110-fc':
        
        return cifar_resnet.SmallResNet(18, filters = [32, 64, 128],
                                        include_top = True, top_activation = 'softmax' if classification else None,
                                        classes = num_outputs, name = name)
    
    elif architecture == 'wrn-28-10':
        
        return wrn.create_wide_residual_network((32, 32, 3), nb_classes = num_outputs, N = 4, k = 10, verbose = 0,
                                                final_activation = 'softmax' if classification else None, name = name)
        
    elif architecture == 'densenet-100-12':
        
        return densenet.DenseNet(growth_rate = 12, depth = 100, bottleneck = False,
                                 classes = num_outputs, activation = 'softmax' if classification else None, name = name)
    
    elif architecture == 'pyramidnet-272-200':
        
        return cifar_pyramidnet.PyramidNet(272, 200, bottleneck = True,
                                           classes = num_outputs, top_activation = 'softmax' if classification else None, name = name)
    
    elif architecture == 'pyramidnet-110-270':
        
        return cifar_pyramidnet.PyramidNet(110, 270, bottleneck = False,
                                           classes = num_outputs, top_activation = 'softmax' if classification else None, name = name)
        
    elif architecture == 'simple':
        
        return build_simplenet(num_outputs, [64, 64, 'ap', 128, 128, 128, 'ap', 256, 256, 256, 'ap', 512, 'gap', 'fc512'],
                               final_activation = 'softmax' if classification else None,
                               name = name)
    
    elif architecture == 'simple-mp':
        
        return build_simplenet(num_outputs, [64, 64, 'mp', 128, 128, 128, 'mp', 256, 256, 256, 'mp', 512, 'gap', 'fc512'],
                               final_activation = 'softmax' if classification else None,
                               name = name)
    
    # ImageNet architectures
    
    elif architecture == 'vgg16':
        
        return vgg16.VGG16(classes = num_outputs, final_activation = 'softmax' if classification else None)
    
    elif architecture == 'resnet-50':
        
        rn50 = keras.applications.ResNet50(include_top=False, weights=None)
        x = keras.layers.GlobalAvgPool2D(name='avg_pool')(rn50.layers[-2].output)
        x = keras.layers.Dense(num_outputs, activation = 'softmax' if classification else None, name = 'prob' if classification else 'embedding')(x)
        return keras.models.Model(rn50.inputs, x, name=name)
    
    else:
        
        raise ValueError('Unknown network architecture: {}'.format(architecture))


def get_custom_objects(architecture):
    
    if architecture in ('resnet-32', 'resnet-110', 'resnet-110-fc', 'pyramidnet-272-200', 'pyramidnet-110-270'):
        return { 'ChannelPadding' : cifar_resnet.ChannelPadding }
    else:
        return {}


def get_lr_schedule(schedule, num_samples, batch_size, schedule_args = {}):

    if schedule.lower() == 'sgd':
    
        if 'sgd_patience' not in schedule_args:
            schedule_args['sgd_patience'] = 10
        if 'sgd_min_lr' not in schedule_args:
            schedule_args['sgd_min_lr'] = 1e-4
        return [
            keras.callbacks.ReduceLROnPlateau('val_loss', patience = schedule_args['sgd_patience'], epsilon = 1e-4, min_lr = schedule_args['sgd_min_lr'], verbose = True)
        ], 200
    
    elif schedule.lower() == 'sgdr':
    
        if 'sgdr_base_len' not in schedule_args:
            schedule_args['sgdr_base_len'] = 12
        if 'sgdr_mul' not in schedule_args:
            schedule_args['sgdr_mul'] = 2
        if 'sgdr_max_lr' not in schedule_args:
            schedule_args['sgdr_max_lr'] = 0.1
        return (
            [SGDR(1e-6, schedule_args['sgdr_max_lr'], schedule_args['sgdr_base_len'], schedule_args['sgdr_mul'])],
            sum(schedule_args['sgdr_base_len'] * (schedule_args['sgdr_mul'] ** i) for i in range(5))
        )
        
    elif schedule.lower() == 'clr':
    
        if 'clr_step_len' not in schedule_args:
            schedule_args['clr_step_len'] = 12
        if 'clr_min_lr' not in schedule_args:
            schedule_args['clr_min_lr'] = 1e-5
        if 'clr_max_lr' not in schedule_args:
            schedule_args['clr_max_lr'] = 0.1
        return (
            [CyclicLR(schedule_args['clr_min_lr'], schedule_args['clr_max_lr'], schedule_args['clr_step_len'] * (num_samples // batch_size), mode = 'triangular')],
            schedule_args['clr_step_len'] * 20
        )
    
    elif schedule.lower() == 'resnet-schedule':
    
        def resnet_scheduler(epoch):
            if epoch >= 120:
                return 0.001
            elif epoch >= 80:
                return 0.01
            elif epoch >= 1:
                return 0.1
            else:
                return 0.01
        
        return [keras.callbacks.LearningRateScheduler(resnet_scheduler)], 164
    
    else:
    
        raise ValueError('Unknown learning rate schedule: {}'.format(schedule))



class TemplateModelCheckpoint(keras.callbacks.ModelCheckpoint):
    """Saves a given model after each epoch (for multi GPU training). """

    def __init__(self, tpl_model, filepath, *args, **kwargs):

        super(TemplateModelCheckpoint, self).__init__(filepath, *args, **kwargs)
        self.tpl_model = tpl_model


    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.tpl_model.save_weights(filepath, overwrite=True)
                        else:
                            self.tpl_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.tpl_model.save_weights(filepath, overwrite=True)
                else:
                    self.tpl_model.save(filepath, overwrite=True)

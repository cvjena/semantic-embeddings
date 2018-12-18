import sys, os.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'DenseNet'))

import numpy as np

import keras
from keras import backend as K

import warnings

from models import cifar_resnet, cifar_pyramidnet, plainnet, wide_residual_network as wrn
import densenet  # pylint: disable=import-error
from clr_callback import CyclicLR
from sgdr_callback import SGDR



ARCHITECTURES = ['simple', 'resnet-32', 'resnet-110', 'resnet-110-fc', 'resnet-110-wfc', 'wrn-28-10',
                 'densenet-100-12', 'densenet-100-24', 'densenet-bc-190-40', 'pyramidnet-272-200', 'pyramidnet-110-270',
                 'resnet-50', 'rn18', 'rn34', 'rn50', 'rn101', 'rn152', 'rn200', 'nasnet-a']

LR_SCHEDULES = ['SGD', 'SGDR', 'CLR', 'ResNet-Schedule']



def squared_distance(y_true, y_pred):
    """ Computes the squared Euclidean distance between corresponding pairs of samples in two tensors. """
    return K.sum(K.square(y_pred - y_true), axis=-1)


def mean_distance(y_true, y_pred):
    """ Computes the Euclidean distance between corresponding pairs of samples in two tensors. """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def inv_correlation(y_true, y_pred):
    """ Computes 1 minus the dot product between corresponding pairs of samples in two tensors. """
    return 1. - K.sum(y_true * y_pred, axis = -1)


def nn_accuracy(embedding, dot_prod_sim = False):
    """ Metric computing classification accuracy by assigning samples to the class with the nearest embedding in feature space.

    # Arguments:

    - embedding: 2-d numpy array whose rows are class embeddings.

    - dot_prod_sim: If True, the dot product will be used to find the most similar embedding (assumes L2-normalized embeddings and features).
                    Otherwise, Euclidean distance will be used.
    
    # Returns:
        a Keras metric function taking y_true and y_pred as inputs and returning a tensor of sample-wise accuracies.
    """

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
    """ The ranking loss used by DeViSE.

    # Arguments:

    - embedding: 2-d numpy array whose rows are class embeddings.

    - margin: margin for the ranking loss.

    # Returns:
        a Keras loss function taking y_true and y_pred as inputs and returning a loss tensor.
    """
    
    def _loss(y_true, y_pred):
        embedding_t = K.constant(embedding.T)
        true_sim = K.sum(y_true * y_pred, axis = -1)
        other_sim = K.dot(y_pred, embedding_t)
        return K.sum(K.relu(margin - true_sim[:,None] + other_sim), axis = -1) - margin
    
    return _loss


def l2norm(x):
    """ L2-normalizes a tensor along the last axis. """
    return K.tf.nn.l2_normalize(x, -1)


def build_network(num_outputs, architecture, classification = False, no_softmax = False, name = None):
    """ Constructs a CNN.
    
    # Arguments:
    
    - num_outputs: number of final output units.
    
    - architecture: name of the architecture. See ARCHITECTURES for a list of possible values and README.md for descriptions.
    
    - classification: If `True`, the final layer will have a softmax activation, otherwise no activation at all.

    - no_softmax: Usually, the last layer will have a softmax activation if `classification` is True. However, if `no_softmax` is set
                  to True as well, the last layer will not have any activation.
    
    - name: The name of the network.
    
    # Returns:
        keras.models.Model
    """
    
    if architecture.lower().endswith('-selu'):
        activation = 'selu'
        architecture = architecture[:-5]
    else:
        activation = 'relu'
    
    # CIFAR-100 architectures
    
    if architecture == 'resnet-32':
        
        return cifar_resnet.SmallResNet(5, filters = [16, 32, 64], activation = activation,
                                        include_top = classification, top_activation = None if no_softmax else 'softmax',
                                        classes = num_outputs, name = name)
        
    elif architecture == 'resnet-110':
        
        return cifar_resnet.SmallResNet(18, filters = [16, 32, 64], activation = activation,
                                        include_top = classification, top_activation = None if no_softmax else 'softmax',
                                        classes = num_outputs, name = name)
    
    elif architecture == 'resnet-110-fc':
        
        return cifar_resnet.SmallResNet(18, filters = [16, 32, 64], activation = activation,
                                        include_top = True, top_activation = 'softmax' if classification and (not no_softmax) else None,
                                        classes = num_outputs, name = name)
    
    elif architecture == 'resnet-110-wfc':
        
        return cifar_resnet.SmallResNet(18, filters = [32, 64, 128], activation = activation,
                                        include_top = True, top_activation = 'softmax' if classification and (not no_softmax) else None,
                                        classes = num_outputs, name = name)
    
    elif architecture == 'wrn-28-10':
        
        return wrn.create_wide_residual_network((32, 32, 3), nb_classes = num_outputs, N = 4, k = 10, verbose = 0,
                                                final_activation = 'softmax' if classification and (not no_softmax) else None, name = name)
        
    elif architecture == 'densenet-100-12':
        
        return densenet.DenseNet(growth_rate = 12, depth = 100, nb_dense_block = 3, bottleneck = False, nb_filter = 16, reduction = 0.0,
                                 classes = num_outputs, activation = 'softmax' if classification and (not no_softmax) else None, name = name)
    
    elif architecture == 'densenet-100-24':
        
        return densenet.DenseNet(growth_rate = 24, depth = 100, nb_dense_block = 3, bottleneck = False, nb_filter = 16, reduction = 0.0,
                                 classes = num_outputs, activation = 'softmax' if classification and (not no_softmax) else None, name = name)
    
    elif architecture == 'densenet-bc-190-40':
        
        return densenet.DenseNet(growth_rate = 40, depth = 190, nb_dense_block = 3, bottleneck = True, nb_filter = -1, reduction = 0.5,
                                 classes = num_outputs, activation = 'softmax' if classification and (not no_softmax) else None, name = name)
    
    elif architecture == 'pyramidnet-272-200':
        
        return cifar_pyramidnet.PyramidNet(272, 200, bottleneck = True, activation = activation,
                                           classes = num_outputs, top_activation = 'softmax' if classification and (not no_softmax) else None, name = name)
    
    elif architecture == 'pyramidnet-110-270':
        
        return cifar_pyramidnet.PyramidNet(110, 270, bottleneck = False, activation = activation,
                                           classes = num_outputs, top_activation = 'softmax' if classification and (not no_softmax) else None, name = name)
        
    elif architecture == 'simple':
        
        return plainnet.PlainNet(num_outputs,
                                 activation = activation,
                                 final_activation = 'softmax' if classification and (not no_softmax) else None,
                                 name = name)
    
    # ImageNet architectures
    
    elif architecture == 'resnet-50':
        
        rn50 = keras.applications.ResNet50(include_top=False, weights=None)
        rn50_out = rn50.layers[-2].output if isinstance(rn50.layers[-1], keras.layers.AveragePooling2D) else rn50.layers[-1].output
        x = keras.layers.GlobalAvgPool2D(name='avg_pool')(rn50_out)
        x = keras.layers.Dense(num_outputs, activation = 'softmax' if classification and (not no_softmax) else None, name = 'prob' if classification else 'embedding')(x)
        return keras.models.Model(rn50.inputs, x, name=name)
    
    elif architecture.startswith('rn'):

        import keras_resnet.models
        factories = {
            'rn18'  : keras_resnet.models.ResNet18,
            'rn34'  : keras_resnet.models.ResNet34,
            'rn50'  : keras_resnet.models.ResNet50,
            'rn101' : keras_resnet.models.ResNet101,
            'rn152' : keras_resnet.models.ResNet152,
            'rn200' : keras_resnet.models.ResNet200
        }
        input_ = keras.layers.Input((3, None, None)) if K.image_data_format() == 'channels_first' else keras.layers.Input((None, None, 3))
        rn = factories[architecture](input_, include_top = classification and (not no_softmax), classes = num_outputs, freeze_bn = False, name = name)
        if (not classification) or no_softmax:
            x = keras.layers.GlobalAvgPool2D(name = 'avg_pool')(rn.outputs[-1])
            x = keras.layers.Dense(num_outputs, name = 'prob' if classification else 'embedding', activation = None if no_softmax else 'softmax')(x)
            rn = keras.models.Model(input_, x, name = name)
        return rn
    
    elif architecture == 'nasnet-a':
        
        nasnet = keras.applications.NASNetLarge(include_top=False, input_shape=(224,224,3), weights=None, pooling='avg')
        x = keras.layers.Dense(num_outputs, activation = 'softmax' if classification and (not no_softmax) else None, name = 'prob' if classification else 'embedding')(nasnet.output)
        return keras.models.Model(nasnet.inputs, x, name=name)
    
    else:
        
        raise ValueError('Unknown network architecture: {}'.format(architecture))


def get_custom_objects(architecture):
    """ Provides a dictionary with custom objects required for loading a certain model architecture using `keras.models.load_model`. """
    
    if architecture in ('resnet-32', 'resnet-110', 'resnet-110-fc', 'resnet-110-wfc', 'pyramidnet-272-200', 'pyramidnet-110-270'):
        return { 'ChannelPadding' : cifar_resnet.ChannelPadding }
    else:
        return {}


def get_lr_schedule(schedule, num_samples, batch_size, schedule_args = {}):
    """ Creates a learning rate schedule.

    # Arguments:

    - schedule: Name of the schedule. Possible values:
                - 'sgd': Stochastic Gradient Descent with ReduceLROnPlateau or LearningRateSchedule callback.
                - 'sgdr': Stochastic Gradient Descent with Cosine Annealing and Warm Restarts.
                - 'clr': Cyclical Learning Rates.
                - 'resnet-schedule': Hand-crafted schedule used by He et al. for training ResNet.
    
    - num_samples: Number of training samples.

    - batch_size: Number of samples per batch.

    - schedule_args: Further arguments for the specific learning rate schedule.
                     'sgd' supports:
                        - 'sgd_patience': Number of epochs without improvement before reducing the LR. Default: 10.
                        - 'sgd_min_lr': Minimum learning rate. Default : 1e-4
                        - 'sgd_schedule': Comma-separated list of `epoch:lr` pairs, defining a learning rate schedule.
                                          The total number of epochs can be appended to this list, separated by a comma as well.
                                          If this is specified, the learning rate will not be reduced on plateaus automatically
                                          and `sgd_patience` and `sgd_min_lr` will be ignored.
                                          The following example would mean to train for 50 epochs, starting with a learning rate
                                          of 0.1 and reducing it by a factor of 10 after 30 and 40 epochs: "1:0.1,31:0.01,41:0.001,50".
                     'sgdr' supports:
                        - 'sgdr_base_len': Length of the first cycle. Default: 12.
                        - 'sgdr_mul': Factor multiplied with the length of the cycle after the end of each one. Default: 2.
                        - 'sgdr_max_lr': Initial learning rate at the beginning of each cycle. Default: 0.1.
                     'clr' supports:
                        - 'clr_step_len': Number of training epochs per half-cycle. Default: 12.
                        - 'clr_min_lr': Minimum learning rate. Default: 1e-5.
                        - 'clr_max_lr': Maximum learning rate: Default: 0.1.
    
    # Returns:
        - a list of callbacks for being passed to the fit function,
        - a suggested number of training epochs.
    """

    if schedule.lower() == 'sgd':
    
        if ('sgd_schedule' in schedule_args) and (schedule_args['sgd_schedule'] is not None) and (schedule_args['sgd_schedule'] != ''):

            def lr_scheduler(schedule, epoch, cur_lr):
                if schedule[0][0] > epoch:
                    return cur_lr
                for i in range(1, len(schedule)):
                    if schedule[i][0] > epoch:
                        return schedule[i-1][1] if schedule[i-1][1] is not None else cur_lr
                return schedule[-1][1] if schedule[-1][1] is not None else cur_lr
            
            schedule = [(int(point[0]) - 1, float(point[1]) if len(point) > 1 else None)
                        for sched_tuple in schedule_args['sgd_schedule'].split(',') for point in [sched_tuple.split(':')]]
            schedule.sort()
            return [keras.callbacks.LearningRateScheduler(
                lambda ep, cur_lr: lr_scheduler(schedule, ep, cur_lr)
            )], schedule[-1][0] + 1

        else:

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


def add_lr_schedule_arguments(parser):
    """ Adds common command-line arguments for controlling different learning rate schedules to a given `argparse.ArgumentParser`. """

    arggroup = parser.add_argument_group('Parameters for --lr_schedule=SGD')
    arggroup.add_argument('--sgd_patience', type = int, default = None, help = 'Patience of learning rate reduction in epochs.')
    arggroup.add_argument('--sgd_lr', type = float, default = 0.1, help = 'Initial learning rate.')
    arggroup.add_argument('--sgd_min_lr', type = float, default = None, help = 'Minimum learning rate.')
    arggroup.add_argument('--sgd_schedule', type = str, default = None,
                          help = 'Comma-separated list of `epoch:lr` pairs, defining a learning rate schedule. The total number of epochs can be appended to this list, separated by a comma as well.')
    arggroup = parser.add_argument_group('Parameters for --lr_schedule=SGDR')
    arggroup.add_argument('--sgdr_base_len', type = int, default = None, help = 'Length of first cycle in epochs.')
    arggroup.add_argument('--sgdr_mul', type = int, default = None, help = 'Multiplier for cycle length after each cycle.')
    arggroup.add_argument('--sgdr_max_lr', type = float, default = None, help = 'Maximum learning rate.')
    arggroup = parser.add_argument_group('Parameters for --lr_schedule=CLR')
    arggroup.add_argument('--clr_step_len', type = int, default = None, help = 'Length of each step in epochs.')
    arggroup.add_argument('--clr_min_lr', type = float, default = None, help = 'Minimum learning rate.')
    arggroup.add_argument('--clr_max_lr', type = float, default = None, help = 'Maximum learning rate.')



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

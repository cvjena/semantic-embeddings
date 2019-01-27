import numpy as np

import argparse
import pickle
import os
import shutil
from collections import OrderedDict

import keras
from keras import backend as K

import utils
from datasets import get_data_generator



def transform_inputs(X, y, num_classes, label_smoothing = 0):
    
    Y = keras.utils.to_categorical(y, num_classes)
    if (label_smoothing > 0) and (label_smoothing < 1):
        Y = Y * (1 - label_smoothing) + (1 - Y) * (label_smoothing / (num_classes - 1))
    return X, Y



if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Learns an image classifier.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    arggroup = parser.add_argument_group('Data parameters')
    arggroup.add_argument('--dataset', type = str, required = True, help = 'Training dataset. See README.md for a list of available datasets.')
    arggroup.add_argument('--data_root', type = str, required = True, help = 'Root directory of the dataset.')
    arggroup.add_argument('--class_list', type = str, default = None, help = 'Path to a file containing the IDs of the subset of classes to be used (as first words per line).')
    arggroup = parser.add_argument_group('Training parameters')
    arggroup.add_argument('--architecture', type = str, default = 'simple', choices = utils.ARCHITECTURES, help = 'Type of network architecture.')
    arggroup.add_argument('--label_smoothing', type = float, default = 0.0, help = 'Smooth the target distribution by subtracting this value from the target probability of the ground-truth class.')
    arggroup.add_argument('--lr_schedule', type = str, default = 'SGDR', choices = utils.LR_SCHEDULES, help = 'Type of learning rate schedule.')
    arggroup.add_argument('--clipgrad', type = float, default = 10.0, help = 'Gradient norm clipping.')
    arggroup.add_argument('--max_decay', type = float, default = 0.0, help = 'Learning Rate decay at the end of training.')
    arggroup.add_argument('--nesterov', action = 'store_true', default = False, help = 'Use Nesterov momentum instead of standard momentum.')
    arggroup.add_argument('--epochs', type = int, default = None, help = 'Number of training epochs.')
    arggroup.add_argument('--batch_size', type = int, default = 100, help = 'Batch size.')
    arggroup.add_argument('--val_batch_size', type = int, default = None, help = 'Validation batch size.')
    arggroup.add_argument('--snapshot', type = str, default = None, help = 'Path where snapshots should be stored after every epoch. If existing, it will be used to resume training.')
    arggroup.add_argument('--snapshot_best', type = str, nargs = '?', default = None, const = 'val_loss', help = 'Only store best-performing model as checkpoint, identified by monitoring the specified metric.')
    arggroup.add_argument('--initial_epoch', type = int, default = 0, help = 'Initial epoch for resuming training from snapshot.')
    arggroup.add_argument('--finetune', type = str, default = None, help = 'Path to pre-trained weights to be fine-tuned (will be loaded by layer name).')
    arggroup.add_argument('--finetune_init', type = int, default = 3, help = 'Number of initial epochs for training just the new layers before fine-tuning.')
    arggroup.add_argument('--gpus', type = int, default = 1, help = 'Number of GPUs to be used.')
    arggroup.add_argument('--read_workers', type = int, default = 8, help = 'Number of parallel data pre-processing processes.')
    arggroup.add_argument('--queue_size', type = int, default = 100, help = 'Maximum size of data queue.')
    arggroup.add_argument('--gpu_merge', action = 'store_true', default = False, help = 'Merge weights on the GPU.')
    arggroup = parser.add_argument_group('Output parameters')
    arggroup.add_argument('--model_dump', type = str, default = None, help = 'Filename where the learned model definition and weights should be written to.')
    arggroup.add_argument('--weight_dump', type = str, default = None, help = 'Filename where the learned model weights should be written to (without model definition).')
    arggroup.add_argument('--feature_dump', type = str, default = None, help = 'Filename where learned features for test images should be written to.')
    arggroup.add_argument('--log_dir', type = str, default = None, help = 'Tensorboard log directory.')
    arggroup.add_argument('--no_progress', action = 'store_true', default = False, help = 'Do not display training progress, but just the final performance.')
    utils.add_lr_schedule_arguments(parser)
    
    args = parser.parse_args()
    
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size

    # Configure environment
    K.set_session(K.tf.Session(config = K.tf.ConfigProto(gpu_options = { 'allow_growth' : True })))

    # Load dataset
    if args.class_list is not None:
        with open(args.class_list) as class_file:
            class_list = list(OrderedDict((l.strip().split()[0], None) for l in class_file if l.strip() != '').keys())
            try:
                class_list = [int(lbl) for lbl in class_list]
            except ValueError:
                pass
    else:
        class_list = None
    data_generator = get_data_generator(args.dataset, args.data_root, classes = class_list)

    # Construct and train model
    if (args.gpus <= 1) or args.gpu_merge:
        if args.snapshot and os.path.exists(args.snapshot):
            print('Resuming from snapshot {}'.format(args.snapshot))
            model = keras.models.load_model(args.snapshot, custom_objects = utils.get_custom_objects(args.architecture), compile = False)
        else:
            model = utils.build_network(data_generator.num_classes, args.architecture, True)
        par_model = model if args.gpus <= 1 else keras.utils.multi_gpu_model(model, gpus = args.gpus, cpu_merge = False)
    else:
        with K.tf.device('/cpu:0'):
            if args.snapshot and os.path.exists(args.snapshot):
                print('Resuming from snapshot {}'.format(args.snapshot))
                model = keras.models.load_model(args.snapshot, custom_objects = utils.get_custom_objects(args.architecture), compile = False)
            else:
                model = utils.build_network(data_generator.num_classes, args.architecture, True)
        par_model = keras.utils.multi_gpu_model(model, gpus = args.gpus)
    
    if not args.no_progress:
        model.summary()
    
    batch_transform_kwargs = { 'num_classes' : data_generator.num_classes, 'label_smoothing' : args.label_smoothing }
    
    # Load pre-trained weights and train last layer for a few epochs
    if args.finetune:
        print('Loading pre-trained weights from {}'.format(args.finetune))
        model.load_weights(args.finetune, by_name=True, skip_mismatch=True)
        if args.finetune_init > 0:
            print('Pre-training last layer')
            for layer in model.layers[:-1]:
                layer.trainable = False
            par_model.compile(optimizer = keras.optimizers.SGD(lr=args.sgd_lr, momentum=0.9, nesterov=args.nesterov, clipnorm = args.clipgrad),
                              loss = 'categorical_crossentropy', metrics = ['accuracy'])
            par_model.fit_generator(
                    data_generator.train_sequence(args.batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
                    validation_data = data_generator.test_sequence(args.val_batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
                    epochs = args.finetune_init, verbose = not args.no_progress,
                    max_queue_size = args.queue_size, workers = args.read_workers, use_multiprocessing = True)
            for layer in model.layers[:-1]:
                layer.trainable = True
            print('Full model training')

    # Train model
    callbacks, num_epochs = utils.get_lr_schedule(args.lr_schedule, data_generator.num_train, args.batch_size, schedule_args = { arg_name : arg_val for arg_name, arg_val in vars(args).items() if arg_val is not None })

    if args.log_dir:
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir, ignore_errors = True)
        callbacks.append(keras.callbacks.TensorBoard(log_dir = args.log_dir, write_graph = False))
    
    if args.snapshot:
        snapshot_kwargs = {}
        if args.snapshot_best:
            snapshot_kwargs['save_best_only'] = True
            snapshot_kwargs['monitor'] = args.snapshot_best
        callbacks.append(keras.callbacks.ModelCheckpoint(args.snapshot, **snapshot_kwargs) if args.gpus <= 1 else utils.TemplateModelCheckpoint(model, args.snapshot, **snapshot_kwargs))

    if args.max_decay > 0:
        decay = (1.0/args.max_decay - 1) / ((data_generator.num_train // args.batch_size) * (args.epochs if args.epochs else num_epochs))
    else:
        decay = 0.0
    par_model.compile(optimizer = keras.optimizers.SGD(lr=args.sgd_lr, decay=decay, momentum=0.9, nesterov=args.nesterov, clipnorm = args.clipgrad),
                      loss = 'categorical_crossentropy', metrics = ['accuracy'])


    par_model.fit_generator(
              data_generator.train_sequence(args.batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
              validation_data = data_generator.test_sequence(args.val_batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
              epochs = args.epochs if args.epochs else num_epochs, initial_epoch = args.initial_epoch,
              callbacks = callbacks, verbose = not args.no_progress,
              max_queue_size = args.queue_size, workers = args.read_workers, use_multiprocessing = True)

    # Evaluate final performance
    print(par_model.evaluate_generator(data_generator.test_sequence(args.val_batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs)))
    test_pred = par_model.predict_generator(data_generator.test_sequence(args.val_batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs)).argmax(axis=-1)
    class_freq = np.bincount(data_generator.labels_test)
    print('Average Accuracy: {:.4f}'.format(
        ((test_pred == np.asarray(data_generator.labels_test)).astype(np.float) / class_freq[np.asarray(data_generator.labels_test)]).sum() / len(class_freq)
    ))

    # Save model
    if args.weight_dump:
        try:
            model.save_weights(args.weight_dump)
        except Exception as e:
            print('An error occurred while saving the model weights: {}'.format(e))
    if args.model_dump:
        try:
            model.save(args.model_dump)
        except Exception as e:
            print('An error occurred while saving the model: {}'.format(e))

    # Save test image features
    if args.feature_dump:
        feat_model = keras.models.Model(model.inputs, model.layers[-2].output if not isinstance(model.layers[-2], keras.layers.BatchNormalization) else model.layers[-3].output)
        pred_features = feat_model.predict_generator(data_generator.flow_test(1, False), data_generator.num_test)
        with open(args.feature_dump,'wb') as dump_file:
            pickle.dump({ 'feat' : dict(enumerate(pred_features)) }, dump_file)

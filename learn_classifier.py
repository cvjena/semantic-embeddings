import numpy as np

import argparse
import pickle
import os
import shutil

import keras
from keras import backend as K

import utils
from datasets import DATASETS, get_data_generator



def gen_inputs(gen, num_classes):
    
    for X, y in gen:
        yield X, keras.utils.to_categorical(y, num_classes)



if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Learns an image classifier.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    arggroup = parser.add_argument_group('Data parameters')
    arggroup.add_argument('--dataset', type = str, required = True, choices = DATASETS, help = 'Training dataset.')
    arggroup.add_argument('--data_root', type = str, required = True, help = 'Root directory of the dataset.')
    arggroup = parser.add_argument_group('Training parameters')
    arggroup.add_argument('--architecture', type = str, default = 'simple', choices = utils.ARCHITECTURES, help = 'Type of network architecture.')
    arggroup.add_argument('--lr_schedule', type = str, default = 'SGDR', choices = utils.LR_SCHEDULES, help = 'Type of learning rate schedule.')
    arggroup.add_argument('--clipgrad', type = float, default = 10.0, help = 'Gradient norm clipping.')
    arggroup.add_argument('--max_decay', type = float, default = 0.0, help = 'Learning Rate decay at the end of training.')
    arggroup.add_argument('--epochs', type = int, default = None, help = 'Number of training epochs.')
    arggroup.add_argument('--batch_size', type = int, default = 100, help = 'Batch size.')
    arggroup.add_argument('--val_batch_size', type = int, default = None, help = 'Validation batch size.')
    arggroup.add_argument('--snapshot', type = str, default = None, help = 'Path where snapshots should be stored after every epoch. If existing, it will be used to resume training.')
    arggroup.add_argument('--initial_epoch', type = int, default = 0, help = 'Initial epoch for resuming training from snapshot.')
    arggroup.add_argument('--gpus', type = int, default = 1, help = 'Number of GPUs to be used.')
    arggroup = parser.add_argument_group('Output parameters')
    arggroup.add_argument('--model_dump', type = str, default = None, help = 'Filename where the learned model definition and weights should be written to.')
    arggroup.add_argument('--weight_dump', type = str, default = None, help = 'Filename where the learned model weights should be written to (without model definition).')
    arggroup.add_argument('--feature_dump', type = str, default = None, help = 'Filename where learned features for test images should be written to.')
    arggroup.add_argument('--log_dir', type = str, default = None, help = 'Tensorboard log directory.')
    arggroup.add_argument('--no_progress', action = 'store_true', default = False, help = 'Do not display training progress, but just the final performance.')
    arggroup = parser.add_argument_group('Parameters for --lr_schedule=SGD')
    arggroup.add_argument('--sgd_patience', type = int, default = None, help = 'Patience of learning rate reduction in epochs.')
    arggroup.add_argument('--sgd_lr', type = float, default = 0.1, help = 'Initial learning rate.')
    arggroup.add_argument('--sgd_min_lr', type = float, default = None, help = 'Minimum learning rate.')
    arggroup = parser.add_argument_group('Parameters for --lr_schedule=SGDR')
    arggroup.add_argument('--sgdr_base_len', type = int, default = None, help = 'Length of first cycle in epochs.')
    arggroup.add_argument('--sgdr_mul', type = int, default = None, help = 'Multiplier for cycle length after each cycle.')
    arggroup.add_argument('--sgdr_max_lr', type = float, default = None, help = 'Maximum learning rate.')
    arggroup = parser.add_argument_group('Parameters for --lr_schedule=CLR')
    arggroup.add_argument('--clr_step_len', type = int, default = None, help = 'Length of each step in epochs.')
    arggroup.add_argument('--clr_min_lr', type = float, default = None, help = 'Minimum learning rate.')
    arggroup.add_argument('--clr_max_lr', type = float, default = None, help = 'Maximum learning rate.')
    
    args = parser.parse_args()
    
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size

    # Configure environment
    K.set_session(K.tf.Session(config = K.tf.ConfigProto(gpu_options = { 'allow_growth' : True })))

    # Load dataset
    data_generator = get_data_generator(args.dataset, args.data_root)

    # Construct and train model
    if args.gpus <= 1:
        if args.snapshot and os.path.exists(args.snapshot):
            print('Resuming from snapshot {}'.format(args.snapshot))
            model = keras.models.load_model(args.snapshot, custom_objects = utils.get_custom_objects(args.architecture), compile = False)
        else:
            model = utils.build_network(data_generator.num_classes, args.architecture, True)
        par_model = model
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

    callbacks, num_epochs = utils.get_lr_schedule(args.lr_schedule, data_generator.num_train, args.batch_size, schedule_args = { arg_name : arg_val for arg_name, arg_val in vars(args).items() if arg_val is not None })

    if args.log_dir:
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir, ignore_errors = True)
        callbacks.append(keras.callbacks.TensorBoard(log_dir = args.log_dir, write_graph = False))
    
    if args.snapshot:
        callbacks.append(keras.callbacks.ModelCheckpoint(args.snapshot) if args.gpus <= 1 else utils.TemplateModelCheckpoint(model, args.snapshot))

    if args.max_decay > 0:
        decay = (1.0/args.max_decay - 1) / ((data_generator.num_train // args.batch_size) * (args.epochs if args.epochs else num_epochs))
    else:
        decay = 0.0
    par_model.compile(optimizer = keras.optimizers.SGD(lr=args.sgd_lr, decay=decay, momentum=0.9, clipnorm = args.clipgrad),
                      loss = 'categorical_crossentropy', metrics = ['accuracy'])

    par_model.fit_generator(
              gen_inputs(data_generator.flow_train(args.batch_size), data_generator.num_classes),
              data_generator.num_train // args.batch_size,
              validation_data = gen_inputs(data_generator.flow_test(args.val_batch_size), data_generator.num_classes),
              validation_steps = data_generator.num_test // args.val_batch_size,
              epochs = args.epochs if args.epochs else num_epochs, initial_epoch = args.initial_epoch,
              callbacks = callbacks, verbose = not args.no_progress)

    # Evaluate final performance
    print(par_model.evaluate_generator(gen_inputs(data_generator.flow_test(args.val_batch_size), data_generator.num_classes), data_generator.num_test // args.val_batch_size))

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

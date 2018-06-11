import numpy as np

import argparse
import pickle
import os
import shutil

import keras
from keras import backend as K

import utils
from datasets import DATASETS, get_data_generator



def transform_inputs(X, y, embedding):
    
    return X, embedding[y]



if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Learns to map image features onto word embeddings of labels using DeViSE.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    arggroup = parser.add_argument_group('Data parameters')
    arggroup.add_argument('--dataset', type = str, required = True, choices = DATASETS, help = 'Training dataset.')
    arggroup.add_argument('--data_root', type = str, required = True, help = 'Root directory of the dataset.')
    arggroup.add_argument('--embedding', type = str, required = True, help = 'Path to a pickle dump of embeddings in the same format as used by compute_class_embeddings.py.')
    arggroup = parser.add_argument_group('Training parameters')
    arggroup.add_argument('--architecture', type = str, default = 'simple', choices = utils.ARCHITECTURES, help = 'Type of network architecture.')
    arggroup.add_argument('--init_weights', type = str, default = None, help = 'Path to a weights file to initialize the model with.')
    arggroup.add_argument('--init_epochs', type = int, default = 25, help = 'Number of training epochs for the linear transformation layer only, keeping the rest of the network fixed.')
    arggroup.add_argument('--ft_epochs', type = int, default = 75, help = 'Number of training epochs for fine-tuning the full network.')
    arggroup.add_argument('--init_lr', type = float, default = 0.01, help = 'Learning rate for Adagrad during initial training of the linear transformation.')
    arggroup.add_argument('--ft_lr', type = float, default = 0.001, help = 'Learning rate for Adagrad during fine-tuning of the full network.')
    arggroup.add_argument('--batch_size', type = int, default = 100, help = 'Batch size.')
    arggroup.add_argument('--val_batch_size', type = int, default = None, help = 'Validation batch size.')
    arggroup.add_argument('--max_decay', type = float, default = 0.0, help = 'Learning Rate decay at the end of training.')
    arggroup.add_argument('--margin', type = float, default = 0.1, help = 'Margin of the hinge ranking loss.')
    arggroup = parser.add_argument_group('Output parameters')
    arggroup.add_argument('--model_dump', type = str, default = None, help = 'Filename where the learned model definition and weights should be written to.')
    arggroup.add_argument('--weight_dump', type = str, default = None, help = 'Filename where the learned model weights should be written to (without model definition).')
    arggroup.add_argument('--feature_dump', type = str, default = None, help = 'Filename where learned embeddings for test images should be written to.')
    arggroup.add_argument('--log_dir', type = str, default = None, help = 'Tensorboard log directory.')
    arggroup.add_argument('--no_progress', action = 'store_true', default = False, help = 'Do not display training progress, but just the final performance.')
    args = parser.parse_args()
    
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size

    # Configure environment
    K.set_session(K.tf.Session(config = K.tf.ConfigProto(gpu_options = { 'allow_growth' : True })))

    # Load and L2-normalize class embeddings
    with open(args.embedding, 'rb') as pf:
        embedding = pickle.load(pf)
        embed_labels = embedding['ind2label']
        embedding = embedding['embedding']
    embedding /= np.linalg.norm(embedding, axis = -1, keepdims = True)

    # Load dataset
    data_generator = get_data_generator(args.dataset, args.data_root, classes = embed_labels)

    # Construct and train model
    if args.init_weights:
        print('Initializing with model {}'.format(args.init_weights))
        model = keras.models.load_model(args.init_weights, custom_objects = utils.get_custom_objects(args.architecture), compile = False)
        new_output = keras.layers.Dense(embedding.shape[1], name = 'embedding')(model.layers[-1].input)
        model = keras.models.Model(model.inputs, new_output)
    else:
        model = utils.build_network(embedding.shape[1], args.architecture)
    
    if not args.no_progress:
        model.summary()

    callbacks = []
    batch_transform_kwargs = { 'embedding' : embedding }

    if args.init_weights and (args.init_epochs > 0):
        print('Pre-training linear transformation')
        for layer in model.layers[:-1]:
            layer.trainable = False
        
        model.compile(optimizer = keras.optimizers.Adagrad(lr=args.init_lr),
                      loss = utils.devise_ranking_loss(embedding, args.margin),
                      metrics = [utils.nn_accuracy(embedding, dot_prod_sim = True)])

        model.fit_generator(
              data_generator.train_sequence(args.batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
              validation_data = data_generator.test_sequence(args.val_batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
              epochs = args.init_epochs,
              callbacks = callbacks, verbose = not args.no_progress,
              max_queue_size = 100, workers = 8, use_multiprocessing = True)
        
        for layer in model.layers[:-1]:
            layer.trainable = True
    
    if args.log_dir:
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir, ignore_errors = True)
        callbacks.append(keras.callbacks.TensorBoard(log_dir = args.log_dir, write_graph = False))
    
    if args.ft_epochs > 0:
        print('Fine-tuning all layers')
        
        if args.max_decay > 0:
            decay = (1.0/args.max_decay - 1) / ((data_generator.num_train // args.batch_size) * args.ft_epochs)
        else:
            decay = 0.0
        
        model.compile(optimizer = keras.optimizers.Adagrad(lr=args.ft_lr, decay=decay),
                      loss = utils.devise_ranking_loss(embedding, args.margin),
                      metrics = [utils.nn_accuracy(embedding, dot_prod_sim = True)])

        model.fit_generator(
              data_generator.train_sequence(args.batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
              validation_data = data_generator.test_sequence(args.val_batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
              epochs = args.ft_epochs,
              callbacks = callbacks, verbose = not args.no_progress,
              max_queue_size = 100, workers = 8, use_multiprocessing = True)

    # Evaluate final performance
    print(model.evaluate_generator(data_generator.test_sequence(args.val_batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs)))

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

    # Save test image embeddings
    if args.feature_dump:
        pred_features = model.predict_generator(data_generator.flow_test(1, False), data_generator.num_test)
        with open(args.feature_dump,'wb') as dump_file:
            pickle.dump({ 'feat' : dict(enumerate(pred_features)) }, dump_file)

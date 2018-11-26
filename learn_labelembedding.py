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



def cross_entropy(logit, prob):
    return K.sum(prob * K.tf.nn.log_softmax(logit), axis = 1)


def labelembed_loss(out1, out2, tar, targets, tau = 2., alpha = 0.9, beta = 0.5, num_classes = 100):
    
    out2_prob = K.softmax(out2)
    tau2_prob = K.stop_gradient(K.softmax(out2 / tau))
    soft_tar = K.stop_gradient(K.softmax(tar))
    
    L_o1_y = K.sparse_categorical_crossentropy(output = K.softmax(out1), target = targets)
    
    pred = K.argmax(out2, axis = -1)
    mask = K.stop_gradient(K.cast(K.equal(pred, K.cast(targets, 'int64')), K.floatx()))
    L_o1_emb = -cross_entropy(out1, soft_tar)  # pylint: disable=invalid-unary-operand-type
    
    L_o2_y = K.sparse_categorical_crossentropy(output = out2_prob, target = targets)
    L_emb_o2 = -cross_entropy(tar, tau2_prob) * mask * (K.cast(K.shape(mask)[0], K.floatx())/(K.sum(mask)+1e-8))  # pylint: disable=invalid-unary-operand-type
    L_re = K.relu(K.sum(out2_prob * K.one_hot(K.cast(targets, 'int64'), num_classes), axis = -1) - alpha)
    
    return beta * L_o1_y + (1-beta) * L_o1_emb + L_o2_y + L_emb_o2 + L_re


def labelembed_model(base_model, num_classes, **kwargs):
    
    input_ = base_model.input
    embedding = base_model.output
    
    out = keras.layers.Activation('relu')(embedding)
    out = keras.layers.BatchNormalization(name = 'embedding_bn')(out)
    out1 = keras.layers.Dense(num_classes, name = 'prob')(out)
    out2 = keras.layers.Dense(num_classes, name = 'out2')(keras.layers.Lambda(lambda x: K.stop_gradient(x))(out))
    
    cls_input_ = keras.layers.Input((1,), name = 'labels')
    cls_embedding_layer = keras.layers.Embedding(num_classes, num_classes, embeddings_initializer = 'identity', name = 'labelembeddings')
    cls_embedding = keras.layers.Flatten()(cls_embedding_layer(cls_input_))
    
    loss = keras.layers.Lambda(lambda x: labelembed_loss(x[0], x[1], x[2], K.flatten(x[3]), num_classes = num_classes, **kwargs)[:,None], name = 'labelembed_loss')([out1, out2, cls_embedding, cls_input_])
    
    return keras.models.Model([input_, cls_input_], [embedding, out1, loss])


def transform_inputs(X, y, num_classes):
    
    return [X, y], { 'labelembed_loss' : np.zeros((len(X), 1)), 'prob' : keras.utils.to_categorical(y, num_classes) }



if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Trains a label embedding network (Sun et al.).', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    arggroup = parser.add_argument_group('Data parameters')
    arggroup.add_argument('--dataset', type = str, required = True, help = 'Training dataset. See README.md for a list of available datasets.')
    arggroup.add_argument('--data_root', type = str, required = True, help = 'Root directory of the dataset.')
    arggroup.add_argument('--class_list', type = str, default = None, help = 'Path to a file containing the IDs of the subset of classes to be used (as first words per line).')
    arggroup = parser.add_argument_group('Label embedding parameters')
    arggroup.add_argument('--embed_dim', type = int, default = 100, help = 'Embedding dimensionality.')
    arggroup.add_argument('--tau', type = float, default = 2., help = 'Softmax temperature.')
    arggroup.add_argument('--alpha', type = float, default = 0.9)
    arggroup.add_argument('--beta', type = float, default = 0.5)
    arggroup = parser.add_argument_group('Training parameters')
    arggroup.add_argument('--architecture', type = str, default = 'simple', choices = utils.ARCHITECTURES, help = 'Type of network architecture.')
    arggroup.add_argument('--lr_schedule', type = str, default = 'SGDR', choices = utils.LR_SCHEDULES, help = 'Type of learning rate schedule.')
    arggroup.add_argument('--clipgrad', type = float, default = 10.0, help = 'Gradient norm clipping.')
    arggroup.add_argument('--max_decay', type = float, default = 0.0, help = 'Learning Rate decay at the end of training.')
    arggroup.add_argument('--nesterov', action = 'store_true', default = False, help = 'Use Nesterov momentum instead of standard momentum.')
    arggroup.add_argument('--epochs', type = int, default = None, help = 'Number of training epochs.')
    arggroup.add_argument('--batch_size', type = int, default = 100, help = 'Batch size.')
    arggroup.add_argument('--val_batch_size', type = int, default = None, help = 'Validation batch size.')
    arggroup.add_argument('--finetune', type = str, default = None, help = 'Path to pre-trained weights to be fine-tuned (will be loaded by layer name).')
    arggroup.add_argument('--finetune_init', type = int, default = 3, help = 'Number of initial epochs for training just the new layers before fine-tuning.')
    arggroup.add_argument('--gpus', type = int, default = 1, help = 'Number of GPUs to be used.')
    arggroup.add_argument('--read_workers', type = int, default = 8, help = 'Number of parallel data pre-processing processes.')
    arggroup.add_argument('--queue_size', type = int, default = 100, help = 'Maximum size of data queue.')
    arggroup.add_argument('--gpu_merge', action = 'store_true', default = False, help = 'Merge weights on the GPU.')
    arggroup = parser.add_argument_group('Output parameters')
    arggroup.add_argument('--model_dump', type = str, default = None, help = 'Filename where the learned model definition and weights should be written to.')
    arggroup.add_argument('--weight_dump', type = str, default = None, help = 'Filename where the learned model weights should be written to (without model definition).')
    arggroup.add_argument('--feature_dump', type = str, default = None, help = 'Filename where learned embeddings for test images should be written to.')
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
        embed_model = utils.build_network(args.embed_dim, args.architecture)
        model = labelembed_model(embed_model, data_generator.num_classes, tau = args.tau, alpha = args.alpha, beta = args.beta)
        par_model = model if args.gpus <= 1 else keras.utils.multi_gpu_model(model, gpus = args.gpus, cpu_merge = False)
    else:
        with K.tf.device('/cpu:0'):
            embed_model = utils.build_network(args.embed_dim, args.architecture)
            model = labelembed_model(embed_model, data_generator.num_classes, tau = args.tau, alpha = args.alpha, beta = args.beta)
        par_model = keras.utils.multi_gpu_model(model, gpus = args.gpus)
    if not args.no_progress:
        model.summary()
    
    batch_transform_kwargs = { 'num_classes' : data_generator.num_classes }
    
    # Load pre-trained weights and train last layer for a few epochs
    if args.finetune:
        print('Loading pre-trained weights from {}'.format(args.finetune))
        model.load_weights(args.finetune, by_name=True, skip_mismatch=True)
        if args.finetune_init > 0:
            print('Pre-training new layers')
            for layer in model.layers:
                layer.trainable = (layer.name in ('embedding', 'embedding_bn', 'prob', 'out2', 'labelembeddings'))
            embed_model.layers[-1].trainable = True
            par_model.compile(optimizer = keras.optimizers.SGD(lr=args.sgd_lr, momentum=0.9, nesterov=args.nesterov, clipnorm = args.clipgrad),
                            loss = { 'labelembed_loss' : lambda y_true, y_pred: y_pred[:,0], 'embedding' : None, 'prob' : lambda y_true, y_pred: K.tf.zeros(K.shape(y_true)[:1], dtype=K.floatx()) },
                            metrics = { 'prob' : 'accuracy' })
            par_model.fit_generator(
                    data_generator.train_sequence(args.batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
                    validation_data = data_generator.test_sequence(args.val_batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
                    epochs = args.finetune_init, verbose = not args.no_progress,
                    max_queue_size = args.queue_size, workers = args.read_workers, use_multiprocessing = True)
            for layer in model.layers:
                layer.trainable = True
            print('Full model training')
    
    # Train model
    callbacks, num_epochs = utils.get_lr_schedule(args.lr_schedule, data_generator.num_train, args.batch_size, schedule_args = { arg_name : arg_val for arg_name, arg_val in vars(args).items() if arg_val is not None })

    if args.log_dir:
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir, ignore_errors = True)
        callbacks.append(keras.callbacks.TensorBoard(log_dir = args.log_dir, write_graph = False))

    if args.max_decay > 0:
        decay = (1.0/args.max_decay - 1) / ((data_generator.num_train // args.batch_size) * (args.epochs if args.epochs else num_epochs))
    else:
        decay = 0.0
    par_model.compile(optimizer = keras.optimizers.SGD(lr=args.sgd_lr, decay=decay, momentum=0.9, nesterov=args.nesterov, clipnorm = args.clipgrad),
                      loss = { 'labelembed_loss' : lambda y_true, y_pred: y_pred[:,0], 'embedding' : None, 'prob' : lambda y_true, y_pred: K.tf.zeros(K.shape(y_true)[:1], dtype=K.floatx()) },
                      metrics = { 'prob' : 'accuracy' })

    par_model.fit_generator(
              data_generator.train_sequence(args.batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
              validation_data = data_generator.test_sequence(args.val_batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs),
              epochs = args.epochs if args.epochs else num_epochs,
              callbacks = callbacks, verbose = not args.no_progress,
              max_queue_size = args.queue_size, workers = args.read_workers, use_multiprocessing = True)

    # Evaluate final performance
    print(par_model.evaluate_generator(data_generator.test_sequence(args.val_batch_size, batch_transform = transform_inputs, batch_transform_kwargs = batch_transform_kwargs)))
    try:
        test_pred = par_model.predict_generator(data_generator.flow_test(args.val_batch_size, False), data_generator.num_test // args.val_batch_size)[1].argmax(axis = -1)
        class_freq = np.bincount(data_generator.labels_test)
        print('Accuracy: {:.4f}'.format(np.mean(test_pred == np.asarray(data_generator.labels_test))))
        print('Average Accuracy: {:.4f}'.format(
            ((test_pred == np.asarray(data_generator.labels_test)).astype(np.float) / class_freq[np.asarray(data_generator.labels_test)]).sum() / len(class_freq)
        ))
    except:
        pass

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
        pred_features = embed_model.predict_generator(data_generator.flow_test(1, False), data_generator.num_test)
        with open(args.feature_dump,'wb') as dump_file:
            pickle.dump({ 'feat' : dict(enumerate(pred_features)) }, dump_file)

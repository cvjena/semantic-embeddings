import numpy as np
from sklearn.svm import LinearSVC
import keras

import sys, argparse, pickle, os.path
from collections import OrderedDict

import utils
from datasets import DATASETS, get_data_generator
from class_hierarchy import ClassHierarchy
from learn_labelembedding import labelembed_loss



METRICS = ['Accuracy', 'Hierarchical Accuracy']



def train_and_predict(data, model, layer = None, normalize = False, augmentation_epochs = 1, C = 1.0, custom_objects = {}):
    
    # Load model
    if isinstance(model, str):
        model = keras.models.load_model(model, custom_objects = custom_objects, compile = False)
    if layer is not None:
        model = keras.models.Model(model.inputs[0], model.layers[layer].output if isinstance(layer, int) else model.get_layer(layer).output)
    
    # Extract features
    sys.stderr.write('Extracting features...\n')
    X_train = model.predict_generator(data.flow_train(100, False, shuffle = False, augment = augmentation_epochs > 1), augmentation_epochs * (data.num_train // 100), verbose = 1)
    X_test = model.predict_generator(data.flow_test(100, False, shuffle = False, augment = False), data.num_test // 100, verbose = 1)
    if normalize:
        X_train /= np.linalg.norm(X_train, axis = -1, keepdims = True)
        X_test /= np.linalg.norm(X_test, axis = -1, keepdims = True)
    else:
        X_max = np.abs(X_train).max(axis = 0, keepdims = True)
        X_train /= np.maximum(1e-8, X_max)
        X_test /= np.maximum(1e-8, X_max)
    
    # Train SVM
    sys.stderr.write('Training SVM...\n')
    svm = LinearSVC(C = C, verbose = 1)
    svm.fit(X_train, np.tile(data.labels_train, augmentation_epochs))
    
    # Predict test classes
    sys.stderr.write('\nPredicting and evaluating...\n')
    return svm.predict(X_test)


def evaluate(y_pred, y_true, hierarchy, ind2label = None):
    
    perf = OrderedDict()
    perf['Accuracy'] = np.mean(y_pred == y_true)
    
    perf['Hierarchical Accuracy'] = 0.0
    for yp, yt in zip(y_pred, y_true):
        perf['Hierarchical Accuracy'] += 1.0 - (hierarchy.lcs_height(int(yp), int(yt)) if ind2label is None else hierarchy.lcs_height(ind2label[int(yp)], ind2label[int(yt)]))
    perf['Hierarchical Accuracy'] /= len(y_true)
    
    return perf


def print_performance(perf, metrics = METRICS):
    
    print()
    
    # Print header
    max_name_len = max(len(lbl) for lbl in perf.keys())
    print(' | '.join([' ' * max_name_len] + ['{:^6s}'.format(metric) for metric in METRICS]))
    print('-' * (max_name_len + sum(3 + max(6, len(metric)) for metric in METRICS)))

    # Print result rows
    for lbl, results in perf.items():
        print('{:{}s} | {}'.format(lbl, max_name_len, ' | '.join('{:>{}.4f}'.format(results[metric], max(len(metric), 6)) for metric in METRICS)))

    print()


def str2bool(v):
    
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Trains a linear SVM on different image embeddings and evaluates flat and hierarchical accuracy.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    arggroup = parser.add_argument_group('Dataset')
    arggroup.add_argument('--dataset', type = str, required = True, choices = DATASETS, help = 'Training dataset.')
    arggroup.add_argument('--data_root', type = str, required = True, help = 'Root directory of the dataset.')
    arggroup.add_argument('--hierarchy', type = str, required = True, help = 'Path to a file containing parent-child relationships (one per line).')
    arggroup.add_argument('--str_ids', action = 'store_true', default = False, help = 'If given, class IDs are treated as strings instead of integers.')
    arggroup.add_argument('--classes_from', type = str, default = None, help = 'Optionally, a path to a pickle dump containing a dictionary with item "ind2label" specifying the classes to be considered.')
    arggroup.add_argument('--augmentation_epochs', type = int, default = 1, help = 'Number of training image augmentations.')
    arggroup.add_argument('--C', type = float, default = 0.1, help = 'Weight of the error in SVM loss.')
    arggroup = parser.add_argument_group('Features')
    arggroup.add_argument('--architecture', type = str, default = 'simple', choices = utils.ARCHITECTURES, help = 'Type of network architecture.')
    arggroup.add_argument('--model', type = str, action = 'append', required = True, help = 'Path to a keras model dump used for extracting image features.')
    arggroup.add_argument('--layer', type = str, action = 'append', required = True, help = 'Name or index of the layer to extract features from.')
    arggroup.add_argument('--label', type = str, action = 'append', help = 'Label for the corresponding features.')
    arggroup.add_argument('--norm', type = str2bool, action = 'append', help = 'Whether to L2-normalize the corresponding features or not (defaults to False).')
    args = parser.parse_args()
    
    # Load dataset
    if args.classes_from:
        with open(args.classes_from, 'rb') as f:
            embed_labels = pickle.load(f)['ind2label']
    else:
        embed_labels = None
    data_generator = get_data_generator(args.dataset, args.data_root, classes = embed_labels)
    
    # Load class hierarchy
    id_type = str if args.str_ids else int
    hierarchy = ClassHierarchy.from_file(args.hierarchy, id_type = id_type)
    
    # Learn SVM classifier on training data and evaluate on test data
    custom_objects = utils.get_custom_objects(args.architecture)
    custom_objects['labelembed_loss'] = labelembed_loss
    perf = OrderedDict()
    for i, model in enumerate(args.model):
        model_name = args.label[i] if (args.label is not None) and (i < len(args.label)) else os.path.splitext(os.path.basename(model))[0]
        if (args.layer is not None) and (i < len(args.layer)):
            try:
                layer = int(args.layer[i])
            except ValueError:
                layer = args.layer[i]
        else:
            layer = None
        normalize = args.norm[i] if (args.norm is not None) and (i < len(args.norm)) else False
        sys.stderr.write('-- {} --\n'.format(model_name))
        perf[model_name] = evaluate(train_and_predict(data_generator, model, layer, normalize, args.augmentation_epochs, args.C, custom_objects), data_generator.labels_test, hierarchy, embed_labels)
    
    # Show results
    print_performance(perf)
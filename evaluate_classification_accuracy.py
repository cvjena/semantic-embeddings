import numpy as np
from sklearn.svm import LinearSVC
from scipy.spatial.distance import cdist
import keras

import sys, argparse, pickle, os.path
from collections import OrderedDict

import utils
from datasets import DATASETS, get_data_generator
from class_hierarchy import ClassHierarchy
from learn_labelembedding import labelembed_loss



METRICS = ['Accuracy', 'Top-5 Accuracy', 'Avg. Accuracy', 'Hierarchical Accuracy']



def train_and_predict(data, model, layer = None, normalize = False, augmentation_epochs = 1, C = 1.0, custom_objects = {}, batch_size = 1):
    """ Extracts image features, trains a linear SVM for classification, and returns predictions on the test data. """
    
    # Load model
    if isinstance(model, str):
        model = keras.models.load_model(model, custom_objects = custom_objects, compile = False)
    if layer is not None:
        model = keras.models.Model(model.inputs[0], model.layers[layer].output if isinstance(layer, int) else model.get_layer(layer).output)
    
    # Extract features
    sys.stderr.write('Extracting features...\n')
    X_train = model.predict_generator(data.flow_train(10, False, shuffle = False, augment = augmentation_epochs > 1), augmentation_epochs * (data.num_train // 10), verbose = 1)
    X_test = model.predict_generator(data.flow_test(batch_size, False, shuffle = False, augment = False), data.num_test // batch_size, verbose = 1)
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
    return svm.decision_function(X_test).argsort(axis = -1)[:,::-1]


def nn_classification(data, centroids, model, layer = None, custom_objects = {}, batch_size = 1):
    """ Extracts image embeddings and performs classification by assigning samples to the class of the nearest embedding. """
    
    # Load class centroids
    if isinstance(centroids, str):
        with open(centroids, 'rb') as f:
            centroids = pickle.load(f)['embedding']
    
    # Load model
    if isinstance(model, str):
        model = keras.models.load_model(model, custom_objects = custom_objects, compile = False)
    if layer is not None:
        model = keras.models.Model(model.inputs[0], model.layers[layer].output if isinstance(layer, int) else model.get_layer(layer).output)
    
    # Extract features
    sys.stderr.write('Extracting features...\n')
    feat = model.predict_generator(data.flow_test(batch_size, False, shuffle = False, augment = False), data.num_test // batch_size, verbose = 1)
    
    # Classify
    sys.stderr.write('Searching for nearest class centroids...\n')
    return cdist(feat, centroids, 'sqeuclidean').argsort(axis = -1)


def extract_predictions(data, model, layer = None, custom_objects = {}, batch_size = 1):
    """ Extracts class predictions. """
    
    # Load model
    if isinstance(model, str):
        model = keras.models.load_model(model, custom_objects = custom_objects, compile = False)
    if layer is not None:
        model = keras.models.Model(model.inputs[0], model.layers[layer].output if isinstance(layer, int) else model.get_layer(layer).output)
    
    # Extract predictions
    sys.stderr.write('Predicting and evaluating...\n')
    return model.predict_generator(data.flow_test(batch_size, False, shuffle = False, augment = False), data.num_test // batch_size, verbose = 1).argsort(axis = -1)[:,::-1]


def evaluate(y_pred, data_generator, hierarchy):
    
    perf = OrderedDict()
    y_true = np.asarray(data_generator.labels_test)
    if y_pred.ndim == 2:
        perf['Top-5 Accuracy'] = np.mean(np.any(y_pred[:,:5] == y_true[:,None], axis = -1))
        y_pred = y_pred[:,0]
    
    perf['Accuracy'] = np.mean(y_pred == y_true)

    class_freq = np.bincount(y_true)
    perf['Avg. Accuracy'] = ((y_pred == y_true).astype(np.float) / class_freq[y_true]).sum() / len(class_freq)
    
    perf['Hierarchical Accuracy'] = 0.0
    for yp, yt in zip(y_pred, y_true):
        perf['Hierarchical Accuracy'] += 1.0 - hierarchy.lcs_height(data_generator.classes[int(yp)], data_generator.classes[int(yt)])
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
        print('{:{}s} | {}'.format(lbl, max_name_len, ' | '.join('{:>{}.4f}'.format(results[metric], max(len(metric), 6)) if metric in results else '{:>{}s}'.format('--', max(len(metric), 6)) for metric in METRICS)))

    print()


def str2bool(v):
    
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Evaluates flat, balanced, and hierarchical accuracy of several models.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    arggroup = parser.add_argument_group('Dataset')
    arggroup.add_argument('--dataset', type = str, required = True, choices = DATASETS, help = 'Training dataset.')
    arggroup.add_argument('--data_root', type = str, required = True, help = 'Root directory of the dataset.')
    arggroup.add_argument('--hierarchy', type = str, required = True, help = 'Path to a file containing parent-child relationships (one per line).')
    arggroup.add_argument('--is_a', action = 'store_true', default = False, help = 'If given, --hierarchy is assumed to contain is-a instead of parent-child relationships.')
    arggroup.add_argument('--str_ids', action = 'store_true', default = False, help = 'If given, class IDs are treated as strings instead of integers.')
    arggroup.add_argument('--classes_from', type = str, default = None, help = 'Optionally, a path to a pickle dump containing a dictionary with item "ind2label" specifying the classes to be considered. These should be in the same order as the classes predicted by the model.')
    arggroup.add_argument('--augmentation_epochs', type = int, default = 1, help = 'Number of training image augmentations when training an SVM on top of embeddings.')
    arggroup.add_argument('--C', type = float, default = 0.1, help = 'Weight of the error in SVM loss.')
    arggroup.add_argument('--batch_size', type = int, default = 1, help = 'Batch size for feature extraction. Must divide the number of test images evenly.')
    arggroup = parser.add_argument_group('Features')
    arggroup.add_argument('--architecture', type = str, default = 'simple', choices = utils.ARCHITECTURES, help = 'Type of network architecture.')
    arggroup.add_argument('--model', type = str, action = 'append', required = True, help = 'Path to a keras model dump used for extracting image features.')
    arggroup.add_argument('--layer', type = str, action = 'append', required = True, help = 'Name or index of the layer to extract features from.')
    arggroup.add_argument('--label', type = str, action = 'append', help = 'Label for the corresponding features.')
    arggroup.add_argument('--norm', type = str2bool, action = 'append', help = 'Whether to L2-normalize the corresponding features or not (defaults to False).')
    arggroup.add_argument('--prob_features', type = str2bool, action = 'append', help = 'Whether to use the extracted features as class probabilities instead of training an SVM.')
    arggroup.add_argument('--centroids', type = str, action = 'append', help = 'Optionally, a pickle dump containing a dictionary with an item "embedding" referring to a numpy array of class centroids for performing nearest-neighbor classification.')
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
    hierarchy = ClassHierarchy.from_file(args.hierarchy, is_a_relations = args.is_a, id_type = id_type)
    
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
        prob_features = args.prob_features[i] if (args.prob_features is not None) and (i < len(args.prob_features)) else False
        centroids = args.centroids[i] if (args.centroids is not None) and (i < len(args.centroids)) else ''
        sys.stderr.write('-- {} --\n'.format(model_name))
        if prob_features:
            pred = extract_predictions(data_generator, model, layer, custom_objects, args.batch_size)
        elif centroids:
            pred = nn_classification(data_generator, centroids, model, layer, custom_objects, args.batch_size)
        else:
            pred = train_and_predict(data_generator, model, layer, normalize, args.augmentation_epochs, args.C, custom_objects, args.batch_size)
        perf[model_name] = evaluate(pred, data_generator, hierarchy)
    
    # Show results
    print_performance(perf)
import numpy as np
import numexpr as ne

import argparse, pickle, os.path
from collections import OrderedDict

from datasets import DATASETS, get_data_generator
from class_hierarchy import ClassHierarchy

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it



METRICS = ['P@1 (WUP)', 'P@10 (WUP)', 'P@50 (WUP)', 'P@100 (WUP)', 'AHP (WUP)', 'P@1 (LCS_HEIGHT)', 'P@10 (LCS_HEIGHT)', 'P@50 (LCS_HEIGHT)', 'P@100 (LCS_HEIGHT)', 'AHP (LCS_HEIGHT)']



def pairwise_retrieval(features, normalize = False):
    
    # Convert feature list to numpy array
    if isinstance(features, str):
        with open(features, 'rb') as feat_dump:
            features = pickle.load(feat_dump)
    if isinstance(features, dict):
        if 'feat' in features:
            features = features['feat']
        ind2id = dict(enumerate(features.keys()))
        features = np.stack(list(features.values()))
        if features.ndim > 2:
            features = features.mean(axis = tuple(range(2, features.ndim)))  # avg pooling
    else:
        ind2id = None
    
    # Compute pairwise distances
    if normalize:
        features /= np.linalg.norm(features, axis = -1, keepdims = True)
        pdist = -np.dot(features, features.T)
    else:
        sqnorm = np.sum(features ** 2, axis = -1)
        pdist = ne.evaluate('A + B - 2 * C', { 'A' : sqnorm[:,None], 'B' : sqnorm[None,:], 'C' : np.dot(features, features.T) })
    
    # Rank images
    ranking = np.argsort(pdist, axis = -1)
    return { (ind2id[i] if ind2id is not None else i) : [ind2id[int(r)] if ind2id is not None else int(r) for r in ret] for i, ret in enumerate(ranking) }


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


def write_performance(perf, csv_file, prec_type = 'LCS_HEIGHT'):
    
    with open(csv_file, 'w') as f:
        f.write('k;' + ';'.join(perf.keys()) + '\n')
        k = 1
        while True:
            try:
                f.write('{};{}\n'.format(k, ';'.join(str(res['P@{} ({})'.format(k, prec_type)]) for res in perf.values())))
                k += 1
            except KeyError:
                break


def plot_performance(perf, kmax = 100, prec_type = 'LCS_HEIGHT'):
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('Hierarchical Precision')
    plt.xlim(0, kmax)
    plt.ylim(0, 1)
    plt.grid()
    
    min_prec = 1.0
    for lbl, metrics in perf.items():
        precs = [metrics['P@{} ({})'.format(k, prec_type)] for k in range(1, kmax+1)]
        plt.plot(np.arange(kmax), precs, label = lbl)
        min_prec = min(min_prec, min(precs))
    
    min_prec = np.floor(min_prec * 20) / 20
    if min_prec >= 0.3:
        plt.ylim(min_prec, 1)
    
    plt.legend(fontsize = 'x-small')
    
    
    plt.figure()
    plt.xlabel('Mean Average Hierarchical Precision')
    plt.yticks([])
    plt.grid(axis = 'x')
    
    for i, (lbl, metrics) in enumerate(perf.items()):
        mAHP = metrics['AHP ({})'.format(prec_type)]
        plt.barh(i + 0.5, mAHP, 0.8)
        plt.text(0.01, i + 0.5, lbl, verticalalignment = 'center', horizontalalignment = 'left', color = 'white', fontsize = 'small')
        plt.text(mAHP - 0.01, i + 0.5, '{:.1%}'.format(mAHP), verticalalignment = 'center', horizontalalignment = 'right', color = 'white')
    
    
    plt.show()


def str2bool(v):
    
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Evaluates hierarchical precision of nearest neighbour search performed on different image embeddings.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    arggroup = parser.add_argument_group('Dataset')
    arggroup.add_argument('--dataset', type = str, required = True, choices = DATASETS, help = 'Training dataset.')
    arggroup.add_argument('--data_root', type = str, required = True, help = 'Root directory of the dataset.')
    arggroup.add_argument('--hierarchy', type = str, required = True, help = 'Path to a file containing parent-child relationships (one per line).')
    arggroup.add_argument('--str_ids', action = 'store_true', default = False, help = 'If given, class IDs are treated as strings instead of integers.')
    arggroup.add_argument('--classes_from', type = str, default = None, help = 'Optionally, a path to a pickle dump containing a dictionary with item "ind2label" specifying the classes to be considered.')
    arggroup = parser.add_argument_group('Features')
    arggroup.add_argument('--feat', type = str, action = 'append', required = True, help = 'Pickle file containing a dictionary mapping image IDs to features.')
    arggroup.add_argument('--label', type = str, action = 'append', help = 'Label for the corresponding features.')
    arggroup.add_argument('--norm', type = str2bool, action = 'append', help = 'Whether to L2-normalize the corresponding features or not (defaults to False).')
    arggroup = parser.add_argument_group('Output')
    arggroup.add_argument('--plot_max', type = int, default = 250, help = 'Plot hierarchical precision up to this number of retrieved images. Set this to 0 to disable plotting.')
    arggroup.add_argument('--prec_type', type = str, default = 'LCS_HEIGHT', choices = ['WUP', 'LCS_HEIGHT'], help = 'Measure for semantic similarity between classes to be used.')
    arggroup.add_argument('--csv', type = str, default = None, help = 'Name of a CSV file where performance metrics will be written to.')
    args = parser.parse_args()
    
    # Load dataset
    if args.classes_from:
        with open(args.classes_from, 'rb') as f:
            embed_labels = pickle.load(f)['ind2label']
    else:
        embed_labels = None
    data_generator = get_data_generator(args.dataset, args.data_root, classes = embed_labels)
    labels_test = [embed_labels[lbl] for lbl in data_generator.labels_test] if embed_labels is not None else data_generator.labels_test
    
    # Load class hierarchy
    id_type = str if args.str_ids else int
    hierarchy = ClassHierarchy.from_file(args.hierarchy, id_type = id_type)
    
    # Perform image retrieval using all images in the dataset as queries
    ks = list(range(1, args.plot_max + 1))
    for k in [1, 10, 50, 100]:
        if (len(ks) == 0) or (ks[-1] < k):
            ks.append(k)
    perf = OrderedDict()
    for i, feat_dump in tqdm(enumerate(args.feat), total = len(args.feat)):
        feat_name = args.label[i] if (args.label is not None) and (i < len(args.label)) else os.path.splitext(os.path.basename(feat_dump))[0]
        normalize = args.norm[i] if (args.norm is not None) and (i < len(args.norm)) else False
        perf[feat_name] = hierarchy.hierarchical_precision(pairwise_retrieval(feat_dump, normalize), data_generator.labels_test, ks, compute_ahp = True, all_ids = list(range(data_generator.num_test)))[0]
    
    # Show results
    print_performance(perf)
    if args.csv:
        write_performance(perf, args.csv, args.prec_type)
    if args.plot_max > 0:
        plot_performance(perf, args.plot_max, args.prec_type)
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

import argparse, pickle, os.path
from collections import OrderedDict

from datasets import DATASETS, get_data_generator
from evaluate_retrieval import pairwise_retrieval, str2bool

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Plots the average recall-precision curve of nearest neighbour search performed on different image embeddings.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    arggroup = parser.add_argument_group('Dataset')
    arggroup.add_argument('--dataset', type = str, required = True, choices = DATASETS, help = 'Training dataset.')
    arggroup.add_argument('--data_root', type = str, required = True, help = 'Root directory of the dataset.')
    arggroup.add_argument('--classes_from', type = str, default = None, help = 'Optionally, a path to a pickle dump containing a dictionary with item "ind2label" specifying the classes to be considered.')
    arggroup = parser.add_argument_group('Features')
    arggroup.add_argument('--feat', type = str, action = 'append', required = True, help = 'Pickle file containing a dictionary mapping image IDs to features.')
    arggroup.add_argument('--label', type = str, action = 'append', help = 'Label for the corresponding features.')
    arggroup.add_argument('--norm', type = str2bool, action = 'append', help = 'Whether to L2-normalize the corresponding features or not (defaults to False).')
    arggroup = parser.add_argument_group('Plot')
    arggroup.add_argument('--bins', type = int, default = None, help = 'Optional, number of recall levels to be distinguished.')
    args = parser.parse_args()
    
    # Load dataset
    if args.classes_from:
        with open(args.classes_from, 'rb') as f:
            embed_labels = pickle.load(f)['ind2label']
    else:
        embed_labels = None
    data_generator = get_data_generator(args.dataset, args.data_root, classes = embed_labels)
    labels_test = [embed_labels[lbl] for lbl in data_generator.labels_test] if embed_labels is not None else data_generator.labels_test
    
    # Create figure
    plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()

    # Draw recall-precision curves for all features
    for i, feat_dump in tqdm(enumerate(args.feat), total = len(args.feat)):
        
        feat_name = args.label[i] if (args.label is not None) and (i < len(args.label)) else os.path.splitext(os.path.basename(feat_dump))[0]
        normalize = args.norm[i] if (args.norm is not None) and (i < len(args.norm)) else False
        recprec = {}
        aps = []

        for qid, retrieved in tqdm(pairwise_retrieval(feat_dump, normalize, True), total = data_generator.num_test):
            
            rp = {}
            
            correct = np.asarray([labels_test[r] == labels_test[qid] for r in retrieved if r != qid])
            aps.append(average_precision_score(correct, -np.arange(len(correct))))
            
            tp = correct.astype(np.float).cumsum()
            recall = tp / tp[-1]
            precision = tp / np.arange(1, len(tp) + 1)
            for r, p in zip(recall, precision):
                if args.bins:
                    r = int(r * args.bins) / args.bins + 1/(2*args.bins)
                rp[r] = max(rp[r], p) if r in rp else p
            for r, p in rp.items():
                if r in recprec:
                    recprec[r].append(p)
                else:
                    recprec[r] = [p]
        
        levels = sorted(recprec.keys())
        plt.plot(levels, [np.mean(recprec[r]) for r in levels], label = '{} (mAP: {:.2%})'.format(feat_name, np.mean(aps)))
    
    # Show figure
    plt.legend(fontsize = 'x-small')
    plt.show()
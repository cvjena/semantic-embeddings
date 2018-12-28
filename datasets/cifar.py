import numpy as np
import pickle
import os

from .common import TinyDatasetGenerator



class CifarGenerator(TinyDatasetGenerator):
    """ Data generator for CIFAR-10 and CIFAR-100. """

    def __init__(self, root_dir, classes = None, reenumerate = False, cifar10 = False, **kwargs):
        """ Data generator for CIFAR-10 and CIFAR-100.

        # Arguments:

        - root_dir: Root directory of the dataset.

        - classes: List of classes to restrict the dataset to.
                   If set to `None`, all available classes will be used.
        
        - reenumerate: If true, the classes given in `classes` will be re-enumerated in ascending order, beginning from 0.
        
        - cifar10: Set this to True for CIFAR-10 and to False for CIFAR-100.

        Further keyword arguments such as `generator_kwargs` and `train_generator_kwargs` will be
        forwarded to the constructor of `TinyDatasetGenerator`.
        """
        
        self.root_dir = root_dir

        # Load dataset
        if cifar10:
            X_train, y_train = [], []
            for i in range(1, 6):
                with open(os.path.join(self.root_dir, 'data_batch_{}'.format(i)), 'rb') as pf:
                    dump = pickle.load(pf, encoding='bytes')
                    X_train.append(dump[b'data'].astype(np.float32))
                    y_train += dump[b'labels']
                    del dump
            X_train = np.concatenate(X_train)
        else:
            with open(os.path.join(self.root_dir, 'train'), 'rb') as pf:
                dump = pickle.load(pf, encoding='bytes')
                X_train, y_train = dump[b'data'].astype(np.float32), dump[b'fine_labels']
                del dump

        with open(os.path.join(self.root_dir, 'test_batch' if cifar10 else 'test'), 'rb') as pf:
            dump = pickle.load(pf, encoding='bytes')
            X_test, y_test = dump[b'data'].astype(np.float32), dump[b'labels' if cifar10 else b'fine_labels']
            del dump
        
        # Restrict labels to the given classes and re-enumerate them
        if classes is not None:
            
            sel_train = np.array([lbl in classes for lbl in y_train])
            sel_test = np.array([lbl in classes for lbl in y_test])
            X_train = X_train[sel_train]
            y_train = [lbl for lbl, sel in zip(y_train, sel_train) if sel]
            X_test = X_test[sel_test]
            y_test = [lbl for lbl, sel in zip(y_test, sel_test) if sel]
            
            self.classes = classes
            if reenumerate:
                self.class_indices = dict(zip(self.classes, range(len(self.classes))))
                y_train = [self.class_indices[lbl] for lbl in y_train]
                y_test = [self.class_indices[lbl] for lbl in y_test]
        
        else:

            self.classes = np.arange(max(y_train) + 1)
            self.class_indices = dict(zip(self.classes, self.classes))

        # Reshape data to images
        X_train = X_train.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        X_test = X_test.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

        # Call parent constructor
        super(CifarGenerator, self).__init__(X_train, X_test, y_train, y_test, **kwargs)

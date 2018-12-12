import numpy as np
import pickle
import os

try:
    from keras.preprocessing.image import ImageDataGenerator
except ImportError:
    import keras
    from keras_preprocessing.image import ImageDataGenerator

from keras import backend as K

from .common import DataSequence



class CifarGenerator(object):
    """ Data generator for CIFAR-10 and CIFAR-100. """

    def __init__(self, root_dir, classes = None, reenumerate = False, cifar10 = False,
                 randzoom_range = 0., rotation_range = 0.):
        """ Data generator for CIFAR-10 and CIFAR-100.

        # Arguments:

        - root_dir: Root directory of the dataset.

        - classes: List of classes to restrict the dataset to.
                   If set to `None`, all available classes will be used.
        
        - reenumerate: If true, the classes given in `classes` will be re-enumerated in ascending order, beginning from 0.
        
        - cifar10: Set this to True for CIFAR-10 and to False for CIFAR-100.

        - randzoom_range: Float or [lower, upper]. Range for random zoom.
                          If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        
        - rotation_range: Int. Degree range for random rotations.
        """
        
        super(CifarGenerator, self).__init__()
        self.root_dir = root_dir

        # Load dataset
        if cifar10:
            self.X_train, self.y_train = [], []
            for i in range(1, 6):
                with open(os.path.join(self.root_dir, 'data_batch_{}'.format(i)), 'rb') as pf:
                    dump = pickle.load(pf, encoding='bytes')
                    self.X_train.append(dump[b'data'].astype(np.float32))
                    self.y_train += dump[b'labels']
                    del dump
            self.X_train = np.concatenate(self.X_train)
        else:
            with open(os.path.join(self.root_dir, 'train'), 'rb') as pf:
                dump = pickle.load(pf, encoding='bytes')
                self.X_train, self.y_train = dump[b'data'].astype(np.float32), dump[b'fine_labels']
                del dump

        with open(os.path.join(self.root_dir, 'test_batch' if cifar10 else 'test'), 'rb') as pf:
            dump = pickle.load(pf, encoding='bytes')
            self.X_test, self.y_test = dump[b'data'].astype(np.float32), dump[b'labels' if cifar10 else b'fine_labels']
            del dump
        
        # Restrict labels to the given classes and re-enumerate them
        if classes is not None:
            
            sel_train = np.array([lbl in classes for lbl in self.y_train])
            sel_test = np.array([lbl in classes for lbl in self.y_test])
            self.X_train = self.X_train[sel_train]
            self.y_train = [lbl for lbl, sel in zip(self.y_train, sel_train) if sel]
            self.X_test = self.X_test[sel_test]
            self.y_test = [lbl for lbl, sel in zip(self.y_test, sel_test) if sel]
            
            self.classes = classes
            if reenumerate:
                self.class_indices = dict(zip(self.classes, range(len(self.classes))))
                self.y_train = [self.class_indices[lbl] for lbl in self.y_train]
                self.y_test = [self.class_indices[lbl] for lbl in self.y_test]
        
        else:

            self.classes = np.arange(max(self.y_train) + 1)
            self.class_indices = dict(zip(self.classes, self.classes))

        # Reshape data to images
        self.X_train = self.X_train.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        self.X_test = self.X_test.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

        # Set up pre-processing
        self.image_generator = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True, horizontal_flip = True,
                                                  width_shift_range = 0.15, height_shift_range = 0.15, zoom_range = randzoom_range, rotation_range = rotation_range)
        self.image_generator.fit(self.X_train)

        self.test_image_generator = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True)
        self.test_image_generator.fit(self.X_train)
    
    
    def flow_train(self, batch_size = 32, include_labels = True, shuffle = True, augment = True):
        """ A generator yielding batches of pre-processed and augmented training images.

        # Arguments:

        - batch_size: Number of images per batch.

        - include_labels: If true, target labels will be yielded as well.

        - shuffle: If True, the order of images will be shuffled after each epoch.
        
        - augment: Whether data augmentation should be applied or not.

        # Yields:
            If `include_labels` is True, a tuple of inputs and targets for each batch.
            Otherwise, only inputs will be yielded.
        """
        
        image_generator = self.image_generator if augment else self.test_image_generator
        return image_generator.flow(self.X_train, self.y_train if include_labels else None,
                                    batch_size=batch_size, shuffle=shuffle)
    
    
    def flow_test(self, batch_size = 32, include_labels = True, shuffle = False, augment = False):
        """ A generator yielding batches of pre-processed and augmented test images.

        # Arguments:

        - batch_size: Number of images per batch.

        - include_labels: If true, target labels will be yielded as well.

        - shuffle: If True, the order of images will be shuffled after each epoch.
        
        - augment: Whether data augmentation should be applied or not.

        # Yields:
            If `include_labels` is True, a tuple of inputs and targets for each batch.
            Otherwise, only inputs will be yielded.
        """
        
        image_generator = self.image_generator if augment else self.test_image_generator
        return image_generator.flow(self.X_test, self.y_test if include_labels else None,
                                    batch_size=batch_size, shuffle=shuffle)


    def train_sequence(self, batch_size = 32, shuffle = True, augment = True, batch_transform = None, batch_transform_kwargs = {}):
        """ Creates a `DataSequence` with pre-processed and augmented training images that can be passed to the Keras methods expecting a generator for efficient and safe multi-processing.

        # Arguments:

        - batch_size: Number of images per batch.

        - shuffle: If True, the order of images will be shuffled after each epoch.
        
        - augment: Whether data augmentation should be applied or not.

        - batch_transform: Optionally, a function that takes the inputs and targets of a batch and returns
                           transformed inputs and targets that will be provided by the sequence instead of
                           the original ones.
        
        - batch_transform_kwargs: Additional keyword arguments passed to `batch_transform`.

        # Returns:
            a DataSequence instance
        """
        
        return DataSequence(self, np.arange(len(self.X_train)), self.y_train,
                            train=True, augment=augment,
                            batch_size=batch_size, shuffle=shuffle, batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)
    
    
    def test_sequence(self, batch_size = 32, shuffle = False, augment = False, batch_transform = None, batch_transform_kwargs = {}):
        """ Creates a `DataSequence` with pre-processed and augmented test images that can be passed to the Keras methods expecting a generator for efficient and safe multi-processing.

        # Arguments:

        - batch_size: Number of images per batch.

        - shuffle: If True, the order of images will be shuffled after each epoch.
        
        - augment: Whether data augmentation should be applied or not.

        - batch_transform: Optionally, a function that takes the inputs and targets of a batch and returns
                           transformed inputs and targets that will be provided by the sequence instead of
                           the original ones.
        
        - batch_transform_kwargs: Additional keyword arguments passed to `batch_transform`.

        # Returns:
            a DataSequence instance
        """
        
        return DataSequence(self, np.arange(len(self.X_test)), self.y_test,
                            train=False, augment=augment,
                            batch_size=batch_size, shuffle=shuffle, batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)
    

    def compose_batch(self, indices, train, augment = False):
        """ Composes a batch of augmented images given by their indices.

        # Arguments:

        - indices: List with image indices to be contained in the batch.

        - train: If True, images will be taken from the training data matrix, otherwise from the test data.

        - augment: Whether data augmentation should be applied or not.

        # Returns:
            a batch of images as 4-dimensional numpy array.
        """

        X = self.X_train if train else self.X_test
        image_generator = self.image_generator if augment else self.test_image_generator

        batch = np.zeros((len(indices),) + tuple(X.shape[1:]), dtype=K.floatx())
        for i, j in enumerate(indices):
            x = X[j]
            x = image_generator.random_transform(x.astype(K.floatx()))
            x = image_generator.standardize(x)
            batch[i] = x
        
        return batch
    

    @property
    def labels_train(self):
        """ List with labels corresponding to the training files in `self.X_train`.
        
        This is an alias for `self.y_train` for compatibility with other data generators.
        """
        
        return self.y_train
    
    
    @property
    def labels_test(self):
        """ List with labels corresponding to the test files in `self.X_test`.
        
        This is an alias for `self.y_test` for compatibility with other data generators.
        """
        
        return self.y_test
    
    
    @property
    def num_classes(self):
        """ Number of unique classes in the dataset. """
        
        return max(self.y_train) + 1
    
    
    @property
    def num_train(self):
        """ Number of training images in the dataset. """
        
        return len(self.X_train)
    
    
    @property
    def num_test(self):
        """ Number of test images in the dataset. """
        
        return len(self.X_test)

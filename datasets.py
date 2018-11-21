import numpy as np
import pickle
import PIL.Image
import os
import warnings
from collections import Counter

try:
    from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, list_pictures
except ImportError:
    import keras
    from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array, list_pictures
from keras.utils import Sequence
from keras import backend as K

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        for x in it:
            yield x



DATASETS = ['CIFAR-10', 'CIFAR-100', 'CIFAR-100-a', 'CIFAR-100-b', 'CIFAR-100-a-consec', 'CIFAR-100-b-consec', 'ILSVRC', 'ILSVRC-caffe', 'NAB']



def get_data_generator(dataset, data_root, classes = None):
    """ Shortcut for creating a data generator with default settings.

    # Arguments:

    - dataset: The name of the dataset. Possible values can be found in the `DATASETS` constant.

    - data_root: Root directory of the dataset.

    - classes: Optionally, a list of classes to be included. If not given, all available classes will be used.

    # Returns:
        a data generator object
    """
    
    dataset = dataset.lower()
    if dataset == 'cifar-10':
        return CifarGenerator(data_root, classes, reenumerate = True, cifar10 = True, randzoom_range = 0.25)
    elif dataset == 'cifar-100':
        return CifarGenerator(data_root, classes, reenumerate = True)
    elif dataset.startswith('cifar-100-a'):
        return CifarGenerator(data_root, np.arange(50), reenumerate = dataset.endswith('-consec'))
    elif dataset.startswith('cifar-100-b'):
        return CifarGenerator(data_root, np.arange(50, 100), reenumerate = dataset.endswith('-consec'))
    elif dataset == 'ilsvrc':
        return ILSVRCGenerator(data_root, classes)
    elif dataset == 'ilsvrc-caffe':
        return ILSVRCGenerator(data_root, classes, mean = [123.68, 116.779, 103.939], std = [1., 1., 1.], color_mode = 'bgr')
    elif dataset == 'nab':
        return NABGenerator(data_root, classes, 'images', randzoom_range = (256, 480))
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))



class DataSequence(Sequence):
    """ Helper class representing a sequence that can be passed to Keras functions expecting a generator. """

    def __init__(self, data_generator, ids, labels, batch_size = 32, shuffle = False, oversample = False, batch_transform = None, batch_transform_kwargs = {}, **kwargs):
        """
        # Arguments:

        - data_generator: The data generator instance that created this sequence.
                          Must provide a `compose_batch` method that takes a list image indices as first argument
                          and optionally any additional keyword arguments passed to this constructor and returns
                          a batch of images as numpy array.
        
        - ids: List with IDs of all images.

        - labels: List with labels corresponding to the images in `ids`.

        - batch_size: The size of the batches provided by this sequence.

        - shuffle: Whether to shuffle the order of images after each epoch.

        - oversample: Whether to oversample smaller classes to the size of the largest one.

        - batch_transform: Optionally, a function that takes the inputs and targets of a batch and returns
                           transformed inputs and targets that will be provided by this sequence instead of
                           the original ones.
        
        - batch_transform_kwargs: Additional keyword arguments passed to `batch_transform`.
        """

        super(DataSequence, self).__init__()
        self.data_generator = data_generator
        self.ids = ids
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.oversample = oversample
        self.batch_transform = batch_transform
        self.batch_transform_kwargs = batch_transform_kwargs
        self.kwargs = kwargs

        if self.oversample:
            self.class_sizes = Counter(labels)
            self.max_class_size = max(self.class_sizes.values())
            self.class_members = { lbl : np.where(np.asarray(labels) == lbl)[0] for lbl in self.class_sizes.keys() }
            self.ind = np.concatenate([
                np.repeat(members, int(np.ceil(self.max_class_size / len(members))))[:self.max_class_size]
                for lbl, members in self.class_members.items()
            ])
        else:
            self.ind = np.arange(len(self.ids))
        
        self.on_epoch_end()


    def __len__(self):
        """ Returns the number of batches per epoch. """

        if self.oversample:
            return int(np.ceil((len(self.class_sizes) * self.max_class_size) / self.batch_size))
        else:
            return int(np.ceil(len(self.ids) / self.batch_size))


    def __getitem__(self, idx):
        """ Returns the batch with the given index. """
        
        batch_ind = self.ind[idx*self.batch_size:(idx+1)*self.batch_size]
        X = self.data_generator.compose_batch([self.ids[i] for i in batch_ind], **self.kwargs)
        y = self.labels[batch_ind]
        if self.batch_transform is not None:
            return self.batch_transform(X, y, **self.batch_transform_kwargs)  # pylint: disable=not-callable
        else:
            return X, y


    def on_epoch_end(self):
        """ Called by Keras after each epoch. Handles shuffling of the data if required. """

        if self.shuffle:
            
            if self.oversample:
                self.ind = np.concatenate([
                    np.concatenate([
                        np.random.choice(members, len(members), replace = False)
                        for _ in range(int(np.ceil(self.max_class_size / len(members))))
                    ])[:self.max_class_size]
                    for lbl, members in self.class_members.items()
                ])
            
            np.random.shuffle(self.ind)



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



class FileDatasetGenerator(object):
    """ Abstract base class for image generators. """

    def __init__(self, root_dir, classes = None, cropsize = (224, 224), default_target_size = -1,
                 randzoom_range = None, randerase_prob = 0.0, randerase_params = { 'sl' : 0.02, 'sh' : 0.4, 'r1' : 0.3, 'r2' : 1./0.3 },
                 color_mode = 'rgb'):
        """ Abstract base class for image generators.

        # Arguments:

        - root_dir: Root directory of the dataset.

        - classes: List of classes to restrict the dataset to. Numeric labels will be assigned to these classes in ascending order.
                   If set to `None`, all available classes will be used.
        
        - cropsize: Tuple with width and height of crops extracted from the images.

        - randzoom_range: Tuple with minimum and maximum size of the smaller image dimension for random scale augmentation.
                          May either be given as integer specifying absolute pixel values or float specifying the relative scale of the image.
                          If set to `None`, no scale augmentation will be performed.
        
        - randerase_prob: Probability for random erasing.

        - randerase_params: Random erasing parameters (see Zhong et al. (2017): "Random erasing data augmentation.").

        - color_mode: Image color mode, either "rgb" or "bgr".
        """
        
        super(FileDatasetGenerator, self).__init__()
        
        self.root_dir = root_dir
        self.cropsize = cropsize
        self.default_target_size = default_target_size
        self.randzoom_range = randzoom_range
        self.randerase_prob = randerase_prob
        self.randerase_params = randerase_params
        self.color_mode = color_mode.lower()
        
        self.classes = []
        self.train_img_files = []
        self.test_img_files = []
        self._train_labels = []
        self._test_labels = []
        
        warnings.filterwarnings('ignore', '.*[Cc]orrupt EXIF data.*', UserWarning)
    
    
    def _compute_stats(self, mean = None, std = None):
        """ Computes channel-wise mean and standard deviation of all images in the dataset.
        
        If `mean` and `std` arguments are given, they will just be stored instead of being re-computed.

        The channel order of both is always "RGB", independent of `color_mode`.
        """
        
        if mean is None:
            mean = 0
            for fn in tqdm(self.train_img_files, desc = 'Computing channel mean'):
                mean += np.mean(np.asarray(load_img(fn), dtype=np.float64), axis = (0,1))
            mean /= len(self.train_img_files)
            print('Channel-wise mean:               {}'.format(mean))
        self.mean = np.asarray(mean, dtype=np.float32)
        if (mean is None) or (std is None):
            std = 0
            for fn in tqdm(self.train_img_files, desc = 'Computing channel variance'):
                std += np.mean((np.asarray(load_img(fn), dtype=np.float64) - self.mean) ** 2, axis = (0,1))
            std = np.sqrt(std / (len(self.train_img_files) - 1))
            print('Channel-wise standard deviation: {}'.format(std))
        self.std = np.asarray(std, dtype=np.float32)
    
    
    def flow_train(self, batch_size = 32, include_labels = True, shuffle = True, target_size = None, augment = True):
        """ A generator yielding batches of pre-processed and augmented training images.

        # Arguments:

        - batch_size: Number of images per batch.

        - include_labels: If true, target labels will be yielded as well.

        - shuffle: If True, the order of images will be shuffled after each epoch.

        - target_size: Int or tuple of ints. Specifies the target size which the image will be resized to (before cropping).
                       If a single int is given, it specifies the size of the smaller side of the image and the aspect ratio will be retained.
                       If set to -1, the image won't be resized.
                       If set to None, the default_target_size passed to the constructor will be used.
        
        - augment: Whether data augmentation should be applied or not.

        # Yields:
            If `include_labels` is True, a tuple of inputs and targets for each batch.
            Otherwise, only inputs will be yielded.
        """
        
        return self._flow(self.train_img_files, self._train_labels if include_labels else None,
                          batch_size=batch_size, shuffle=shuffle, target_size=target_size,
                          normalize=True, hflip=augment, vflip=False, randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment)
    
    
    def flow_test(self, batch_size = 32, include_labels = True, shuffle = False, target_size = None, augment = False):
        """ A generator yielding batches of pre-processed and augmented test images.

        # Arguments:

        - batch_size: Number of images per batch.

        - include_labels: If true, target labels will be yielded as well.

        - shuffle: If True, the order of images will be shuffled after each epoch.

        - target_size: Int or tuple of ints. Specifies the target size which the image will be resized to (before cropping).
                       If a single int is given, it specifies the size of the smaller side of the image and the aspect ratio will be retained.
                       If set to -1, the image won't be resized.
                       If set to None, the default_target_size passed to the constructor will be used.
        
        - augment: Whether data augmentation should be applied or not.

        # Yields:
            If `include_labels` is True, a tuple of inputs and targets for each batch.
            Otherwise, only inputs will be yielded.
        """
        
        return self._flow(self.test_img_files, self._test_labels if include_labels else None,
                          batch_size=batch_size, shuffle=shuffle, target_size=target_size,
                          normalize=True, hflip=augment, vflip=False, randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment)
    

    def train_sequence(self, batch_size = 32, shuffle = True, target_size = None, augment = True, batch_transform = None, batch_transform_kwargs = {}):
        """ Creates a `DataSequence` with pre-processed and augmented training images that can be passed to the Keras methods expecting a generator for efficient and safe multi-processing.

        # Arguments:

        - batch_size: Number of images per batch.

        - shuffle: If True, the order of images will be shuffled after each epoch.

        - target_size: Int or tuple of ints. Specifies the target size which the image will be resized to (before cropping).
                       If a single int is given, it specifies the size of the smaller side of the image and the aspect ratio will be retained.
                       If set to -1, the image won't be resized.
                       If set to None, the default_target_size passed to the constructor will be used.
        
        - augment: Whether data augmentation should be applied or not.

        - batch_transform: Optionally, a function that takes the inputs and targets of a batch and returns
                           transformed inputs and targets that will be provided by the sequence instead of
                           the original ones.
        
        - batch_transform_kwargs: Additional keyword arguments passed to `batch_transform`.

        # Returns:
            a DataSequence instance
        """
        
        return DataSequence(self, self.train_img_files, self._train_labels,
                            batch_size=batch_size, shuffle=shuffle,
                            target_size=target_size, normalize=True, hflip=augment, vflip=False,
                            randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment,
                            batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)
    
    
    def test_sequence(self, batch_size = 32, shuffle = False, target_size = None, augment = False, batch_transform = None, batch_transform_kwargs = {}):
        """ Creates a `DataSequence` with pre-processed and augmented test images that can be passed to the Keras methods expecting a generator for efficient and safe multi-processing.

        # Arguments:

        - batch_size: Number of images per batch.

        - shuffle: If True, the order of images will be shuffled after each epoch.

        - target_size: Int or tuple of ints. Specifies the target size which the image will be resized to (before cropping).
                       If a single int is given, it specifies the size of the smaller side of the image and the aspect ratio will be retained.
                       If set to -1, the image won't be resized.
                       If set to None, the default_target_size passed to the constructor will be used.
        
        - augment: Whether data augmentation should be applied or not.

        - batch_transform: Optionally, a function that takes the inputs and targets of a batch and returns
                           transformed inputs and targets that will be provided by the sequence instead of
                           the original ones.
        
        - batch_transform_kwargs: Additional keyword arguments passed to `batch_transform`.

        # Returns:
            a DataSequence instance
        """

        return DataSequence(self, self.test_img_files, self._test_labels,
                            batch_size=batch_size, shuffle=shuffle,
                            target_size=target_size, normalize=True, hflip=augment, vflip=False,
                            randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment,
                            batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)
    
    
    def _flow(self, filenames, labels = None, batch_size = 32, shuffle = False, **kwargs):
        """ A generator yielding batches of pre-processed and augmented images.

        # Arguments:

        - filenames: List with the filenames of all images to sample batches from.

        - labels: Optionally, labels corresponding to the images specified by `filenames`.

        - batch_size: Number of images per batch.

        - shuffle: If True, the order of images will be shuffled after each epoch.

        Remaining keyword arguments will be passed through to `compose_batch`.

        # Yields:
            If `labels` is not None, a tuple of inputs and targets for each batch.
            Otherwise, only inputs will be yielded.
        """
        
        ind = np.arange(len(filenames))
        if shuffle:
            np.random.shuffle(ind)
        
        if labels is not None:
            labels = np.asarray(labels)
        
        offs = 0
        while True:
            
            if offs >= len(ind):
                offs = 0
                if shuffle:
                    np.random.shuffle(ind)
            
            batch_ind = ind[offs:offs+batch_size]
            offs += batch_size
            
            X = self.compose_batch([filenames[i] for i in batch_ind], **kwargs)

            if labels is not None:
                yield X, labels[batch_ind]
            else:
                yield X


    def compose_batch(self, filenames, cropsize = None, randcrop = False, data_format = None, **kwargs):
        """ Composes a batch of augmented images given by their filenames.

        # Arguments:

        - filenames: List with image filenames to be contained in the batch.

        - cropsize: Int or tuple of ints specifying the size which the images will be cropped to.
                    If a single int is given, a square crop will be extracted.
                    If set to None, the batch will be cropped to the median size of the images in the batch.
        
        - randcrop: If True, a random crop will be extracted from each image, otherwise the center crop.

        - data_format: The image data format (either 'channels_first' or 'channels_last'). Set to None for the default value.

        Remaining keyword arguments will be passed through to `_load_and_transform`.

        # Returns:
            a batch of images as 4-dimensional numpy array.
        """

        if data_format is None:
            data_format = K.image_data_format()
        if data_format == 'channels_first':
            x_axis, y_axis = 2, 1
        else:
            x_axis, y_axis = 1, 0

        X = [self._load_and_transform(fn, data_format=data_format, **kwargs) for fn in filenames]
        if cropsize is not None:
            crop_width, crop_height = cropsize
        else:
            crop_height = int(np.median([img.shape[y_axis] for img in X]))
            crop_width = int(np.median([img.shape[x_axis] for img in X]))
        for i, img in enumerate(X):
            y_pad = x_pad = 0
            if img.shape[y_axis] > crop_height:
                y_offs = np.random.randint(img.shape[y_axis] - crop_height + 1) if randcrop else (img.shape[y_axis] - crop_height) // 2
                img = img[:,y_offs:y_offs+crop_height,:] if data_format == 'channels_first' else img[y_offs:y_offs+crop_height,:,:]
            elif img.shape[y_axis] < crop_height:
                y_pad = np.random.randint(crop_height - img.shape[y_axis] + 1) if randcrop else (crop_height - img.shape[y_axis]) // 2
            if img.shape[x_axis] > crop_width:
                x_offs = np.random.randint(img.shape[x_axis] - crop_width + 1) if randcrop else (img.shape[x_axis] - crop_width) // 2
                img = img[:,:,x_offs:x_offs+crop_width] if data_format == 'channels_first' else img[:,x_offs:x_offs+crop_width,:]
            elif img.shape[x_axis] < crop_width:
                x_pad = np.random.randint(crop_width - img.shape[x_axis] + 1) if randcrop else (crop_width - img.shape[x_axis]) // 2
            X[i] = np.pad(
                img,
                ((0,0), (y_pad, crop_height - img.shape[1] - y_pad), (x_pad, crop_width - img.shape[2] - x_pad)) if data_format == 'channels_first' else \
                ((y_pad, crop_height - img.shape[0] - y_pad), (x_pad, crop_width - img.shape[1] - x_pad), (0,0)),
                'reflect'
            )
        return np.stack(X)


    def _load_and_transform(self, filename, target_size = None, normalize = True, hflip = False, vflip = False, randzoom = False, randerase = False, data_format = None):
        """ Loads an image file and applies normalization and data augmentation.
        
        # Arguments:

        - filename: The path of the image file.

        - target_size: Int or tuple of ints. Specifies the target size which the image will be resized to.
                       If a single int is given, it specifies the size of the smaller side of the image and the aspect ratio will be retained.
                       If set to -1, the image won't be resized.
                       If set to None, the default_target_size passed to the constructor will be used.
                       The actual size may be modified further is `randzoom` is True.
        
        - normalize: If True, the image will be normalized by subtracting the channel-wise mean and dividing by the channel-wise standard deviation.

        - hflip: If True, the image will be flipped horizontally with a chance of 50%.

        - vflip: If True, the image will be flipped vertically with a chance of 50%.

        - randzoom: If True and `self.randzoom_range` is not None, random zooming will be applied.
                    If `self.randzoom_range` is given as floats defining a range relative to the image size,
                    `target_size` will be used as reference if it is not None, otherwise the original image size.
        
        - randerase: If True, random erasing will be applied with probability `self.randerase_prob`.

        - data_format: The image data format (either 'channels_first' or 'channels_last'). Set to None for the default value.

        # Returns:
            the image as 3-dimensional numpy array.
        """
        
        if data_format is None:
            data_format = K.image_data_format()
        
        # Load and resize image
        img = load_img(filename)
        if target_size is None:
            target_size = self.default_target_size
        if (target_size > 0) or (randzoom and (self.randzoom_range is not None)):
            if target_size <= 0:
                target_size = img.size
            if randzoom and (self.randzoom_range is not None):
                if isinstance(self.randzoom_range[0], float):
                    target_size = np.round(np.array(target_size) * np.random.uniform(self.randzoom_range[0], self.randzoom_range[1])).astype(int).tolist()
                else:
                    target_size = np.random.randint(self.randzoom_range[0], self.randzoom_range[1])
            if isinstance(target_size, int):
                target_size = (target_size, round(img.size[1] * (target_size / img.size[0]))) if img.size[0] < img.size[1] else (round(img.size[0] * (target_size / img.size[1])), target_size)
            img = img.resize(target_size, PIL.Image.BILINEAR)
        img = img_to_array(img, data_format=data_format)
        
        # Normalize image
        if normalize:
            img -= self.mean[:,None,None] if data_format == 'channels_first' else self.mean[None,None,:]
            img /= self.std[:,None,None] if data_format == 'channels_first' else self.std[None,None,:]
        
        # RGB -> BGR conversion
        if self.color_mode == 'bgr':
            img = img[::-1,:,:] if data_format == 'channels_first' else img[:,:,::-1]
        
        # Random Flipping
        if hflip and (np.random.random() < 0.5):
            img = img[:,:,::-1] if data_format == 'channels_first' else img[:,::-1,:]
        
        if vflip and (np.random.random() < 0.5):
            img = img[:,::-1,:] if data_format == 'channels_first' else img[::-1,:,:]
        
        # Random erasing
        if randerase and (self.randerase_prob > 0) and (np.random.random() < self.randerase_prob):
            while True:
                se = np.random.uniform(self.randerase_params['sl'], self.randerase_params['sh']) * (img.shape[0] * img.shape[1])
                re = np.random.uniform(self.randerase_params['r1'], self.randerase_params['r2'])
                he, we = int(np.sqrt(se * re)), int(np.sqrt(se / re))
                if (he < img.shape[0]) and (we < img.shape[1]):
                    break
            xe, ye = np.random.randint(0, img.shape[1] - we), np.random.randint(0, img.shape[0] - he)
            img[ye:ye+he,xe:xe+we,:] = (np.random.uniform(0., 255., (he, we, img.shape[-1])) \
                                       - (self.mean[:,None,None] if data_format == 'channels_first' else self.mean[None,None,:])) \
                                       / (self.std[:,None,None] if data_format == 'channels_first' else self.std[None,None,:])
        
        return img
    
    
    @property
    def labels_train(self):
        """ List with labels corresponding to the training files in `self.train_img_files`.
        
        These are not the original labels, but automatically assigned numeric labels in the range `[0, num_classes-1]`.
        The look-up table in `self.class_indices` can be used to obtain the original label for each class.
        """
        
        return self._train_labels
    
    
    @property
    def labels_test(self):
        """ List with labels corresponding to the test files in `self.test_img_files`.
        
        These are not the original labels, but automatically assigned numeric labels in the range `[0, num_classes-1]`.
        The look-up table in `self.class_indices` can be used to obtain the original label for each class.
        """
        
        return self._test_labels
    
    
    @property
    def num_classes(self):
        """ Number of unique classes in the dataset. """
        
        return len(self.classes)
    
    
    @property
    def num_train(self):
        """ Number of training images in the dataset. """
        
        return len(self.train_img_files)
    
    
    @property
    def num_test(self):
        """ Number of test images in the dataset. """
        
        return len(self.test_img_files)



class ILSVRCGenerator(FileDatasetGenerator):

    def __init__(self, root_dir, classes = None, mean = [122.65435242, 116.6545058, 103.99789959], std = [71.40583196, 69.56888997, 73.0440314], color_mode = "rgb"):
        """ ILSVRC data generator.

        # Arguments:

        - root_dir: Root directory of the ILSVRC dataset, containing directories "ILSVRC2012_img_train" and "ILSVRC2012_img_val", both containing
                    sub-directories with names of synsets and the images for each synset in the corresponding sub-directories.

        - classes: List of synsets to restrict the dataset to. Numeric labels will be assigned to these synsets in ascending order.
                   If set to `None`, all available synsets will be used and enumerated in the lexicographical order.
        
        - mean: Channel-wise image mean for normalization (in "RGB" order). If set to `None`, mean and standard deviation will be computed from the images.

        - std: Channel-wise standard deviation for normalization (in "RGB" order). If set to `None`, standard deviation will be computed from the images.

        - color_mode: Image color mode, either "rgb" or "bgr".
        """
        
        super(ILSVRCGenerator, self).__init__(root_dir, classes, default_target_size = 256, randzoom_range = (256, 480), color_mode = color_mode)
        self.train_dir = os.path.join(self.root_dir, 'ILSVRC2012_img_train')
        self.test_dir = os.path.join(self.root_dir, 'ILSVRC2012_img_val')
        
        # Search for classes
        if classes is None:
            classes = []
            for subdir in sorted(os.listdir(self.train_dir)):
                if os.path.isdir(os.path.join(self.train_dir, subdir)):
                    classes.append(subdir)
        self.classes = classes
        self.class_indices = dict(zip(self.classes, range(len(self.classes))))
        
        # Search for images
        for lbl, subdir in enumerate(self.classes):
            cls_files = sorted(list_pictures(os.path.join(self.train_dir, subdir), 'JPE?G|jpe?g'))
            self.train_img_files += cls_files
            self._train_labels += [lbl] * len(cls_files)
            cls_files = sorted(list_pictures(os.path.join(self.test_dir, subdir), 'JPE?G|jpe?g'))
            self.test_img_files += cls_files
            self._test_labels += [lbl] * len(cls_files)
        print('Found {} training and {} validation images from {} classes.'.format(self.num_train, self.num_test, self.num_classes))
        
        # Compute mean and standard deviation
        self._compute_stats(mean, std)



class NABGenerator(FileDatasetGenerator):

    def __init__(self, root_dir, classes = None, img_dir = 'images',
                 cropsize = (224, 224), default_target_size = 256, randzoom_range = None, randerase_prob = 0.5, randerase_params = { 'sl' : 0.02, 'sh' : 0.3, 'r1' : 0.3, 'r2' : 1./0.3 },
                 mean = [125.30513277, 129.66606421, 118.45121113], std = [57.0045467, 56.70059436, 68.44430446], color_mode = "rgb"):
        """ NABirds data generator.

        # Arguments:

        - root_dir: Root directory of the NAB dataset, containing the files `images.txt`, `image_class_labels.txt`, and `train_test_split.txt`.

        - classes: List of classes to restrict the dataset to. Numeric labels will be assigned to these classes in ascending order.
                   If set to `None`, all available classes will be used and enumerated in ascending order.
        
        - img_dir: Name of the sub-directory of `root_dir` that contains the images in further sub-directories named by their class label.

        - cropsize: Tuple with width and height of crops extracted from the images.

        - randzoom_range: Tuple with minimum and maximum size of the smaller image dimension for random scale augmentation.
                          May either be given as integer specifying absolute pixel values or float specifying the relative scale of the image.
                          If set to `None`, no scale augmentation will be performed.
        
        - randerase_prob: Probability for random erasing.

        - randerase_params: Random erasing parameters (see Zhong et al. (2017): "Random erasing data augmentation.").
        
        - mean: Channel-wise image mean for normalization (in "RGB" order). If set to `None`, mean and standard deviation will be computed from the images.

        - std: Channel-wise standard deviation for normalization (in "RGB" order). If set to `None`, standard deviation will be computed from the images.

        - color_mode: Image color mode, either "rgb" or "bgr".
        """
        
        super(NABGenerator, self).__init__(root_dir, classes = classes, cropsize = cropsize, default_target_size = default_target_size, randzoom_range = randzoom_range,
                                           randerase_prob = randerase_prob, randerase_params = randerase_params, color_mode = color_mode)
        self.imgs_dir = os.path.join(root_dir, img_dir)
        self.img_list_file = os.path.join(root_dir, 'images.txt')
        self.label_file = os.path.join(root_dir, 'image_class_labels.txt')
        self.split_file = os.path.join(root_dir, 'train_test_split.txt')
        
        # Read train/test split information
        with open(self.split_file) as f:
            is_train = { img_id : (flag != '0') for l in f if l.strip() != '' for img_id, flag in [l.strip().split()] }
        
        # Read labels
        with open(self.label_file) as f:
            img_labels = { img_id : int(lbl) for l in f if l.strip() != '' for img_id, lbl in [l.strip().split()] }
        self.classes = classes if classes is not None else sorted(set(img_labels.values()))
        self.class_indices = dict(zip(self.classes, range(len(self.classes))))
        
        # Search for images
        with open(self.img_list_file) as f:
            for l in f:
                if l.strip() != '':
                    img_id, fn = l.strip().split()
                    if img_labels[img_id] in self.class_indices:
                        if is_train[img_id]:
                            self.train_img_files.append(os.path.join(self.imgs_dir, fn))
                            self._train_labels.append(self.class_indices[img_labels[img_id]])
                        else:
                            self.test_img_files.append(os.path.join(self.imgs_dir, fn))
                            self._test_labels.append(self.class_indices[img_labels[img_id]])
        print('Found {} training and {} validation images from {} classes.'.format(self.num_train, self.num_test, self.num_classes))
        
        # Compute mean and standard deviation
        self._compute_stats(mean, std)

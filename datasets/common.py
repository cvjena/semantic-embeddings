import numpy as np
import PIL.Image
import warnings
from collections import Counter

try:
    from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
except ImportError:
    import keras
    from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras import backend as K
from keras.utils import Sequence

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        for x in it:
            yield x



class DataSequence(Sequence):
    """ Helper class representing a sequence that can be passed to Keras functions expecting a generator. """

    def __init__(self, data_generator, ids, labels, batch_size = 32, shuffle = False, oversample = False, repeats = 1,
                 batch_transform = None, batch_transform_kwargs = {}, **kwargs):
        """
        # Arguments:

        - data_generator: The data generator instance that created this sequence.
                          Must provide a `compose_batch` method that takes a list of image indices as first argument
                          and optionally any additional keyword arguments passed to this constructor and returns
                          a batch of images as numpy array.
        
        - ids: List with IDs of all images.

        - labels: List with labels corresponding to the images in `ids`.

        - batch_size: The size of the batches provided by this sequence.

        - shuffle: Whether to shuffle the order of images after each epoch.

        - oversample: Whether to oversample smaller classes to the size of the largest one.

        - repeats: Number of repeats per epoch. If this was set to 3, for example, a single epoch would actually
                   comprise 3 epochs.

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
        self.repeats = repeats
        self.batch_transform = batch_transform
        self.batch_transform_kwargs = batch_transform_kwargs
        self.kwargs = kwargs

        if self.oversample:
            self.class_sizes = Counter(labels)
            self.max_class_size = max(self.class_sizes.values())
            self.class_members = { lbl : np.where(np.asarray(labels) == lbl)[0] for lbl in self.class_sizes.keys() }
            self.permutations = [np.concatenate([
                np.repeat(members, int(np.ceil(self.max_class_size / len(members))))[:self.max_class_size]
                for lbl, members in self.class_members.items()
            ]) for i in range(self.repeats)]
            self.epoch_len = int(np.ceil((len(self.class_sizes) * self.max_class_size) / self.batch_size))
        else:
            self.permutations = [np.arange(len(self.ids)) for i in range(self.repeats)]
            self.epoch_len = int(np.ceil(len(self.ids) / self.batch_size))
        
        self.on_epoch_end()


    def __len__(self):
        """ Returns the number of batches per epoch. """

        return self.repeats * self.epoch_len


    def __getitem__(self, idx):
        """ Returns the batch with the given index. """
        
        subepoch = idx // self.epoch_len
        idx = idx % self.epoch_len
        batch_ind = self.permutations[subepoch][idx*self.batch_size:(idx+1)*self.batch_size]
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
                self.permutations = [np.concatenate([
                    np.concatenate([
                        np.random.choice(members, len(members), replace = False)
                        for _ in range(int(np.ceil(self.max_class_size / len(members))))
                    ])[:self.max_class_size]
                    for lbl, members in self.class_members.items()
                ]) for i in range(self.repeats)]
            
            for i in range(self.repeats):
                np.random.shuffle(self.permutations[i])



class FileDatasetGenerator(object):
    """ Abstract base class for image generators. """

    def __init__(self, root_dir, cropsize = (224, 224), default_target_size = -1,
                 randzoom_range = None, randrot_max = 0,
                 distort_colors = False, colordistort_params = {},
                 randerase_prob = 0.0, randerase_params = { 'sl' : 0.02, 'sh' : 0.4, 'r1' : 0.3, 'r2' : 1./0.3 },
                 color_mode = 'rgb'):
        """ Abstract base class for image generators.

        # Arguments:

        - root_dir: Root directory of the dataset.
        
        - cropsize: Tuple with width and height of crops extracted from the images.

        - default_target_size: Int or tuple of ints. Specifies the default target size which images will be resized to (before cropping)
                               if not specified differently in calls to `flow_train/test` or `train/test_sequence`.
                               If a single int is given, it specifies the size of the smaller side of the image and the aspect ratio will be retained.
                               If set to -1, the image won't be resized.

        - randzoom_range: Tuple with minimum and maximum size of the smaller image dimension for random scale augmentation.
                          May either be given as integer specifying absolute pixel values or float specifying the relative scale of the image.
                          If set to `None`, no scale augmentation will be performed.
        
        - randrot_max: Maximum number of degrees for random rotations.

        - distort_colors: Boolean specifying whether to apply color distortions as data augmentation.

        - colordistort_params: Parameters for color distortions, passed as keyword arguments to `distort_colors()`.
        
        - randerase_prob: Probability for random erasing.

        - randerase_params: Random erasing parameters (see Zhong et al. (2017): "Random erasing data augmentation.").

        - color_mode: Image color mode, either "rgb" or "bgr".
        """
        
        super(FileDatasetGenerator, self).__init__()
        
        self.root_dir = root_dir
        self.cropsize = cropsize
        self.default_target_size = default_target_size
        self.randzoom_range = randzoom_range
        self.randrot_max = randrot_max
        self.distort_colors = distort_colors
        self.colordistort_params = colordistort_params
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
                          normalize=True, hflip=augment, vflip=False, colordistort=self.distort_colors and augment,
                          randzoom=augment, randrot=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment)
    
    
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
                          normalize=True, hflip=augment, vflip=False, colordistort=False,
                          randzoom=augment, randrot=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment)
    

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
                            target_size=target_size, normalize=True, hflip=augment, vflip=False, colordistort=self.distort_colors and augment,
                            randzoom=augment, randrot=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment,
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
                            target_size=target_size, normalize=True, hflip=augment, vflip=False, colordistort=False,
                            randzoom=augment, randrot=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment,
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


    def _load_image(self, filename, target_size = None, randzoom = False):
        """ Loads an image file.

        # Arguments:

        - filename: The path of the image file.

        - target_size: Int or tuple of ints. Specifies the target size which the image will be resized to.
                       If a single int is given, it specifies the size of the smaller side of the image and the aspect ratio will be retained.
                       If set to -1, the image won't be resized.
                       If set to None, the default_target_size passed to the constructor will be used.
                       The actual size may be modified further is `randzoom` is True.
        
        - randzoom: If True and `self.randzoom_range` is not None, random zooming will be applied.
                    If `self.randzoom_range` is given as floats defining a range relative to the image size,
                    `target_size` will be used as reference if it is not None, otherwise the original image size.
        
        # Returns:
            the image as PIL image.
        """

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
        
        return img


    def _transform(self, img, normalize = True,
                   hflip = False, vflip = False, randrot = False, colordistort = False, randerase = False,
                   data_format = None):
        """ Loads an image file and applies normalization and data augmentation.
        
        # Arguments:

        - img: the unnormalized and untransformed image as PIL image.

        - normalize: If True, the image will be normalized by subtracting the channel-wise mean and dividing by the channel-wise standard deviation.

        - hflip: If True, the image will be flipped horizontally with a chance of 50%.

        - vflip: If True, the image will be flipped vertically with a chance of 50%.

        - randerase: If True, random erasing will be applied with probability `self.randerase_prob`.

        - data_format: The image data format (either 'channels_first' or 'channels_last'). Set to None for the default value.

        # Returns:
            the transformed image as 3-dimensional numpy array.
        """
        
        if data_format is None:
            data_format = K.image_data_format()
        
        # Rotate image
        if randrot and (self.randrot_max > 0):
            angle = np.random.uniform(-self.randrot_max, self.randrot_max)
            img = img.rotate(angle, PIL.Image.BILINEAR)

        # Convert PIL image to array
        img = img_to_array(img, data_format=data_format)

        # Color distortions
        if colordistort:
            img = distort_color(img, data_format=data_format, **self.colordistort_params)
        
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


    def _load_and_transform(self, filename, target_size = None, normalize = True,
                            hflip = False, vflip = False, randzoom = False, randrot = False, colordistort = False, randerase = False,
                            data_format = None):
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
        
        return self._transform(
            self._load_image(filename, target_size=target_size, randzoom=randzoom),
            normalize=normalize, hflip=hflip, vflip=vflip, randrot=randrot, colordistort=colordistort, randerase=randerase, data_format=data_format
        )
    
    
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
    

    @property
    def num_channels(self):
        """ Number of channels (e.g., 3 for RGB, 1 for grayscale). """

        return 3



class TinyDatasetGenerator(object):
    """ Abstract base class for datasets with low-resolution images that fit entirely into memory (e.g., CIFAR). """

    def __init__(self, X_train, X_test, y_train, y_test,
                 generator_kwargs = { 'featurewise_center' : True, 'featurewise_std_normalization' : True },
                 train_generator_kwargs = { 'horizontal_flip' : True, 'width_shift_range' : 0.15, 'height_shift_range' : 0.15 }):
        """ Abstract base class for interfaces to datasets with low-resolution images that fit entirely into memory (e.g., CIFAR).

        # Arguments:

        - X_train: 4-D numpy array with the training images.

        - X_test: 4-D numpy array with the test images.

        - y_train: list with numeric labels for the training images.

        - y_test: list with numeric labels for the test images.

        - generator_kwargs: Dictionary with keyword arguments passed to Keras' ImageDataGenerator for both training and test.

        - train_generator_kwargs: Dictionary with keyword arguments passed to Keras' ImageDataGenerator for the training set.
        """
        
        super(TinyDatasetGenerator, self).__init__()

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Set up pre-processing
        self.image_generator = ImageDataGenerator(**generator_kwargs, **train_generator_kwargs)
        self.image_generator.fit(self.X_train)

        self.test_image_generator = ImageDataGenerator(**generator_kwargs)
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
    

    @property
    def num_channels(self):
        """ Number of channels (e.g., 3 for RGB, 1 for grayscale). """

        return self.X_train.shape[-1]



def distort_color(img, fast_mode=True,
                  brightness_delta=32./255., hue_delta=0.2, saturation_range=(0.5, 1.5), contrast_range=(0.5, 1.5),
                  data_format='channels_last'):
    
    nonnormalized = (img.max() > 2.0)
    if nonnormalized:
        img = img.astype(np.float32) / 255.
    if data_format == 'channels_first':
        img = np.transpose(img, (1, 2, 0))
    if (not nonnormalized) and (data_format == 'channels_last'):
        img = img.copy()

    noop = lambda x: x
    brightness_hsv = (lambda x: random_brightness_hsv(x, max_delta=brightness_delta)) if brightness_delta > 0 else noop
    saturation = (lambda x: random_saturation(x, *saturation_range)) if (saturation_range[0] <= saturation_range[1]) and ((saturation_range[0] != 1) or (saturation_range[1] != 1)) else noop
    
    if fast_mode:

        ordering = np.random.choice(2)
        if ordering == 0:
            img = hsv_to_rgb(saturation(brightness_hsv(rgb_to_hsv(img))))
        else:
            img = hsv_to_rgb(brightness_hsv(saturation(rgb_to_hsv(img))))

    else:

        brightness = (lambda x: random_brightness(x, max_delta=brightness_delta)) if brightness_delta > 0 else noop
        hue = (lambda x: random_hue(x, max_delta=hue_delta)) if hue_delta > 0 else noop
        contrast = (lambda x: random_contrast(x, *contrast_range)) if (contrast_range[0] <= contrast_range[1]) and ((contrast_range[0] != 1) or (contrast_range[1] != 1)) else noop

        ordering = np.random.choice(4)
        if ordering == 0:
            img = contrast(hsv_to_rgb(hue(saturation(rgb_to_hsv(brightness(img))))))
        elif ordering == 1:
            img = hsv_to_rgb(hue(rgb_to_hsv(contrast(brightness(hsv_to_rgb(saturation(rgb_to_hsv(img))))))))
        elif ordering == 2:
            img = hsv_to_rgb(saturation(brightness_hsv(hue(rgb_to_hsv(contrast(img))))))
        elif ordering == 3:
            img = brightness(contrast(hsv_to_rgb(saturation(hue(rgb_to_hsv(img))))))

    if data_format == 'channels_first':
        img = np.transpose(img, (2, 0, 1))
    if nonnormalized:
        img = img * 255.
    
    return img


def random_brightness(img, max_delta=32./255.):
    """ Randomly adjusts the brightness of a given RGB image. """
    
    img += np.random.uniform(-max_delta, max_delta)
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def random_brightness_hsv(img, max_delta=32./255.):
    """ Randomly adjusts the brightness of a given HSV image. """
    
    val = img[:,:,2]
    val += np.random.uniform(-max_delta, max_delta)
    val[val > 1] = 1
    val[val < 0] = 0
    return img


def random_hue(img, max_delta=0.2):
    """ Randomly shifts the hue of a given HSV image. """
    
    delta = np.random.uniform(-max_delta, max_delta)
    hue = img[:,:,0]
    hue += delta
    hue[hue > 1.0] -= 1.0
    hue[hue < 0.0] += 1.0
    return img


def random_saturation(img, low=0.5, high=1.5):
    """ Randomly scales the saturation of a given HSV image. """
    
    sat = img[:,:,1]
    sat *= np.random.uniform(low, high)
    sat[sat > 1] = 1
    sat[sat < 0] = 0
    return img


def random_contrast(img, low=0.5, high=1.5):
    """ Randomly scales the contrast of a given RGB image. """
    
    mean = img.mean(axis=(0,1), keepdims=True)
    cf = np.random.uniform(low, high, mean.shape)
    img -= mean
    img *= cf
    img += mean
    img[img > 1] = 1
    img[img < 0] = 0
    return img

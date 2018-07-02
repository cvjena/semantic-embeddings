import numpy as np
import pickle
import PIL.Image
import os
import warnings

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



DATASETS = ['CIFAR-10', 'CIFAR-100', 'CIFAR-100-a', 'CIFAR-100-b', 'CIFAR-100-a-consec', 'CIFAR-100-b-consec', 'ILSVRC', 'NAB', 'NAB-cropped', 'NAB-cropped-sq']



def get_data_generator(dataset, data_root, classes = None):
    
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
    elif dataset == 'nab':
        return NABGenerator(data_root, classes, 'images')
    elif dataset == 'nab-cropped':
        return NABGenerator(data_root, classes, 'images_cropped', mean = [121.29065134, 121.44002115, 109.69898554], std = [50.45762169, 42.66789459, 40.12496913])
    elif dataset == 'nab-cropped-sq':
        return NABGenerator(data_root, classes, 'images_cropped_sq', mean = [124.31297374, 127.28434878, 115.89100229], std = [42.16245978, 30.43073297, 25.00008084])
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))



class DataSequence(Sequence):

    def __init__(self, data_generator, ids, labels, batch_size = 32, shuffle = False, batch_transform = None, batch_transform_kwargs = {}, **kwargs):

        super(DataSequence, self).__init__()
        self.data_generator = data_generator
        self.ids = ids
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_transform = batch_transform
        self.batch_transform_kwargs = batch_transform_kwargs
        self.kwargs = kwargs

        self.ind = np.arange(len(self.ids))
        self.on_epoch_end()


    def __len__(self):

        return int(np.ceil(len(self.ids) / self.batch_size))


    def __getitem__(self, idx):
        
        batch_ind = self.ind[idx*self.batch_size:(idx+1)*self.batch_size]
        X = self.data_generator.compose_batch([self.ids[i] for i in batch_ind], **self.kwargs)
        y = self.labels[batch_ind]
        if self.batch_transform is not None:
            return self.batch_transform(X, y, **self.batch_transform_kwargs)  # pylint: disable=not-callable
        else:
            return X, y


    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.ind)



class CifarGenerator(object):

    def __init__(self, root_dir, classes = None, reenumerate = False, cifar10 = False,
                 randzoom_range = 0., rotation_range = 0.):
        
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
        
        image_generator = self.image_generator if augment else self.test_image_generator
        return image_generator.flow(self.X_train, self.y_train if include_labels else None,
                                    batch_size=batch_size, shuffle=shuffle)
    
    
    def flow_test(self, batch_size = 32, include_labels = True, shuffle = False, augment = False):
        
        image_generator = self.image_generator if augment else self.test_image_generator
        return image_generator.flow(self.X_test, self.y_test if include_labels else None,
                                    batch_size=batch_size, shuffle=shuffle)


    def train_sequence(self, batch_size = 32, shuffle = True, augment = True, batch_transform = None, batch_transform_kwargs = {}):
        
        return DataSequence(self, np.arange(len(self.X_train)), self.y_train,
                            train=True, augment=augment,
                            batch_size=batch_size, shuffle=shuffle, batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)
    
    
    def test_sequence(self, batch_size = 32, shuffle = False, augment = False, batch_transform = None, batch_transform_kwargs = {}):
        
        return DataSequence(self, np.arange(len(self.X_test)), self.y_test,
                            train=False, augment=augment,
                            batch_size=batch_size, shuffle=shuffle, batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)
    

    def compose_batch(self, indices, train, augment = False):

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
        
        return self.y_train
    
    
    @property
    def labels_test(self):
        
        return self.y_test
    
    
    @property
    def num_classes(self):
        
        return max(self.y_train) + 1
    
    
    @property
    def num_train(self):
        
        return len(self.X_train)
    
    
    @property
    def num_test(self):
        
        return len(self.X_test)



class FileDatasetGenerator(object):

    def __init__(self, root_dir, classes = None, cropsize = (224, 224), randzoom_range = None, randerase_prob = 0.0, randerase_params = { 'sl' : 0.02, 'sh' : 0.4, 'r1' : 0.3, 'r2' : 1./0.3 }):
        
        super(FileDatasetGenerator, self).__init__()
        
        self.root_dir = root_dir
        self.cropsize = cropsize
        self.randzoom_range = randzoom_range
        self.randerase_prob = randerase_prob
        self.randerase_params = randerase_params
        
        self.classes = []
        self.train_img_files = []
        self.test_img_files = []
        self._train_labels = []
        self._test_labels = []
        
        warnings.filterwarnings('ignore', '.*[Cc]orrupt EXIF data.*', UserWarning)
    
    
    def _compute_stats(self, mean = None, std = None):
        """ Computes channel-wise mean and standard deviation of all images in the dataset. """
        
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
        
        return self._flow(self.train_img_files, self._train_labels if include_labels else None,
                          batch_size=batch_size, shuffle=shuffle, target_size=target_size,
                          normalize=True, hflip=augment, vflip=False, randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment)
    
    
    def flow_test(self, batch_size = 32, include_labels = True, shuffle = False, target_size = None, augment = False):
        
        return self._flow(self.test_img_files, self._test_labels if include_labels else None,
                          batch_size=batch_size, shuffle=shuffle, target_size=target_size,
                          normalize=True, hflip=augment, vflip=False, randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment)
    

    def train_sequence(self, batch_size = 32, shuffle = True, target_size = None, augment = True, batch_transform = None, batch_transform_kwargs = {}):
        
        return DataSequence(self, self.train_img_files, self._train_labels,
                            batch_size=batch_size, shuffle=shuffle,
                            target_size=target_size, normalize=True, hflip=augment, vflip=False,
                            randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment,
                            batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)
    
    
    def test_sequence(self, batch_size = 32, shuffle = False, target_size = None, augment = False, batch_transform = None, batch_transform_kwargs = {}):
        
        return DataSequence(self, self.test_img_files, self._test_labels,
                            batch_size=batch_size, shuffle=shuffle,
                            target_size=target_size, normalize=True, hflip=augment, vflip=False,
                            randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment,
                            batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)
    
    
    def _flow(self, filenames, labels = None, batch_size = 32, shuffle = False, **kwargs):
        
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
        
        if data_format is None:
            data_format = K.image_data_format()
        
        # Load and resize image
        img = load_img(filename)
        if (target_size is not None) or (randzoom and (self.randzoom_range is not None)):
            if target_size is None:
                target_size = img.size
            if randzoom and (self.randzoom_range is not None):
                if isinstance(self.randzoom_range[0], float):
                    target_size = np.round(np.array(target_size) * np.random.uniform(self.randzoom_range[0], self.randzoom_range[1])).astype(int).tolist()
                else:
                    target_size = np.random.randint(self.randzoom_range[0], self.randzoom_range[1])
            if isinstance(target_size, int):
                target_size = (target_size, round(img.size[1] * (target_size / img.size[0]))) if img.size[0] < img.size[1] else (round(img.size[0] * (target_size / img.size[1])), target_size)
            img = img.resize(target_size, PIL.Image.BILINEAR)
        img = img_to_array(img)
        
        # Normalize image
        if normalize:
            img -= self.mean[:,None,None] if data_format == 'channels_first' else self.mean[None,None,:]
            img /= self.std[:,None,None] if data_format == 'channels_first' else self.std[None,None,:]
        
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
        
        return self._train_labels
    
    
    @property
    def labels_test(self):
        
        return self._test_labels
    
    
    @property
    def num_classes(self):
        
        return len(self.classes)
    
    
    @property
    def num_train(self):
        
        return len(self.train_img_files)
    
    
    @property
    def num_test(self):
        
        return len(self.test_img_files)



class ILSVRCGenerator(FileDatasetGenerator):

    def __init__(self, root_dir, classes = None, mean = [122.65435242, 116.6545058, 103.99789959], std = [71.40583196, 69.56888997, 73.0440314]):
        
        super(ILSVRCGenerator, self).__init__(root_dir, classes, randzoom_range = (256, 480))
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
    

    def flow_test(self, batch_size = 32, include_labels = True, shuffle = False, target_size = 256, augment = False):
        
        return self._flow(self.test_img_files, self._test_labels if include_labels else None,
                          batch_size=batch_size, shuffle=shuffle, target_size=target_size,
                          normalize=True, hflip=augment, vflip=False, randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment)
    
    
    def test_sequence(self, batch_size = 32, shuffle = False, target_size = 256, augment = False, batch_transform = None, batch_transform_kwargs = {}):
        
        return DataSequence(self, self.test_img_files, self._test_labels,
                            batch_size=batch_size, shuffle=shuffle,
                            target_size=target_size, normalize=True, hflip=augment, vflip=False,
                            randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment,
                            batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)



class NABGenerator(FileDatasetGenerator):

    def __init__(self, root_dir, classes = None, img_dir = 'images',
                 cropsize = (224, 224), randzoom_range = None, randerase_prob = 0.5, randerase_params = { 'sl' : 0.02, 'sh' : 0.3, 'r1' : 0.3, 'r2' : 1./0.3 },
                 mean = [125.30513277, 129.66606421, 118.45121113], std = [57.0045467, 56.70059436, 68.44430446]):
        
        super(NABGenerator, self).__init__(root_dir, classes = classes, cropsize = cropsize, randzoom_range = randzoom_range, randerase_prob = randerase_prob, randerase_params = randerase_params)
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
    
    
    def flow_train(self, batch_size = 32, include_labels = True, shuffle = True, target_size = 256, augment = True):
        
        return self._flow(self.train_img_files, self._train_labels if include_labels else None,
                          batch_size=batch_size, shuffle=shuffle, target_size=target_size,
                          normalize=True, hflip=augment, vflip=False, randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment)
    
    
    def flow_test(self, batch_size = 32, include_labels = True, shuffle = False, target_size = 256, augment = False):
        
        return self._flow(self.test_img_files, self._test_labels if include_labels else None,
                          batch_size=batch_size, shuffle=shuffle, target_size=target_size,
                          normalize=True, hflip=augment, vflip=False, randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment)


    def train_sequence(self, batch_size = 32, shuffle = True, target_size = 256, augment = True, batch_transform = None, batch_transform_kwargs = {}):
        
        return DataSequence(self, self.train_img_files, self._train_labels,
                            batch_size=batch_size, shuffle=shuffle,
                            target_size=target_size, normalize=True, hflip=augment, vflip=False,
                            randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment,
                            batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)
    
    
    def test_sequence(self, batch_size = 32, shuffle = False, target_size = 256, augment = False, batch_transform = None, batch_transform_kwargs = {}):
        
        return DataSequence(self, self.test_img_files, self._test_labels,
                            batch_size=batch_size, shuffle=shuffle,
                            target_size=target_size, normalize=True, hflip=augment, vflip=False,
                            randzoom=augment, cropsize=self.cropsize, randcrop=augment, randerase=augment,
                            batch_transform=batch_transform, batch_transform_kwargs=batch_transform_kwargs)

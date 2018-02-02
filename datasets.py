import numpy as np
import pickle
import os
import warnings

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, list_pictures
from keras import backend as K

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        for x in it:
            yield x



DATASETS = ['CIFAR-100', 'CIFAR-100-a', 'CIFAR-100-b', 'CIFAR-100-a-consec', 'CIFAR-100-b-consec', 'ILSVRC']



def get_data_generator(dataset, data_root, classes = None):
    
    dataset = dataset.lower()
    if dataset == 'cifar-100':
        return CifarGenerator(data_root, classes, True)
    elif dataset.startswith('cifar-100-a'):
        return CifarGenerator(data_root, np.arange(50), dataset.endswith('-consec'))
    elif dataset.startswith('cifar-100-b'):
        return CifarGenerator(data_root, np.arange(50, 100), dataset.endswith('-consec'))
    elif dataset == 'ilsvrc':
        return ILSVRCGenerator(data_root, classes)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))



class CifarGenerator(object):

    def __init__(self, root_dir, classes = None, reenumerate = False):
        
        super(CifarGenerator, self).__init__()
        self.root_dir = root_dir

        # Load dataset
        with open(os.path.join(self.root_dir, 'train'), 'rb') as pf:
            dump = pickle.load(pf, encoding='bytes')
            self.X_train, self.y_train = dump[b'data'].astype(np.float32), dump[b'fine_labels']
            del dump

        with open(os.path.join(self.root_dir, 'test'), 'rb') as pf:
            dump = pickle.load(pf, encoding='bytes')
            self.X_test, self.y_test = dump[b'data'].astype(np.float32), dump[b'fine_labels']
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

        # Reshape data to images
        self.X_train = self.X_train.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        self.X_test = self.X_test.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

        # Set up pre-processing
        self.image_generator = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True, horizontal_flip = True,
                                                  width_shift_range = 0.15, height_shift_range = 0.15)
        self.image_generator.fit(self.X_train)

        self.test_image_generator = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True)
        self.test_image_generator.fit(self.X_train)
    
    
    def flow_train(self, batch_size = 32, include_labels = True, shuffle = True):
        
        return self.image_generator.flow(self.X_train, self.y_train if include_labels else None,
                                         batch_size=batch_size, shuffle=shuffle)
    
    
    def flow_test(self, batch_size = 32, include_labels = True, shuffle = False):
        
        return self.test_image_generator.flow(self.X_test, self.y_test if include_labels else None,
                                              batch_size=batch_size, shuffle=shuffle)
    
    
    @property
    def num_classes(self):
        
        return max(self.y_train) + 1
    
    
    @property
    def num_train(self):
        
        return len(self.X_train)
    
    
    @property
    def num_test(self):
        
        return len(self.X_test)



class ILSVRCGenerator(object):

    def __init__(self, root_dir, classes = None, mean = [122.65435242, 116.6545058, 103.99789959], std = [72.39054456, 74.6065602, 75.43971812]):
        
        super(ILSVRCGenerator, self).__init__()
        self.root_dir = root_dir
        self.train_dir = os.path.join(root_dir, 'ILSVRC2012_img_train')
        self.test_dir = os.path.join(root_dir, 'ILSVRC2012_img_val')
        
        # Search for classes
        if classes is None:
            classes = []
            for subdir in sorted(os.listdir(self.train_dir)):
                if os.path.isdir(os.path.join(self.train_dir, subdir)):
                    classes.append(subdir)
        self.classes = classes
        self.class_indices = dict(zip(self.classes, range(len(self.classes))))
        
        # Search for images
        self.train_img_files = []
        self.test_img_files = []
        self.train_labels = []
        self.test_labels = []
        for lbl, subdir in enumerate(self.classes):
            cls_files = list_pictures(os.path.join(self.train_dir, subdir), 'JPE?G|jpe?g')
            self.train_img_files += cls_files
            self.train_labels += [lbl] * len(cls_files)
            cls_files = list_pictures(os.path.join(self.test_dir, subdir), 'JPE?G|jpe?g')
            self.test_img_files += cls_files
            self.test_labels += [lbl] * len(cls_files)
        print('Found {} training and {} validation images from {} classes.'.format(self.num_train, self.num_test, self.num_classes))
        
        # Compute mean and standard deviation
        if mean is None:
            mean = 0
            for fn in tqdm(self.train_img_files, desc = 'Computing channel mean'):
                mean += np.mean(np.asarray(load_img(fn), dtype=np.float64), axis = (0,1))
            mean /= len(self.train_img_files)
            print('Channel-wise mean:               {}'.format(mean))
        self.mean = np.asarray(mean, dtype=np.float32)
        if (mean is None) or (std is None):
            std = 0
            for img in tqdm(self.train_img_files, desc = 'Computing channel variance'):
                std += np.mean((np.asarray(load_img(fn), dtype=np.float64) - self.mean) ** 2, axis = (0,1))
            std = np.sqrt(std / (len(self.train_img_files) - 1))
            print('Channel-wise standard deviation: {}'.format(std))
        self.std = np.asarray(std, dtype=np.float32)
        
        # Suppress warnings about corrupt EXIF data
        warnings.filterwarnings('ignore', '.*[Cc]orrupt EXIF data.*', UserWarning)
    
    
    def flow_train(self, batch_size = 32, include_labels = True, shuffle = True, target_size = None):
        
        return self._flow(self.train_img_files, self.train_labels if include_labels else None,
                          batch_size=batch_size, shuffle=shuffle, target_size=target_size,
                          normalize=True, hflip=True, vflip=False, cropsize=(224,224), randcrop=True)
    
    
    def flow_test(self, batch_size = 32, include_labels = True, shuffle = False, target_size = None):
        
        return self._flow(self.test_img_files, self.test_labels if include_labels else None,
                          batch_size=batch_size, shuffle=shuffle, target_size=target_size,
                          normalize=True, hflip=False, vflip=False, cropsize=(224,224), randcrop=False)
    
    
    def _flow(self, filenames, labels = None, batch_size = 32, shuffle = False, cropsize = None, randcrop = False, data_format = None, **kwargs):
        
        if data_format is None:
            data_format = K.image_data_format()
        if data_format == 'channels_first':
            x_axis, y_axis = 2, 1
        else:
            x_axis, y_axis = 1, 0
        
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
            
            X = [self._load_and_transform(filenames[i], data_format=data_format, **kwargs) for i in batch_ind]
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
            X = np.stack(X)
            
            if labels is not None:
                yield X, labels[batch_ind]
            else:
                yield X
    
    
    def _load_and_transform(self, filename, target_size = None, normalize = True, hflip = False, vflip = False, data_format = None):
        
        if data_format is None:
            data_format = K.image_data_format()
        
        img = img_to_array(load_img(filename, target_size=target_size))
        
        if normalize:
            img -= self.mean[:,None,None] if data_format == 'channels_first' else self.mean[None,None,:]
            img /= self.std[:,None,None] if data_format == 'channels_first' else self.std[None,None,:]
        
        if hflip and (np.random.randn() < 0.5):
            img = img[:,:,::-1] if data_format == 'channels_first' else img[:,::-1,:]
        
        if vflip and (np.random.randn() < 0.5):
            img = img[:,::-1,:] if data_format == 'channels_first' else img[::-1,:,:]
        
        return img
    
    
    @property
    def num_classes(self):
        
        return len(self.classes)
    
    
    @property
    def num_train(self):
        
        return len(self.train_img_files)
    
    
    @property
    def num_test(self):
        
        return len(self.test_img_files)

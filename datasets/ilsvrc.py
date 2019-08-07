import os

try:
    from keras.preprocessing.image import list_pictures
except ImportError:
    import keras
    from keras_preprocessing.image import list_pictures

from . import IMAGENET_MEAN, IMAGENET_STD
from .common import FileDatasetGenerator



class ILSVRCGenerator(FileDatasetGenerator):

    def __init__(self, root_dir, classes = None, mean = IMAGENET_MEAN, std = IMAGENET_STD, color_mode = "rgb"):
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
        
        super(ILSVRCGenerator, self).__init__(root_dir, default_target_size = 256, randzoom_range = (256, 480), color_mode = color_mode)
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
            cls_files = sorted(list_pictures(os.path.join(self.train_dir, subdir), 'jpeg'))
            self.train_img_files += cls_files
            self._train_labels += [lbl] * len(cls_files)
            cls_files = sorted(list_pictures(os.path.join(self.test_dir, subdir), 'jpeg'))
            self.test_img_files += cls_files
            self._test_labels += [lbl] * len(cls_files)
        print('Found {} training and {} validation images from {} classes.'.format(self.num_train, self.num_test, self.num_classes))
        
        # Compute mean and standard deviation
        self._compute_stats(mean, std)

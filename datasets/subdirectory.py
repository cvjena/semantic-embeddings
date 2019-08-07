import os
from glob import glob

from .common import FileDatasetGenerator



class SubDirectoryGenerator(FileDatasetGenerator):

    def __init__(self, root_dir, classes = None, img_dir = '.', train_list = 'train.txt', test_list = 'test.txt',
                 cropsize = (224, 224), default_target_size = 256, randzoom_range = None, randerase_prob = 0.5, randerase_params = { 'sl' : 0.02, 'sh' : 0.3, 'r1' : 0.3, 'r2' : 1./0.3 },
                 mean = None, std = None, color_mode = "rgb"):
        """ Data generator for images organized in sub-directories.

        This generator expects all images belonging to a class to be placed in a sub-directory named after that class.
        For splitting the dataset into a training and a validation partition, two files can be provided that list the
        images belonging to each partition.


        # Arguments:

        - root_dir: Root directory of the dataset, containing the images for each class in separate sub-directories.

        - classes: List of classes to restrict the dataset to. New numeric labels will be assigned to these classes in ascending order.
                   If set to `None`, all available classes will be used and enumerated in ascending order.

        - img_dir: Directory (relative to root_dir) containing the images for each class in separate sub-directories.

        - train_list: Name of a text file (relative to root_dir) listing all training images (relative to img_dir), one per line.

        - test_list: Name of a text file (relative to root_dir) listing all validation images (relative to img_dir), one per line.

        - cropsize: Tuple with width and height of crops extracted from the images.

        - default_target_size: Int or tuple of ints. Specifies the default target size which images will be resized to (before cropping)
                               if not specified differently in calls to `flow_train/test` or `train/test_sequence`.
                               If a single int is given, it specifies the size of the smaller side of the image and the aspect ratio will be retained.
                               If set to -1, the image won't be resized.

        - randzoom_range: Tuple with minimum and maximum size of the smaller image dimension for random scale augmentation.
                          May either be given as integer specifying absolute pixel values or float specifying the relative scale of the image.
                          If set to `None`, no scale augmentation will be performed.
        
        - randerase_prob: Probability for random erasing.

        - randerase_params: Random erasing parameters (see Zhong et al. (2017): "Random erasing data augmentation.").
        
        - mean: Channel-wise image mean for normalization (in "RGB" order). If set to `None`, mean and standard deviation will be computed from the images.

        - std: Channel-wise standard deviation for normalization (in "RGB" order). If set to `None`, standard deviation will be computed from the images.

        - color_mode: Image color mode, either "rgb" or "bgr".
        """
        
        super(SubDirectoryGenerator, self).__init__(root_dir, cropsize = cropsize, default_target_size = default_target_size, randzoom_range = randzoom_range,
                                                    randerase_prob = randerase_prob, randerase_params = randerase_params, color_mode = color_mode)

        self.img_dir = img_dir if os.path.isabs(img_dir) else os.path.join(root_dir, img_dir)

        # Determine set of classes
        if classes is not None:
            self.classes = classes
        else:
            self.classes = sorted(os.path.basename(dirname) for dirname in glob(os.path.join(self.img_dir, '*')) if (not os.path.basename(dirname).startswith('.')) and os.path.isdir(dirname))
        self.class_indices = dict(zip(self.classes, range(len(self.classes))))

        # Search for images
        with open(train_list if os.path.isabs(train_list) else os.path.join(root_dir, train_list)) as f:
            for l in f:
                if l.strip() != '':
                    classname = os.path.dirname(l.strip())
                    if classname in self.class_indices:
                        self.train_img_files.append(os.path.join(self.img_dir, l.strip()))
                        self._train_labels.append(self.class_indices[classname])
        with open(test_list if os.path.isabs(test_list) else os.path.join(root_dir, test_list)) as f:
            for l in f:
                if l.strip() != '':
                    classname = os.path.dirname(l.strip())
                    if classname in self.class_indices:
                        self.test_img_files.append(os.path.join(self.img_dir, l.strip()))
                        self._test_labels.append(self.class_indices[classname])
        
        print('Found {} training and {} validation images from {} classes.'.format(self.num_train, self.num_test, self.num_classes))
        
        # Compute mean and standard deviation
        self._compute_stats(mean, std)
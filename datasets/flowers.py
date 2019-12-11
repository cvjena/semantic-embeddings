import os
import scipy.io

from .common import FileDatasetGenerator



class FlowersGenerator(FileDatasetGenerator):

    def __init__(self, root_dir, classes = None, img_dir = 'jpg', label_file = 'imagelabels.mat', split_file = 'setid.mat',
                 train_splits = ['trnid', 'valid'], test_splits = ['tstid'],
                 cropsize = (448, 448), default_target_size = 512, randzoom_range = None, distort_colors = False,
                 randerase_prob = 0.5, randerase_params = { 'sl' : 0.02, 'sh' : 0.3, 'r1' : 0.3, 'r2' : 1./0.3 },
                 mean = [110.7799141, 97.65648664, 75.32889973], std = [74.90387818, 62.70218863, 69.7656359], color_mode = "rgb"):
        """ Flowers-102 data generator.

        The dataset can be obtained here:
        http://www.robots.ox.ac.uk/~vgg/data/flowers/


        # Arguments:

        - root_dir: Root directory of the Oxford Flowers-102 dataset, containing the image directory and the annotation files.

        - classes: List of classes to restrict the dataset to. New numeric labels will be assigned to these classes in ascending order.
                   If set to `None`, all available classes will be used and enumerated in ascending order.
                   Note that the classes in Flowers-102 are enumerated beginning with 1, while we will begin with 0.

        - img_dir: Name of the sub-directory of `root_dir` that contains the images, named like 'image_#####.jpg', where '#####' is the
                   5-digit ID of the image.

        - label_file: Name of a MATLAB file (relative to `root_dir`) that contains an array called 'labels' with class labels for each image.

        - split_file: Name of a MATLAB file (relative to `root_dir`) that containing named arrays of image IDs (counting from 1) defining a
                      split of the dataset.

        - train_splits: List with the names of the arrays in `split_file` which should be concatenated to obtain the list of training image IDs.

        - test_splits: List with the names of the arrays in `split_file` which should be concatenated to obtain the list of test image IDs.

        - cropsize: Tuple with width and height of crops extracted from the images.

        - default_target_size: Int or tuple of ints. Specifies the default target size which images will be resized to (before cropping)
                               if not specified differently in calls to `flow_train/test` or `train/test_sequence`.
                               If a single int is given, it specifies the size of the smaller side of the image and the aspect ratio will be retained.
                               If set to -1, the image won't be resized.

        - randzoom_range: Tuple with minimum and maximum size of the smaller image dimension for random scale augmentation.
                          May either be given as integer specifying absolute pixel values or float specifying the relative scale of the image.
                          If set to `None`, no scale augmentation will be performed.
        
        - distort_colors: Boolean specifying whether to apply color distortions as data augmentation.
        
        - randerase_prob: Probability for random erasing.

        - randerase_params: Random erasing parameters (see Zhong et al. (2017): "Random erasing data augmentation.").
        
        - mean: Channel-wise image mean for normalization (in "RGB" order). If set to `None`, mean and standard deviation will be computed from the images.

        - std: Channel-wise standard deviation for normalization (in "RGB" order). If set to `None`, standard deviation will be computed from the images.

        - color_mode: Image color mode, either "rgb" or "bgr".
        """
        
        super(FlowersGenerator, self).__init__(root_dir, cropsize = cropsize, default_target_size = default_target_size, randzoom_range = randzoom_range,
                                               distort_colors=distort_colors, colordistort_params={ 'hue_delta' : 0.0, 'saturation_range' : (0.8, 1.2) },
                                               randerase_prob = randerase_prob, randerase_params = randerase_params, color_mode = color_mode)
        self.img_dir = img_dir if os.path.isabs(img_dir) else os.path.join(self.root_dir, img_dir)
        self.label_file = label_file if os.path.isabs(label_file) else os.path.join(self.root_dir, label_file)
        self.split_file = split_file if os.path.isabs(split_file) else os.path.join(self.root_dir, split_file)
        
        # Read annotations
        img_labels = scipy.io.loadmat(self.label_file, squeeze_me=True)['labels']
        splits = scipy.io.loadmat(self.split_file, squeeze_me=True)

        # Determine set of classes
        self.classes = classes if classes is not None else sorted(set(img_labels))
        self.class_indices = dict(zip(self.classes, range(len(self.classes))))

        # Compose image lists
        for split_name in train_splits:
            for i in splits[split_name]:
                self.train_img_files.append(os.path.join(self.img_dir, 'image_{:05d}.jpg'.format(i)))
                self._train_labels.append(self.class_indices[img_labels[i-1]])
        for split_name in test_splits:
            for i in splits[split_name]:
                self.test_img_files.append(os.path.join(self.img_dir, 'image_{:05d}.jpg'.format(i)))
                self._test_labels.append(self.class_indices[img_labels[i-1]])
        print('Found {} training and {} validation images from {} classes.'.format(self.num_train, self.num_test, self.num_classes))
        
        # Compute mean and standard deviation
        self._compute_stats(mean, std)
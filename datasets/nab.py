import os

from .common import FileDatasetGenerator, DataSequence



class NABGenerator(FileDatasetGenerator):

    def __init__(self, root_dir, classes = None, img_dir = 'images', img_list_file = 'images.txt', split_file = 'train_test_split.txt', label_file = 'image_class_labels.txt',
                 cropsize = (224, 224), default_target_size = 256, randzoom_range = None, randerase_prob = 0.5, randerase_params = { 'sl' : 0.02, 'sh' : 0.3, 'r1' : 0.3, 'r2' : 1./0.3 },
                 mean = [125.30513277, 129.66606421, 118.45121113], std = [57.0045467, 56.70059436, 68.44430446], color_mode = "rgb"):
        """ NABirds and CUB-200-2011 data generator.

        # Arguments:

        - root_dir: Root directory of the NAB/CUB dataset, containing the files `images.txt`, `image_class_labels.txt`, and `train_test_split.txt`.

        - classes: List of classes to restrict the dataset to. Numeric labels will be assigned to these classes in ascending order.
                   If set to `None`, all available classes will be used and enumerated in ascending order.
        
        - img_dir: Name of the sub-directory of `root_dir` that contains the images in further sub-directories named by their class label.

        - img_list_file: Name of a file (relative to `root_dir`) that contains tuples of image IDs and their filenames, separated by white-spaces, one tuple per line.

        - split_file: Name of a file (relative to `root_dir`) that specifies the training/test split as tuples of image IDs and either "1" (for training)
                      or "0" (for test) images.
        
        - label_file: Name of a file (relative to `root_dir`) that specifies the class labels for each image as tuples of image IDs and class labels, one per line,
                      separated by white-spaces.

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
        self.img_list_file = os.path.join(root_dir, img_list_file)
        self.label_file = os.path.join(root_dir, label_file)
        self.split_file = os.path.join(root_dir, split_file)
        
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
                    if (img_id in is_train) and (img_labels[img_id] in self.class_indices):
                        if is_train[img_id]:
                            self.train_img_files.append(os.path.join(self.imgs_dir, fn))
                            self._train_labels.append(self.class_indices[img_labels[img_id]])
                        else:
                            self.test_img_files.append(os.path.join(self.imgs_dir, fn))
                            self._test_labels.append(self.class_indices[img_labels[img_id]])
        print('Found {} training and {} validation images from {} classes.'.format(self.num_train, self.num_test, self.num_classes))
        
        # Compute mean and standard deviation
        self._compute_stats(mean, std)

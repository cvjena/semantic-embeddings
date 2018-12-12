import os
import scipy.io

from .common import FileDatasetGenerator



class CarsGenerator(FileDatasetGenerator):

    def __init__(self, root_dir, classes = None, annotation_file = 'cars_annos.mat',
                 cropsize = (448, 448), default_target_size = 512, randzoom_range = None, randerase_prob = 0.5, randerase_params = { 'sl' : 0.02, 'sh' : 0.3, 'r1' : 0.3, 'r2' : 1./0.3 },
                 mean = [120.03730636, 117.33780928, 116.0130335], std = [75.40415763, 75.15394251, 77.28286728], color_mode = "rgb"):
        """ Stanford-Cars data generator.

        The dataset can be obtained here:
        https://ai.stanford.edu/~jkrause/cars/car_dataset.html

        This data generator was designed to work with the merged training+test version of the dataset
        (because labels are not available for the stand-alone test data).


        # Arguments:

        - root_dir: Root directory of the Stanford-Cars dataset, containing the image directory and the annotations file.

        - classes: List of classes to restrict the dataset to. New numeric labels will be assigned to these classes in ascending order.
                   If set to `None`, all available classes will be used and enumerated in ascending order.
                   Note that the classes in Stanford-Cars are enumerated beginning with 1, while we will begin with 0.

        - annotation_file: Name of a MATLAB file (relative to `root_dir`) that contains a structured array called `annotations`, which has
                           at least the following attributes:
                                - 'relative_im_path': paths of the images in the dataset, relative to `root_dir` (not to the location of the annotations file!).
                                - 'class': class labels for all images,
                                - 'test': indicators whether each image belongs to the training (0) or the test set (1).

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
        
        super(CarsGenerator, self).__init__(root_dir, cropsize = cropsize, default_target_size = default_target_size, randzoom_range = randzoom_range,
                                            randerase_prob = randerase_prob, randerase_params = randerase_params, color_mode = color_mode)
        
        # Read annotations
        self.annotation_file = annotation_file if os.path.isabs(annotation_file) else os.path.join(self.root_dir, annotation_file)
        self._annotations = scipy.io.loadmat(self.annotation_file, squeeze_me=True)['annotations']

        # Determine set of classes
        self.classes = classes if classes is not None else sorted(set(self._annotations['class']))
        self.class_indices = dict(zip(self.classes, range(len(self.classes))))

        # Read image information
        for sample in self._annotations:
            if sample['class'] in self.class_indices:
                fn = sample['relative_im_path'] if os.path.isabs(sample['relative_im_path']) else os.path.join(self.root_dir, sample['relative_im_path'])
                if sample['test']:
                    self.test_img_files.append(fn)
                    self._test_labels.append(self.class_indices[sample['class']])
                else:
                    self.train_img_files.append(fn)
                    self._train_labels.append(self.class_indices[sample['class']])
        
        print('Found {} training and {} validation images from {} classes.'.format(self.num_train, self.num_test, self.num_classes))
        
        # Compute mean and standard deviation
        self._compute_stats(mean, std)
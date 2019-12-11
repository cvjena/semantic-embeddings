import numpy as np


CAFFE_MEAN = [123.68, 116.779, 103.939]
CAFFE_STD = [1., 1., 1.]

IMAGENET_MEAN = [122.65435242, 116.6545058, 103.99789959]
IMAGENET_STD = [71.40583196, 69.56888997, 73.0440314]


from .cifar import CifarGenerator
from .ilsvrc import ILSVRCGenerator
from .nab import NABGenerator
from .cars import CarsGenerator
from .flowers import FlowersGenerator
from .inat import INatGenerator
from .subdirectory import SubDirectoryGenerator



def get_data_generator(dataset, data_root, classes = None):
    """ Shortcut for creating a data generator with default settings.

    # Arguments:

    - dataset: The name of the dataset. Possible values are:
               
               - "cifar-10"
               - "cifar-100"
               - "cifar-100-a" (first 50 classes of cifar-100)
               - "cifar-100-b" (last 50 classes of cifar-100)
               - "ilsvrc"
               - "nab"
               - "nab-large"
               - "cub"
               - "cars"
               - "flowers"
               - "mit67scenes"
               - "inat" / "inat2018" (optionally followed by an underscore and the name of a super-category)
               - "inat2019"
               - "UCMLU"
               - "RESISC45"
               
               To all dataset names except CIFAR, you may append one of the following suffixes:

               - "-ilsvrcmean": use ImageNet statistics for pre-processing
               - "-caffe": Caffe-style pre-processing (BGR instead of RGB, ImageNet mean, no standard deviation)

               For NAB and iNaturalist, the suffix "-large" may be appended to set the default image size to 512 pixels
               and the crop size to 448x448.

    - data_root: Root directory of the dataset.

    - classes: Optionally, a list of classes to be included. If not given, all available classes will be used.

    # Returns:
        a data generator object
    """
    
    dataset = dataset.lower()
    
    if dataset.startswith('inat2018'):
        dataset = 'inat' + dataset[8:]

    kwargs = {}
    if dataset.endswith('-ilsvrcmean'):
        kwargs['mean'] = IMAGENET_MEAN
        kwargs['std'] = IMAGENET_STD
        dataset = dataset[:-11]
    elif dataset.endswith('-caffe'):
        kwargs['mean'] = CAFFE_MEAN
        kwargs['std'] = CAFFE_STD
        kwargs['color_mode'] = 'bgr'
        dataset = dataset[:-6]
    if dataset.endswith('-large'):
        kwargs['cropsize'] = (448, 448)
        kwargs['default_target_size'] = 512
        dataset = dataset[:-6]

    if dataset == 'cifar-10':
    
        return CifarGenerator(data_root, classes, reenumerate = True, cifar10 = True,
                              train_generator_kwargs = { 'horizontal_flip' : True, 'width_shift_range' : 0.15, 'height_shift_range' : 0.15, 'zoom_range' : 0.25 })
    
    elif dataset == 'cifar-100':
    
        return CifarGenerator(data_root, classes, reenumerate = True)
    
    elif dataset.startswith('cifar-100-a'):
    
        return CifarGenerator(data_root, np.arange(50), reenumerate = dataset.endswith('-consec'))
    
    elif dataset.startswith('cifar-100-b'):
    
        return CifarGenerator(data_root, np.arange(50, 100), reenumerate = dataset.endswith('-consec'))
    
    elif dataset == 'ilsvrc':
    
        return ILSVRCGenerator(data_root, classes, **kwargs)
    
    elif dataset == 'nab':
    
        if ('default_target_size' not in kwargs) and ('randzoom_range' not in kwargs):
            kwargs['randzoom_range'] = (256, 480)
        return NABGenerator(data_root, classes, 'images', **kwargs)
    
    elif (dataset == 'cub') or dataset.startswith('cub-sub'):
        
        if 'mean' not in kwargs:
            kwargs['mean'] = [123.82988033, 127.35116805, 110.25606303]
        if 'std' not in kwargs:
            kwargs['std'] = [59.2230949, 58.0736071, 67.80251684]
        if dataset.startswith('cub-sub'):
            samples_per_class = int(dataset[7:])
            kwargs['split_file'] = 'train_test_split_{}.txt'.format(samples_per_class)
            kwargs['train_repeats'] = 30 // samples_per_class
        return NABGenerator(data_root, classes, 'images', cropsize = (448, 448), default_target_size = 512, randzoom_range = None, **kwargs)
    
    elif dataset == 'cars':
        
        return CarsGenerator(data_root, classes, **kwargs)
    
    elif dataset == 'flowers':
        
        return FlowersGenerator(data_root, classes, **kwargs)

    elif (dataset == 'inat') or dataset.startswith('inat_'):

        supercategory = dataset[5:] if dataset.startswith('inat_') else None
        if ('default_target_size' not in kwargs) and ('randzoom_range' not in kwargs):
            kwargs['randzoom_range'] = (256, 480)
        return INatGenerator(data_root, supercategory=supercategory, **kwargs)
    
    elif dataset == 'inat2019':

        if ('mean' not in kwargs) and ('std' not in kwargs):
            kwargs['mean'] = [115.77492586, 120.84414891, 93.51744386]
            kwargs['std'] = [60.46127213, 58.63136496, 63.5872299]
        if ('default_target_size' not in kwargs) and ('randzoom_range' not in kwargs):
            kwargs['randzoom_range'] = (256, 480)
        return INatGenerator(data_root, 'train2019.json', 'val2019.json', **kwargs)

    elif dataset == 'mit67scenes':

        if ('mean' not in kwargs) and ('std' not in kwargs):
            kwargs['mean'] = [124.62788179, 110.01028625, 94.95780545]
            kwargs['std'] = [68.56923599, 66.86607736, 67.35944349]
        return SubDirectoryGenerator(data_root, classes, img_dir='Images', train_list='TrainImages.txt', test_list='TestImages.txt', **kwargs)

    elif dataset == 'ucmlu':

        if ('mean' not in kwargs) and ('std' not in kwargs):
            kwargs['mean'] = [122.65409223, 124.40230701, 114.25659171]
            kwargs['std'] = [55.74499679, 51.65585669, 50.16527551]
        return SubDirectoryGenerator(data_root, classes, **kwargs)

    elif dataset == 'resisc45':

        if ('mean' not in kwargs) and ('std' not in kwargs):
            kwargs['mean'] = [94.17769482, 97.40967803, 87.80359702]
            kwargs['std'] = [51.92246172, 47.22081475, 47.07685676]
        return SubDirectoryGenerator(data_root, classes, **kwargs)
    
    else:
        
        raise ValueError('Unknown dataset: {}'.format(dataset))

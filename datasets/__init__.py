import numpy as np


CAFFE_MEAN = [123.68, 116.779, 103.939]
CAFFE_STD = [1., 1., 1.]

IMAGENET_MEAN = [122.65435242, 116.6545058, 103.99789959]
IMAGENET_STD = [71.40583196, 69.56888997, 73.0440314]


from .cifar import CifarGenerator
from .ilsvrc import ILSVRCGenerator
from .nab import NABGenerator



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
               
               To all dataset names except CIFAR, you may append one of the following suffixes:

               - "-ilsvrcmean": use ImageNet statistics for pre-processing
               - "-caffe": Caffe-style pre-processing (BGR instead of RGB, ImageNet mean, no standard deviation)

    - data_root: Root directory of the dataset.

    - classes: Optionally, a list of classes to be included. If not given, all available classes will be used.

    # Returns:
        a data generator object
    """
    
    dataset = dataset.lower()

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

    if dataset == 'cifar-10':
    
        return CifarGenerator(data_root, classes, reenumerate = True, cifar10 = True, randzoom_range = 0.25)
    
    elif dataset == 'cifar-100':
    
        return CifarGenerator(data_root, classes, reenumerate = True)
    
    elif dataset.startswith('cifar-100-a'):
    
        return CifarGenerator(data_root, np.arange(50), reenumerate = dataset.endswith('-consec'))
    
    elif dataset.startswith('cifar-100-b'):
    
        return CifarGenerator(data_root, np.arange(50, 100), reenumerate = dataset.endswith('-consec'))
    
    elif dataset == 'ilsvrc':
    
        return ILSVRCGenerator(data_root, classes, **kwargs)
    
    elif dataset == 'nab':
    
        return NABGenerator(data_root, classes, 'images', randzoom_range = (256, 480), **kwargs)
    
    elif dataset == 'nab-large':
        
        return NABGenerator(data_root, classes, 'images', cropsize = (448, 448), default_target_size = 512, randzoom_range = None, **kwargs)
    
    else:
        
        raise ValueError('Unknown dataset: {}'.format(dataset))

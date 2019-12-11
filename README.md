# Hierarchy-based Image Embeddings for Semantic Image Retrieval

This repository contains the official source code used to produce the results reported in the following papers:

> [**Hierarchy-based Image Embeddings for Semantic Image Retrieval.**][1]  
> Björn Barz and Joachim Denzler.  
> IEEE Winter Conference on Applications of Computer Vision (WACV), 2019.

> [**Deep Learning on Small Datasets without Pre-Training usine Cosine Loss.**][23]  
> Björn Barz and Joachim Denzler.  
> IEEE Winter Conference on Applications of Computer Vision (WACV), 2020.

If you use this code, please cite one of those papers (the first one when you work with hierarchy-based semantic embeddings, the second one when you use the cosine loss for classification).

The remainder of this ReadMe will focus on learning hierarchy-based semantic image embeddings (the first of the papers mentioned above).
If you came here for more information on how we obtained the results reported in the paper about using the cosine loss for classification on small datasets (the second one), you can find those [here][24].


<details><summary><strong>Table of Contents</strong></summary>

1. [What are hierarchy-based semantic image embeddings?](#1-what-are-hierarchy-based-semantic-image-embeddings)
2. [How to learn semantic image embeddings?](#2-how-to-learn-semantic-embeddings)
    1. [Computing target class embeddings](#21-computing-target-class-embeddings)
    2. [Learning image embeddings](#22-learning-image-embeddings)
    3. [Evaluation](#23-evaluation)
    4. [Supported datasets](#24-supported-datasets)
    5. [Available network architectures](#25-available-network-architectures)
    6. [Learning semantic embeddings for ILSVRC and NABirds](#26-learning-semantic-embeddings-for-ilsvrc-and-nabirds)
3. [Requirements](#3-requirements)
4. [Pre-trained models](#4-pre-trained-models)
    1. [Download links](#41-download-links)
    2. [Pre-processing](#42-pre-processing)
    3. [Troubleshooting](#43-troubleshooting)

</details>


## 1. What are hierarchy-based semantic image embeddings?

![Image retrieval results with classification-based features and semantic embeddings on CIFAR-100](https://user-images.githubusercontent.com/7915048/48264069-8c642c00-e428-11e8-99ec-7260023322d0.png)

Features extracted and aggregated from the last convolutional layer of deep neural networks trained for classification have proven to be useful image descriptors for a variety of tasks, e.g., transfer learning and image retrieval.
Regarding content-based image retrieval, it is often claimed that visually similar images are clustered in this feature space.
However, there are two major problems with this approach:

1. ***Visual* similarity does not always correspond to *semantic* similarity.**
   For example, an orange may appear visually similar to an orange bowl, but they are not related at all from a semantic point of view.
   Similarly, if users provide a query image of a palm tree, they are probably more interested in semantically similar images of other trees such as oaks and maples than in images of spiders.
2. The classification objective does not enforce a **high distance between different classes**, so that the nearest neighbors of some images may belong to completely different classes.

Hierarchy-based semantic embeddings overcome these issues by embedding images into a **feature space where the dot product corresponds directly to semantic similarity**.

To this end, the semantic similarity between classes is derived from a class taxonomy specifying is-a relationships between classes.
These pair-wise similarities are then used for explicitly computing optimal target locations for all classes in the feature space.
Finally, a CNN is trained to maximize the correlation between all images and the embedding of their respective class.

![Illustration of semantic embeddings: not visually similar classes are close together but semantically similar ones. Similarities between classes are derived from a given taxonomy.](https://user-images.githubusercontent.com/7915048/48264159-d64d1200-e428-11e8-97ba-ec4b52e9d76b.png)


## 2. How to learn semantic embeddings?

The learning process is divided into two steps:

1. Computing target class embeddings based on a given hierarchy.
2. Training the CNN to map images onto those embeddings.

In the following, we provide a step-by-step example for the CIFAR-100 dataset.

### 2.1. Computing target class embeddings

We derived a class taxonomy for CIFAR-100 from WordNet, but took care that our taxonomy is a tree, which is required for our method.
The hierarchy is encoded in the file [Cifar-Hierarchy/cifar.parent-child.txt](Cifar-Hierarchy/cifar.parent-child.txt) as a set of `parent child` tuples.
For example the line `100 53` specifies that class 53 is a direct child of class 100.
A more human-readable version of the hierarchy can be found in [Cifar-Hierarchy/hierarchy.txt](Cifar-Hierarchy/hierarchy.txt) and can be translated into the encoded version using [Cifar-Hierarchy/encode_hierarchy.py](Cifar-Hierarchy/encode_hierarchy.py)

Given this set of parent-child tuples, target class embeddings can be computed as follows:

```shell
python compute_class_embedding.py \
    --hierarchy Cifar-Hierarchy/cifar.parent-child.txt \
    --out embeddings/cifar100.unitsphere.pickle
```

If your hierarchy contains child-parent instead of parent-child tuples or the class labels are strings instead of integers, you can pass the arguments `--is_a` or `--str_ids`, respectively.

By default, the number of features in the embedding space equals the number of classes. If you would like to have an embedding space with less dimensions that *approximates* the semantic relationships between classes, specify the desired number of feature dimensions with `--num_dim` and also pass `--method approx_sim`.

The result will be a pickle file containing a dictionary with the following items:

- `embedding`: a numpy array whose rows are the embeddings of the classes.
- `ind2label`: a list with the original labels of all classes, in the order corresponding to the rows of `embedding`.
- `label2ind`: a dictionary mapping labels to the corresponding row index in `embedding`.

Hierarchies for the [North American Birds][5] and [ILSVRC 2012][4] datasets can be found in [NAB-Hierarchy/hierarchy.txt](NAB-Hierarchy/hierarchy.txt) and [ILSVRC/wordnet.parent-child.mintree.txt](ILSVRC/wordnet.parent-child.mintree.txt).
The corresponding pre-computed embeddings are stored in `embeddings/nab.unitsphere.pickle` and `embeddings/imagenet_mintree.unitsphere.pickle`, respectively.

For ILSVRC, three different taxonomies are provided:

- [ILSVRC/wordnet.parent-child.txt](ILSVRC/wordnet.parent-child.txt): The complete WordNet taxonomy for all synsets contained in ImageNet.
  This is not a tree! Do not use it directly.
- [ILSVRC/wordnet.parent-child.mintree.txt](ILSVRC/wordnet.parent-child.mintree.txt): A tree derived from the complete taxonomy, starting with all classes with a unique root-path and then adding those paths to other classes which would cause the fewest changes to the tree. This is used for training.
- [ILSVRC/wordnet.parent-child.pruned.txt](ILSVRC/wordnet.parent-child.pruned.txt): The complete WordNet taxonomy for all synsets, but with all children of ILSVRC classes removed. This is not a tree, but is used for evaluation.

### 2.2. Learning image embeddings

After having computed the target class embeddings based on the hierarchy, we can start with training a CNN for mapping the images from the training dataset onto the embeddings of their respective classes.
First, download CIFAR-100 from [here][2] and extract it to some directory.
Then run:

```shell
python learn_image_embeddings.py \
    --dataset CIFAR-100 \
    --data_root /path/to/your/cifar/directory \
    --embedding embeddings/cifar100.unitsphere.pickle \
    --architecture resnet-110-wfc \
    --cls_weight 0.1 \
    --model_dump cifar100-embedding.model.h5 \
    --feature_dump cifar100-features.pickle
```

This will train a variant of ResNet-100 with twice the number of channels per block for 372 epochs using Stochastic Gradient Descent with Warm Restarts (SGDR).
Thus, it is normal to see a drop of performance after epochs 12, 36, 84, and 180, where the restarts happen.
The resulting model will be stored as `cifar100-embedding.model.h5` and pre-computed features for the test dataset will be written to the pickle file `cifar100-features.pickle`.

This method trains a network with a combination of two objectives: an embedding loss and a classification loss.
For training with the embedding loss only, just omit the `--cls_weight` argument.

A set of pre-trained models can be found below.

### 2.3. Evaluation

To evaluate your learned model and features, two scripts are provided.

Several retrieval metrics for the features learned in the previous step can be computed as follows:

```shell
python evaluate_retrieval.py \
    --dataset CIFAR-100 \
    --data_root /path/to/your/cifar/directory \
    --hierarchy Cifar-Hierarchy/cifar.parent-child.txt \
    --classes_from embeddings/cifar100.unitsphere.pickle \
    --feat cifar100-features.pickle \
    --label "Semantic Embeddings"
```

This provides hierarchical precision in terms of two different measures for semantic similarity between classes: Wu-Palmer Similarity (WUP) and the height of the lowest common subsumer (LCSH).
We used the latter in our paper.

If you want to obtain mAHP@250, as in the paper, instead of mAHP over the entire ranking, pass `--clip_ahp 250` in addition.

The classification accuracy can be evaluated as follows:

```shell
python evaluate_classification_accuracy.py \
    --dataset CIFAR-100 \
    --data_root /path/to/your/cifar/directory \
    --hierarchy Cifar-Hierarchy/cifar.parent-child.txt \
    --classes_from embeddings/cifar100.unitsphere.pickle \
    --architecture resnet-100-fc \
    --batch_size 100 \
    --model cifar100-embedding.model.h5 \
    --layer prob \
    --prob_features yes \
    --label "Semantic Embeddings with Classification"
```

If you have learned a semantic embedding model without classification objective, you can perform classification by assigning samples to the nearest class embedding as follows:

```shell
python evaluate_classification_accuracy.py \
    --dataset CIFAR-100 \
    --data_root /path/to/your/cifar/directory \
    --hierarchy Cifar-Hierarchy/cifar.parent-child.txt \
    --classes_from embeddings/cifar100.unitsphere.pickle \
    --architecture resnet-100-fc \
    --batch_size 100 \
    --model cifar100-embedding.model.h5 \
    --layer l2norm \
    --centroids embeddings/cifar100.unitsphere.pickle \
    --label "Semantic Embeddings"
```

### 2.4. Supported datasets

The following values can be specified for `--dataset`:

- **CIFAR-10**: Interface to the [CIFAR-10][3] dataset with 10 classes.
- **CIFAR-100**: Interface to the [CIFAR-100][3] dataset with 100 classes.
- **CIFAR-100-a**: The first 50 classes of [CIFAR-100][3].
- **CIFAR-100-b**: The second 50 classes of [CIFAR-100][3].
- **CIFAR-100-b-consec**: The second 50 classes of [CIFAR-100][3], but numbered from 0-49 instead of 50-99.
- **ILSVRC**: Interface to the [ILSVRC 2012][4] dataset.
- **NAB**: Interface to the [NABirds][5] dataset, expecting images in the sub-directory `images`.
- **NAB-large**: The NABirds dataset with the default image size being twice as large (512 pixels instead of 256, cropped to 448x448).
- **CUB**: Interface to the [Caltech-UCSD Birds][22] dataset, expecting images in the sub-directory `images`. Despite the lack of any suffix, this dataset interface is equivalent to `NAB-large`, just with the CUB data. That means, the input image size is 448x448.
- **Cars**: Interface to the [Stanford Cars][29] dataset with an input image size of 448x448.
- **Flowers**: Interface to the [Oxford Flowers-102][30] dataset with an input image size of 448x448.
- **iNat** / **iNat2018**: Interface to the [iNaturalist 2018][31] dataset with 224x224 crops from images resized to 256 pixels. An underscore followed by the name of a super-category might be appended to restrict the dataset to that supercategory (e.g., "iNat_Aves").
- **iNat-large**: Like iNat but with the default image size being twice as large (512 pixels instead of 256, cropped to 448x448). For restricting the dataset to a certain super-category, append its name before the "-large" suffix (e.g., "iNat_Aves-large").
- **iNat2019** and **iNat2019-large**: Same as above but for the [2019 edition][32] of the iNaturalist challenge. As opposed to the previous edition, this one does not support restricting the dataset to super-classes.

To all datasets except CIFAR, one of the following suffixes may be appended:

- `-ilsvrcmean`: use mean and standard deviation from the ILSVRC dataset for pre-processing.
- `-caffe`: Caffe-style pre-processing (i.e., BGR channel ordering and no normalization of standard deviation).

For ILSVRC, you need to move the test images into sub-directories for each class. [This script][6] could be used for this, for example.

Own dataset interfaces can be defined by creating a new module in the [`datasets`](datasets/) package, defining a class derived from [`FileDatasetGenerator`](datasets/common.py), importing it in [`datasets/__init__.py`](datasets/__init__.py), and adding a branch for it in the `get_data_generator` function defined there.

### 2.5. Available network architectures

#### 2.5.1. Tested

For CIFAR:

- **simple**: The [Plain-11][7] architecture, a strictly sequential, shallow CNN with 11 trainable layers. Good for conducting quick experiments.
  For training this network architecture, we used `--max_decay 0.1` in addition to the other arguments provided above for the ResNet.
  This causes a continuous decay of the learning rate so that the final learning rate will be 10 times less than the initial one.
- **resnet-110**: The standard [ResNet-110][8].
- **resnet-110-fc**: Like `resnet-110`, but always with a fully-connected layer after the global average pooling, even when used for learning embeddings.
- **resnet-110-wfc**: A variant of `resnet-110-fc` with twice the number of channels per block.
- **wrn-28-10**: A [Wide Residual Network][9] with depth 28 and width 10.
- **pyramidnet-272-200**: A [Deep Pyramidal Residual Network][10]. Provides better performance than ResNet, but is also much slower.

For ImageNet and NABirds:

- **resnet-50**: The standard [ResNet-50][8] implementation from `keras-applications`.

#### 2.5.2. Experimental

For CIFAR:

- **resnet-32**: The standard [ResNet-32][8].
- **densenet-100-12**: A [Densely Connected Convolutional Network][12] with depth 100 and growth-rate 12.
- **densenet-100-24**: A [Densely Connected Convolutional Network][12] with depth 100 and growth-rate 24.
- **densenet-bc-190-40**: A [Densely Connected Convolutional Network][12] with bottleneck blocks and compression (depth 190 and growth-rate 40).

For ImageNet and NABirds:

- **resnet-101** and **resnet-152**: The standard implementaions of [ResNet-101 and ResNet-152][8] introduced in `keras-applications >= 1.0.7`.
- **rn18**, **rn34**, **rn50**, **rn101**, **rn152**, **rn200**: [ResNets][8] with different depths.
  Requires the [keras-resnet][13] package, but be aware that batch normalization is broken in version 0.1.0.
  Thus, you need to either use an earlier or later version or merge [this pull request][14].
- **nasnet-a**: The [NasNet-A][15] implementation from `keras-applications`.


### 2.6. Learning semantic embeddings for ILSVRC and NABirds

The previous sections have shown in detail how to learn semantic image embeddings for CIFAR-100.
In the following, we provide the calls to [learn_image_embeddings.py](learn_image_embeddings.py) that we used to train our semantic embedding models (including classification objective) on the [ILSVRC 2012][4] and [NABirds][5] datasets.

```shell
# ILSVRC
python learn_image_embeddings.py \
    --dataset ILSVRC \
    --data_root /path/to/imagenet/ \
    --embedding embeddings/imagenet_mintree.unitsphere.pickle \
    --architecture resnet-50 \
    --loss inv_corr \
    --cls_weight 0.1 \
    --lr_schedule SGDR \
    --sgdr_base_len 80 \
    --epochs 80 \
    --max_decay 0 \
    --batch_size 128 \
    --gpus 2 \
    --model_dump imagenet_unitsphere-embed+cls_rn50.model.h5

# NAB (from scratch)
python learn_image_embeddings.py \
    --dataset NAB \
    --data_root /path/to/nab/ \
    --embedding embeddings/nab.unitsphere.pickle \
    --architecture resnet-50 \
    --loss inv_corr \
    --cls_weight 0.1 \
    --lr_schedule SGDR \
    --sgdr_max_lr 0.5 \
    --max_decay 0 \
    --epochs 180 \
    --batch_size 128 \
    --gpus 2 \
    --read_workers 10 \
    --queue_size 20 \
    --model_dump nab_unitsphere-embed+cls_rn50.model.h5

# NAB (fine-tuned)
python learn_image_embeddings.py \
    --dataset NAB-ilsvrcmean \
    --data_root /path/to/nab/ \
    --embedding embeddings/nab.unitsphere.pickle \
    --architecture resnet-50 \
    --loss inv_corr \
    --cls_weight 0.1 \
    --finetune imagenet_unitsphere-embed+cls_rn50.model.h5 \
    --finetune_init 8 \
    --lr_schedule SGDR \
    --sgd_lr 0.1 \
    --sgdr_max_lr 0.5 \
    --max_decay 0 \
    --epochs 180 \
    --batch_size 128 \
    --gpus 2 \
    --read_workers 10 \
    --queue_size 20 \
    --model_dump nab_unitsphere-embed+cls_rn50_finetuned.model.h5
```

## 3. Requirements

- Python >= 3.5
- numpy
- numexpr
- keras >= 2.2.0
- tensorflow (we used v1.8)
- sklearn
- scipy
- pillow
- matplotlib


## 4. Pre-trained models

### 4.1. Download links

|  Dataset  |              Model              | Input Size | mAHP@250 | Balanced Accuracy |
|-----------|---------------------------------|:----------:|---------:|------------------:|
| CIFAR-100 | [Plain-11][16]                  |    32x32   |   82.05% |            74.10% |
| CIFAR-100 | [ResNet-110-wfc][17]            |    32x32   |   83.29% |            76.60% |
| CIFAR-100 | [PyramidNet-272-200][18]        |    32x32   |   86.38% |            80.49% |
| NABirds   | [ResNet-50 (from scratch)][19]  |   224x224  |   73.99% |            59.46% |
| NABirds   | [ResNet-50 (fine-tuned)][20]    |   224x224  |   81.46% |            69.49% |
| NABirds   | [ResNet-50 (from scratch)][27]  |   448x448  |   82.33% |            70.43% |
| NABirds   | [ResNet-50 (fine-tuned)][28]    |   448x448  |   88.11% |            76.79% |
| CUB       | [ResNet-50 (from scratch)][25]  |   448x448  |   83.33% |            70.14% |
| CUB       | [ResNet-50 (fine-tuned)][26]    |   448x448  |   92.24% |            80.23% |
| ILSVRC    | [ResNet-50][21] *               |   224x224  |   83.15% |            70.42% |

<p style="font-size: 0.8em">
* This is an updated model with slightly better performance than reported in the paper (~1 percent point).
The original model can be obtained <a href="https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/imagenet_unitsphere-embed+cls_rn50.model.h5">here</a>.
</p>

### 4.2. Pre-processing

The pre-trained models provided above assume input images to be given in RGB color format and standardized by subtracting a dataset-specific channel-wise mean and dividing by a dataset-specific standard deviation.
The means and standard deviations for each dataset are provided in the following table.

|            Dataset          |                     Mean                     |            Standard Deviation            |
|-----------------------------|----------------------------------------------|------------------------------------------|
| CIFAR-100                   | `[129.30386353, 124.06987, 112.43356323]`    | `[68.17019653, 65.39176178, 70.4180603]` |
| NABirds (from scratch)      | `[125.30513277, 129.66606421, 118.45121113]` | `[57.0045467, 56.70059436, 68.44430446]` |
| CUB (from scratch)          | `[123.82988033, 127.35116805, 110.25606303]` | `[59.2230949, 58.0736071, 67.80251684]`  |
| ILSVRC & fine-tuned models  | `[122.65435242, 116.6545058, 103.99789959]`  | `[71.40583196, 69.56888997, 73.0440314]` |

### 4.3. Troubleshooting

Sometimes, loading of the pre-trained models fails with the error message "unknown opcode".
In the case of this or other issues, you can still create the architecture yourself and load the pre-trained weights from the model files provided above.
For CIFAR-100 and the `resnet-110-wfc` architecture, for example, this can be done as follows:

```python
import keras
import utils
from learn_image_embeddings import cls_model

model = utils.build_network(100, 'resnet-110-wfc')
model = keras.models.Model(
    model.inputs,
    keras.layers.Lambda(utils.l2norm, name = 'l2norm')(model.output)
)
model = cls_model(model, 100)

model.load_weights('cifar_unitsphere-embed+cls_resnet-110-wfc.model.h5')
```


[1]: https://arxiv.org/pdf/1809.09924
[2]: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
[3]: https://www.cs.toronto.edu/~kriz/cifar.html
[4]: http://image-net.org/challenges/LSVRC/2012/
[5]: http://dl.allaboutbirds.org/nabirds
[6]: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
[7]: http://hera.inf-cv.uni-jena.de:6680/pdf/Barz18:GoodTraining
[8]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
[9]: https://arxiv.org/pdf/1605.07146.pdf
[10]: https://ieeexplore.ieee.org/abstract/document/8100151
[11]: https://arxiv.org/pdf/1409.1556.pdf
[12]: http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
[13]: https://github.com/broadinstitute/keras-resnet
[14]: https://github.com/broadinstitute/keras-resnet/pull/47
[15]: https://arxiv.org/pdf/1707.07012.pdf
[16]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/cifar_unitsphere-embed+cls_plain11.model.h5
[17]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/cifar_unitsphere-embed+cls_resnet-110-fc.model.h5
[18]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/cifar_unitsphere-embed+cls_pyramidnet-272-200.model.h5
[19]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/nab_unitsphere-embed+cls_rn50.model.h5
[20]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/nab_unitsphere-embed+cls_rn50_finetuned.model.h5
[21]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.1.0/imagenet_unitsphere-embed+cls_rn50.model.h5
[22]: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
[23]: https://arxiv.org/pdf/1901.09054
[24]: CosineLoss.md
[25]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.1.0/cub_unitsphere-embed+cls_deep-hierarchy_rn50.model.h5
[26]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.1.0/cub_unitsphere-embed+cls_deep-hierarchy_rn50_finetuned.model.h5
[27]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.1.0/nab-large_unitsphere-embed+cls_rn50.model.h5
[28]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.1.0/nab-large_unitsphere-embed+cls_rn50_finetuned.model.h5
[29]: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
[30]: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
[31]: https://github.com/visipedia/inat_comp/tree/2018
[32]: https://www.kaggle.com/c/inaturalist-2019-fgvc6/

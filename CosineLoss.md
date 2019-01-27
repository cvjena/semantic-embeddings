# Deep Learning on Small Datasets without Pre-Training using Cosine Loss

This document explains how the code in this repository can be used to produce the results reported in the following paper:

> [**Deep Learning on Small Datasets without Pre-Training using Cosine Loss.**][1]  
> Björn Barz and Joachim Denzler.  


## 1. Results

According to Table 2 in the paper:

|          Loss Function          |    CUB    |    NAB    |   Cars    |  Flowers  | CIFAR-100 |
|---------------------------------|----------:|----------:|----------:|----------:|----------:|
| cross entropy                   |   51.9%   |   59.4%   |   78.2%   |   67.3%   |   77.0%   |
| cross entropy + label smoothing |   55.9%   |   68.3%   |   78.1%   |   66.8%   | **77.5%** |
| cosine loss                     |   67.6%   |   71.7%   |   84.3%   | **71.1%** |   75.3%   |
| cosine loss + cross entropy     | **68.0%** | **71.9%** | **85.0%** |   70.6%   |   76.4%   |


## 2. Requirements

- Python >= 3.5
- numpy
- numexpr
- keras >= 2.2.0
- tensorflow (we used v1.8)
- sklearn
- scipy
- pillow


## 3. Datasets

The following datasets have been used in the paper:

- [Caltech UCSD Birds-200-2011][4] (CUB)
- [North American Birds][3] (NAB-large)
- [Stanford Cars][5] (Cars)
- [Oxford Flowers-102][6] (Flowers)
- [CIFAR-100][2] (CIFAR-100)

The names in parentheses specify the dataset names that can be passed to the scripts mentioned below.


## 4. Training with different loss functions

In the following exemplary python script calls, replace `$DS` with the name of the dataset (see above),
`$DSROOT` with the path to that dataset, and `$LR` with the maximum learning rate for SGDR.

To save the model after training has completed, add `--model_dump` followed by the filename where the model definition and weights should be written to.

### 4.1 Softmax + Cross Entropy

```shell
python learn_classifier.py \
    --dataset $DS --data_root $DSROOT --sgdr_max_lr $LR \
    --architecture resnet-50 --batch_size 96 \
    --gpus 4 --read_workers 16 --queue_size 32 --gpu_merge
```

For label smoothing, add `--label_smoothing 0.1`.

### 4.2 Cosine Loss

```shell
python learn_image_embeddings.py \
    --dataset $DS --data_root $DSROOT --sgdr_max_lr $LR \
    --embedding onehot --architecture resnet-50 --batch_size 96 \
    --gpus 4 --read_workers 16 --queue_size 32 --gpu_merge
```

For the combined cosine + cross-entropy loss, add `--cls_weight 0.1`.

To use semantic embeddings instead of one-hot vectors, pass a path to one of the embedding files in the [`embeddings`](embeddings/) directory to `--embedding` instead of `onehot`.

### 4.3 CIFAR-100

For the CIFAR-100 dataset, use the following parameters:

```shell
python learn_classifier.py \
    --dataset CIFAR-100 --data_root $DSROOT --sgdr_max_lr $LR \
    --architecture resnet-110-wfc --batch_size 100

python learn_image_embeddings.py \
    --dataset CIFAR-100 --data_root $DSROOT --sgdr_max_lr $LR \
    --embedding onehot --architecture resnet-110-wfc --batch_size 100
```

### 4.4 Determining the best performance across different learning rates

For each dataset and loss function, we fine-tuned the learning rate individually by wrapping the training script calls into a bash loop like the following (here shown for training with the cosine loss on CIFAR-100 as an example):

```shell
for LR in 2.5 1.0 0.5 0.1 0.05 0.01 0.005 0.001; do
    echo $LR
    python learn_image_embeddings.py \
        --dataset CIFAR-100 --data_root $DSROOT --sgdr_max_lr $LR \
        --embedding onehot --architecture resnet-110-wfc --batch_size 100 \
        2>/dev/null | grep -oP "val_(prob_)?acc: \K([0-9.]+)" | sort -n | tail -n 1
done
```

The following table lists the values for `--sgdr_max_lr` that led to the best results.

|                  Loss                  |  CUB  |  NAB  | Cars | Flowers | CIFAR-100 |
|----------------------------------------|------:|------:|-----:|--------:|----------:|
| cross entropy                          |  0.05 |  0.05 |  1.0 |     1.0 |       0.1 |
| cross entropy + label smoothing        |  0.05 |   0.1 |  1.0 |     0.1 |       0.1 |
| cosine loss (one-hot)                  |   0.5 |   0.5 |  1.0 |     0.5 |      0.05 |
| cosine loss + cross entropy (one-hot)  |   0.5 |   0.5 |  0.5 |     0.5 |       0.1 |


## 5. Sub-sampling CUB

To experiment with differently sized variants of the CUB dataset, copy the image list files from the directory [`CUB-splits`](CUB-splits/) into the root directory of your CUB dataset and specify the dataset name as `CUB-subX`, where `X` is the number of samples per class.

![Performance comparison for differently sub-sampled variants of the CUB dataset](https://user-images.githubusercontent.com/7915048/51765373-d67bb600-20d7-11e9-85a9-ec6f28cef39b.png)



[1]: https://arxiv.org/pdf/1901.09054
[2]: https://www.cs.toronto.edu/~kriz/cifar.html
[3]: http://dl.allaboutbirds.org/nabirds
[4]: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
[5]: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
[6]: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

>

This is a modified version of Suprised adequacy code to work with new models and datasets.

Code release of a paper ["Guiding Deep Learning System Testing using Surprise Adequacy"](https://arxiv.org/abs/1808.08444)
Code from (https://github.com/coinse/sadl).
```

## Introduction

This folder includes code for computing Surprise Adequacy (SA) and Surprise Coverage (LSC and DSC).


### Files and Directories

- `run.py` - Script processing SA with a benign dataset (MNIST and CIFAR-10).
- `sa.py` - Tools that fetch activation traces, compute LSC and DSC.
- `train_model.py` - Model training script for MNIST , CIFAR-10 and other datasets. It keeps the trained models in the "model" directory (code from [Ma et al.](https://github.com/xingjunm/lid_adversarial_subspace_detection)).
- `model` directory - Used for saving models.
- `tmp` directory - Used for saving activation traces and prediction arrays.

### Command-line Options of run.py

- `-d` - The subject dataset (either mnist or cifar). Default is mnist.
- `-lsa` - If set, computes LSA.
- `-dsa` - If set, computes DSA.=
- `-save_path` - The temporal save path of AT files. Default is tmp directory.
- `-batch_size` - Batch size. Default is 128.
- `-var_threshold` - Variance threshold. Default is 1e-5.
- `-upper_bound` - Upper bound of SA. Default is 2000.
- `-n_bucket` - The number of buckets for coverage. Default is 1000.
- `-num_classes` - The number of classes in dataset. Default is 10.
- `-is_classification` - Set if task is classification problem. Default is True.

## References
- [Suprise Adequacy](https://github.com/coinse/sadl)
- [DeepXplore](https://github.com/peikexin9/deepxplore)
- [DeepTest](https://github.com/ARiSE-Lab/deepTest)
- [Detecting Adversarial Samples from Artifacts](https://github.com/rfeinman/detecting-adversarial-samples)
- [Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality](https://github.com/xingjunm/lid_adversarial_subspace_detection)

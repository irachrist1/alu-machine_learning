# Optimization

Hyperparameter tuning and optimization techniques for neural networks using
NumPy and TensorFlow 1.12 on MNIST and binary classification datasets.

## Requirements

- Ubuntu 16.04 LTS, Python 3.5
- numpy 1.15, tensorflow 1.12
- pycodestyle 2.4
- Only `import numpy as np` and/or `import tensorflow as tf` (no keras)

## Files

| File | Description |
| --- | --- |
| `0-norm_constants.py` | Calculate feature mean and standard deviation. |
| `1-normalize.py` | Standardize a matrix using mean and std. |
| `2-shuffle_data.py` | Shuffle training data consistently. |
| `3-mini_batch.py` | Train a loaded model with mini-batch gradient descent. |
| `4-moving_average.py` | Exponentially weighted moving average with bias correction. |
| `5-momentum.py` | Gradient descent with momentum (NumPy). |
| `6-momentum.py` | Momentum optimizer training op (TensorFlow). |
| `7-RMSProp.py` | RMSProp update rule (NumPy). |
| `8-RMSProp.py` | RMSProp optimizer training op (TensorFlow). |
| `9-Adam.py` | Adam update rule (NumPy). |
| `10-Adam.py` | Adam optimizer training op (TensorFlow). |
| `11-learning_rate_decay.py` | Inverse time decay learning rate (NumPy). |
| `12-learning_rate_decay.py` | Inverse time decay learning rate (TensorFlow). |
| `13-batch_norm.py` | Batch normalization (NumPy). |
| `14-batch_norm.py` | Batch normalization layer (TensorFlow). |
| `15-model.py` | Full model with Adam, mini-batch, LR decay, and batch norm. |

## Blog Post (Task 16)

Add URLs here after publishing on Medium or LinkedIn.

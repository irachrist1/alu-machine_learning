# Autoencoders

Unsupervised learning project on autoencoders using TensorFlow 1.12 / Keras on
the MNIST dataset.

## Requirements

- Ubuntu 16.04 LTS, Python 3.5
- numpy 1.15, tensorflow 1.12
- pycodestyle 2.4
- Only `import tensorflow.keras as keras` is allowed in task modules

## Files

| File | Description |
| --- | --- |
| `0-vanilla.py` | Vanilla fully-connected autoencoder. |
| `1-sparse.py` | Sparse autoencoder with L1 activity regularization. |
| `2-convolutional.py` | Convolutional autoencoder with pooling and upsampling. |
| `3-variational.py` | Variational autoencoder with KL divergence loss. |

## Usage

Each task is tested with its corresponding main script, for example:

```bash
./0-main.py
```

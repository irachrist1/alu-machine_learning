# Tensorflow

Introductory project on building, training, saving, and restoring neural
networks with TensorFlow 1.12 on the MNIST dataset.

## Requirements

- Ubuntu 16.04 LTS, Python 3.5
- numpy 1.15, tensorflow 1.12
- pycodestyle 2.4
- Only `import tensorflow as tf` is allowed (no `keras`)

## Files

| File | Description |
| --- | --- |
| `0-create_placeholders.py` | Create input/label placeholders `x` and `y`. |
| `1-create_layer.py` | Create a `tf.layers.Dense` layer with He initialization. |
| `2-forward_prop.py` | Build the forward propagation graph. |
| `3-calculate_accuracy.py` | Compute decimal accuracy from predictions. |
| `4-calculate_loss.py` | Compute the softmax cross-entropy loss. |
| `5-create_train_op.py` | Create the gradient descent training operation. |
| `6-train.py` | Build, train, and save a classifier; add tensors to the graph collection. |
| `7-evaluate.py` | Restore a saved model and evaluate it on new data. |

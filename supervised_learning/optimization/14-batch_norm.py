#!/usr/bin/env python3
"""TensorFlow batch normalization layer module."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Create a batch normalization layer for a neural network.

    Args:
        prev: activated output of the previous layer.
        n: number of nodes in the layer to create.
        activation: activation function for the layer output.

    Returns:
        A tensor of the activated output for the layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    output = layer(prev)
    mean, variance = tf.nn.moments(output, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
    normalized = tf.nn.batch_normalization(
        output, mean, variance, beta, gamma, 1e-8)
    return activation(normalized)

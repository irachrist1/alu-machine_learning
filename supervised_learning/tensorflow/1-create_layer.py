#!/usr/bin/env python3
"""Create a single dense layer for a neural network."""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Create a dense layer with He et. al initialization.

    Args:
        prev: the tensor output of the previous layer.
        n: the number of nodes in the new layer.
        activation: the activation function for the layer.

    Returns:
        The tensor output of the new layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name="layer")
    return layer(prev)

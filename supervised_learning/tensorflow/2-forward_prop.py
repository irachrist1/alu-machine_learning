#!/usr/bin/env python3
"""Build the forward propagation graph of the neural network."""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Create the forward propagation graph for the neural network.

    Args:
        x: the placeholder for the input data.
        layer_sizes: list with the number of nodes in each layer.
        activations: list with the activation function for each layer.

    Returns:
        The tensor prediction of the network.
    """
    prediction = x
    for i in range(len(layer_sizes)):
        prediction = create_layer(prediction, layer_sizes[i], activations[i])
    return prediction

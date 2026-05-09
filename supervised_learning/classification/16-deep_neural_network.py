#!/usr/bin/env python3
"""Defines a DeepNeuralNetwork class with public attributes."""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize deep neural network with He weight initialization.

        Args:
            nx (int): number of input features.
            layers (list): number of nodes in each layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prev = nx
        for lay in range(self.L):
            if type(layers[lay]) is not int or layers[lay] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights['W{}'.format(lay + 1)] = (
                np.random.randn(layers[lay], prev) * np.sqrt(2 / prev))
            self.weights['b{}'.format(lay + 1)] = np.zeros((layers[lay], 1))
            prev = layers[lay]

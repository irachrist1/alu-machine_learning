#!/usr/bin/env python3
"""Defines a DeepNeuralNetwork class with private attributes."""
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
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for lay, nodes in enumerate(layers, 1):
            self.__weights['W{}'.format(lay)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev))
            self.__weights['b{}'.format(lay)] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        """Getter for number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for intermediary values cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights and biases dictionary."""
        return self.__weights

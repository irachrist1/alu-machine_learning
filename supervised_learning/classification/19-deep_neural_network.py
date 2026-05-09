#!/usr/bin/env python3
"""Defines a DeepNeuralNetwork class with cost calculation."""
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

    def forward_prop(self, X):
        """Calculate forward propagation of the deep neural network.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).

        Returns:
            tuple: output of last layer and cache dictionary.
        """
        self.__cache['A0'] = X
        A = X
        for lay in range(1, self.__L + 1):
            W = self.__weights['W{}'.format(lay)]
            b = self.__weights['b{}'.format(lay)]
            z = np.dot(W, A) + b
            A = 1 / (1 + np.exp(-z))
            self.__cache['A{}'.format(lay)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculate logistic regression cost.

        Args:
            Y (numpy.ndarray): correct labels, shape (1, m).
            A (numpy.ndarray): activated output, shape (1, m).

        Returns:
            float: cost of the model.
        """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(
            1.0000001 - A)) / m

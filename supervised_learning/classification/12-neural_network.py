#!/usr/bin/env python3
"""Defines a NeuralNetwork class with evaluation."""
import numpy as np


class NeuralNetwork:
    """Neural network with one hidden layer for binary classification."""

    def __init__(self, nx, nodes):
        """Initialize neural network.

        Args:
            nx (int): number of input features.
            nodes (int): number of nodes in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for hidden layer weights."""
        return self.__W1

    @property
    def b1(self):
        """Getter for hidden layer bias."""
        return self.__b1

    @property
    def A1(self):
        """Getter for hidden layer activated output."""
        return self.__A1

    @property
    def W2(self):
        """Getter for output neuron weights."""
        return self.__W2

    @property
    def b2(self):
        """Getter for output neuron bias."""
        return self.__b2

    @property
    def A2(self):
        """Getter for output neuron activated output."""
        return self.__A2

    def forward_prop(self, X):
        """Calculate forward propagation of the neural network.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).

        Returns:
            tuple: (__A1, __A2) activated outputs of both layers.
        """
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

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

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels, shape (1, m).

        Returns:
            tuple: predicted labels (1, m) and cost.
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        return (A2 >= 0.5).astype(int), cost

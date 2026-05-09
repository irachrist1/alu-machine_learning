#!/usr/bin/env python3
"""Defines a Neuron class with gradient descent."""
import numpy as np


class Neuron:
    """Single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize neuron with nx input features.

        Args:
            nx (int): number of input features.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for bias."""
        return self.__b

    @property
    def A(self):
        """Getter for activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculate forward propagation using sigmoid activation.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).

        Returns:
            numpy.ndarray: activated output __A.
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

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
        """Evaluate the neuron's predictions.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels, shape (1, m).

        Returns:
            tuple: predicted labels (1, m) and cost.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (A >= 0.5).astype(int), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform one pass of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels, shape (1, m).
            A (numpy.ndarray): activated output, shape (1, m).
            alpha (float): learning rate.
        """
        m = Y.shape[1]
        dz = A - Y
        self.__W -= alpha * np.dot(dz, X.T) / m
        self.__b -= alpha * np.sum(dz) / m

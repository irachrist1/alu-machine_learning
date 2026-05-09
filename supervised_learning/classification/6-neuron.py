#!/usr/bin/env python3
"""Neuron with training loop."""
import numpy as np


class Neuron:
    """Single neuron for binary classification."""

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (A >= 0.5).astype(int), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        m = Y.shape[1]
        dz = A - Y
        self.__W -= alpha * np.dot(dz, X.T) / m
        self.__b -= alpha * np.sum(dz) / m

    def train(self, X, Y, iterations=5000, alpha=0.05):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)

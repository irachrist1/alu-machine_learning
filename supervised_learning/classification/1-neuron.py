#!/usr/bin/env python3
"""Defines a Neuron class with private attributes."""
import numpy as np


class Neuron:
    """Single neuron performing binary classification with private attrs."""

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

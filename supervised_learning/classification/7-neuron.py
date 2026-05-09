#!/usr/bin/env python3
"""Defines a Neuron class with verbose and graph training."""
import matplotlib.pyplot as plt
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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Train the neuron with optional verbose output and cost graph.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels, shape (1, m).
            iterations (int): number of training iterations.
            alpha (float): learning rate.
            verbose (bool): whether to print cost at each step.
            graph (bool): whether to plot cost after training.
            step (int): interval for verbose/graph output.

        Returns:
            tuple: evaluation of training data after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        A = self.forward_prop(X)
        if verbose:
            print("Cost after 0 iterations: {}".format(self.cost(Y, A)))
        if graph:
            costs.append(self.cost(Y, A))
            iters.append(0)

        for i in range(1, iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            if (verbose or graph) and (i % step == 0 or i == iterations):
                cost = self.cost(Y, self.__A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
                    iters.append(i)

        if graph:
            plt.plot(iters, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

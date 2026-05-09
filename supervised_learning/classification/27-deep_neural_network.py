#!/usr/bin/env python3
"""Defines a DeepNeuralNetwork class for multiclass classification."""
import pickle
import matplotlib.pyplot as plt
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network for multiclass classification with softmax."""

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
        """Calculate forward propagation using softmax on output layer.

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
            if lay == self.__L:
                e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
                A = e_z / np.sum(e_z, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-z))
            self.__cache['A{}'.format(lay)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculate categorical cross-entropy cost.

        Args:
            Y (numpy.ndarray): one-hot labels, shape (classes, m).
            A (numpy.ndarray): softmax output, shape (classes, m).

        Returns:
            float: cost of the model.
        """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): one-hot labels, shape (classes, m).

        Returns:
            tuple: one-hot predictions (classes, m) and cost.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        return pred.astype(int), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform one pass of gradient descent on the deep neural network.

        Args:
            Y (numpy.ndarray): one-hot labels, shape (classes, m).
            cache (dict): intermediary values from forward propagation.
            alpha (float): learning rate.
        """
        m = Y.shape[1]
        dz = cache['A{}'.format(self.__L)] - Y
        for lay in range(self.__L, 0, -1):
            A_prev = cache['A{}'.format(lay - 1)]
            W = self.__weights['W{}'.format(lay)]
            dW = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            if lay > 1:
                A_lay = cache['A{}'.format(lay - 1)]
                dz = np.dot(W.T, dz) * A_lay * (1 - A_lay)
            self.__weights['W{}'.format(lay)] -= alpha * dW
            self.__weights['b{}'.format(lay)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Train the deep neural network with optional verbose and graph.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): one-hot labels, shape (classes, m).
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

        A, cache = self.forward_prop(X)
        if verbose:
            print("Cost after 0 iterations: {}".format(self.cost(Y, A)))
        if graph:
            costs.append(self.cost(Y, A))
            iters.append(0)

        for i in range(1, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if (verbose or graph) and (i % step == 0 or i == iterations):
                cost = self.cost(Y, cache['A{}'.format(self.__L)])
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

    def save(self, filename):
        """Save the instance object to a file in pickle format.

        Args:
            filename (str): path to save the object; .pkl added if missing.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork object from file.

        Args:
            filename (str): path to the pickle file.

        Returns:
            DeepNeuralNetwork: loaded object, or None if file not found.
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

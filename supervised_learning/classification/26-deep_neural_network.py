#!/usr/bin/env python3
"""Deep neural network with save/load."""
import matplotlib.pyplot as plt
import numpy as np
import pickle


class DeepNeuralNetwork:
    """Deep neural network for binary classification."""

    def __init__(self, nx, layers):
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
        for l, nodes in enumerate(layers, 1):
            self.__weights['W{}'.format(l)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev))
            self.__weights['b{}'.format(l)] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        self.__cache['A0'] = X
        A = X
        for l in range(1, self.__L + 1):
            W = self.__weights['W{}'.format(l)]
            b = self.__weights['b{}'.format(l)]
            z = np.dot(W, A) + b
            A = 1 / (1 + np.exp(-z))
            self.__cache['A{}'.format(l)] = A
        return A, self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (A >= 0.5).astype(int), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        dz = cache['A{}'.format(self.__L)] - Y
        for l in range(self.__L, 0, -1):
            A_prev = cache['A{}'.format(l - 1)]
            W = self.__weights['W{}'.format(l)]
            dW = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            if l > 1:
                A_l = cache['A{}'.format(l - 1)]
                dz = np.dot(W.T, dz) * A_l * (1 - A_l)
            self.__weights['W{}'.format(l)] -= alpha * dW
            self.__weights['b{}'.format(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
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
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

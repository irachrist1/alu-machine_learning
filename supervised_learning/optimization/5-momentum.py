#!/usr/bin/env python3
"""Gradient descent with momentum module."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using gradient descent with momentum.

    Args:
        alpha: learning rate.
        beta1: momentum weight.
        var: numpy.ndarray containing the variable to update.
        grad: numpy.ndarray containing the gradient of var.
        v: previous first moment of var.

    Returns:
        The updated variable and the new moment, respectively.
    """
    v = beta1 * v + (1 - beta1) * grad
    return var - alpha * v, v

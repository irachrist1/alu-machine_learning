#!/usr/bin/env python3
"""RMSProp optimization module."""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Update a variable using the RMSProp optimization algorithm.

    Args:
        alpha: learning rate.
        beta2: RMSProp weight.
        epsilon: small number to avoid division by zero.
        var: numpy.ndarray containing the variable to update.
        grad: numpy.ndarray containing the gradient of var.
        s: previous second moment of var.

    Returns:
        The updated variable and the new moment, respectively.
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (s ** 0.5 + epsilon)
    return var, s

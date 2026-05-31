#!/usr/bin/env python3
"""Adam optimization module."""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update a variable using the Adam optimization algorithm.

    Args:
        alpha: learning rate.
        beta1: weight used for the first moment.
        beta2: weight used for the second moment.
        epsilon: small number to avoid division by zero.
        var: numpy.ndarray containing the variable to update.
        grad: numpy.ndarray containing the gradient of var.
        v: previous first moment of var.
        s: previous second moment of var.
        t: time step used for bias correction.

    Returns:
        The updated variable, new first moment, and new second moment.
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)
    var = var - alpha * v_corrected / (s_corrected ** 0.5 + epsilon)
    return var, v, s

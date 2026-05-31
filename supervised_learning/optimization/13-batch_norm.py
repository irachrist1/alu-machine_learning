#!/usr/bin/env python3
"""Batch normalization module."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalize an unactivated output using batch normalization.

    Args:
        Z: numpy.ndarray of shape (m, n) to normalize.
        gamma: numpy.ndarray of shape (1, n) with scale parameters.
        beta: numpy.ndarray of shape (1, n) with offset parameters.
        epsilon: small number used to avoid division by zero.

    Returns:
        The normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    normalized = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * normalized + beta

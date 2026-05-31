#!/usr/bin/env python3
"""Normalization constants module."""
import numpy as np


def normalization_constants(X):
    """Calculate the normalization constants of a matrix.

    Args:
        X: numpy.ndarray of shape (m, nx) to normalize.

    Returns:
        The mean and standard deviation of each feature, respectively.
    """
    return np.mean(X, axis=0), np.std(X, axis=0)

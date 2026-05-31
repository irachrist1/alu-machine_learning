#!/usr/bin/env python3
"""Data shuffling module."""
import numpy as np


def shuffle_data(X, Y):
    """Shuffle two matrices in the same order.

    Args:
        X: numpy.ndarray of shape (m, nx) to shuffle.
        Y: numpy.ndarray of shape (m, ny) to shuffle.

    Returns:
        The shuffled X and Y matrices.
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]

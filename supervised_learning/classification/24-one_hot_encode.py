#!/usr/bin/env python3
"""Defines a one-hot encoding function."""
import numpy as np


def one_hot_encode(Y, classes):
    """Convert a numeric label vector into a one-hot matrix.

    Args:
        Y (numpy.ndarray): numeric class labels, shape (m,).
        classes (int): maximum number of classes found in Y.

    Returns:
        numpy.ndarray: one-hot encoding of shape (classes, m), or None.
    """
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None

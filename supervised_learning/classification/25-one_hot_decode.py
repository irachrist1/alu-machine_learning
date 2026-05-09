#!/usr/bin/env python3
"""Defines a one-hot decoding function."""
import numpy as np


def one_hot_decode(one_hot):
    """Convert a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray): one-hot encoded array, shape (classes, m).

    Returns:
        numpy.ndarray: numeric labels of shape (m,), or None on failure.
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None

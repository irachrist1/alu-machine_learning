#!/usr/bin/env python3
"""Feature normalization module."""


def normalize(X, m, s):
    """Normalize (standardize) a matrix.

    Args:
        X: numpy.ndarray of shape (d, nx) to normalize.
        m: numpy.ndarray of shape (nx,) with feature means.
        s: numpy.ndarray of shape (nx,) with feature standard deviations.

    Returns:
        The normalized X matrix.
    """
    return (X - m) / s

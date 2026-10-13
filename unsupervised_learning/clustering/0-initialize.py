#!/usr/bin/env python3
"""Initialize K-means centroids within the bounds of a dataset."""
import numpy as np


def initialize(X, k):
    """Return k uniformly sampled centroids, or None for invalid input."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            X.dtype.kind not in 'biuf' or not np.all(np.isfinite(X)) or
            type(k) is not int or k <= 0):
        return None
    return np.random.uniform(X.min(axis=0), X.max(axis=0),
                             size=(k, X.shape[1]))

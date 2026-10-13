#!/usr/bin/env python3
"""Measure total squared distance to the nearest cluster centroid."""
import numpy as np


def variance(X, C):
    """Return the total intra-cluster variance, or None on failure."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            X.dtype.kind not in 'biuf' or not np.all(np.isfinite(X)) or
            not isinstance(C, np.ndarray) or C.ndim != 2 or
            C.shape[0] == 0 or C.shape[1] != X.shape[1] or
            C.dtype.kind not in 'biuf' or not np.all(np.isfinite(C))):
        return None
    distances = np.sum((X.astype(float)[:, None, :] - C) ** 2, axis=2)
    return np.sum(np.min(distances, axis=1))

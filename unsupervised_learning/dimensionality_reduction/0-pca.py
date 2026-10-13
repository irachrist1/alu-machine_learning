#!/usr/bin/env python3
"""Find principal component weights for centered data."""
import numpy as np


def pca(X, var=0.95):
    """Return the leading weights retaining the requested SVD fraction."""
    _, singular, vectors = np.linalg.svd(X, full_matrices=False)
    cumulative = np.cumsum(singular) / np.sum(singular)
    dimensions = np.searchsorted(cumulative, var) + 1
    return vectors[:dimensions].T

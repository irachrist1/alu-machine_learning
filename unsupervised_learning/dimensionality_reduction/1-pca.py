#!/usr/bin/env python3
"""Project data into a fixed number of principal components."""
import numpy as np


def pca(X, ndim):
    """Center X and return its projection onto ndim principal components."""
    centered = X - np.mean(X, axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return np.matmul(centered, vectors[:ndim].T)

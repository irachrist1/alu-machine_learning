#!/usr/bin/env python3
"""Cluster a dataset with Lloyd's K-means algorithm."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Return centroids and their nearest-point labels, or (None, None)."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            X.dtype.kind not in 'biuf' or not np.all(np.isfinite(X)) or
            type(k) is not int or not 1 <= k <= X.shape[0] or
            type(iterations) is not int or iterations <= 0):
        return None, None
    low, high = X.min(axis=0), X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, X.shape[1]))
    for _ in range(iterations):
        distances = np.sum((X[:, None, :] - C) ** 2, axis=2)
        clss = np.argmin(distances, axis=1)
        updated = C.copy()
        for cluster in range(k):
            points = X[clss == cluster]
            if points.shape[0] == 0:
                updated[cluster] = np.random.uniform(low, high)
            else:
                updated[cluster] = points.mean(axis=0)
        if np.array_equal(C, updated):
            break
        C = updated
    clss = np.argmin(np.sum((X[:, None, :] - C) ** 2, axis=2), axis=1)
    return C, clss

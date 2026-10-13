#!/usr/bin/env python3
"""Update Gaussian mixture parameters from posterior probabilities."""
import numpy as np


def maximization(X, g):
    """Return updated priors, means, and covariances, or three Nones."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            X.dtype.kind not in 'biuf' or not np.all(np.isfinite(X)) or
            not isinstance(g, np.ndarray) or g.ndim != 2 or
            g.shape[0] == 0 or g.shape[1] != X.shape[0] or
            g.dtype.kind not in 'biuf' or not np.all(np.isfinite(g)) or
            not np.allclose(g.sum(axis=0), 1)):
        return None, None, None
    counts = g.sum(axis=1)
    if np.any(counts == 0):
        return None, None, None
    pi = counts / X.shape[0]
    m = np.dot(g, X.astype(float)) / counts[:, None]
    S = np.empty((g.shape[0], X.shape[1], X.shape[1]))
    for cluster in range(g.shape[0]):
        centered = X - m[cluster]
        S[cluster] = (np.dot((g[cluster, :, None] * centered).T, centered)
                      / counts[cluster])
    return pi, m, S

#!/usr/bin/env python3
"""Initialize a Gaussian mixture from K-means centroids."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Return uniform priors, K-means means, and identity covariances."""
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    pi = np.full(k, 1.0 / k)
    S = np.tile(np.eye(X.shape[1]), (k, 1, 1))
    return pi, m, S

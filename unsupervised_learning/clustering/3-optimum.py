#!/usr/bin/env python3
"""Compare K-means cluster counts using the decrease in variance."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Return K-means results and variance decreases for a range of k."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            X.dtype.kind not in 'biuf' or not np.all(np.isfinite(X)) or
            type(kmin) is not int or kmin <= 0 or
            type(iterations) is not int or iterations <= 0):
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or not kmin < kmax <= X.shape[0]:
        return None, None
    results, variances = [], []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))
        variances.append(variance(X, C))
    d_vars = (variances[0] - np.asarray(variances)).tolist()
    return results, d_vars

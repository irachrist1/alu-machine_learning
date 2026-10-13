#!/usr/bin/env python3
"""Evaluate the density of a multivariate Gaussian distribution."""
import numpy as np


def pdf(X, m, S):
    """Return Gaussian densities, floored at 1e-300, or None on failure."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            X.dtype.kind not in 'biuf' or not np.all(np.isfinite(X))):
        return None
    d = X.shape[1]
    if (not isinstance(m, np.ndarray) or m.shape != (d,) or
            m.dtype.kind not in 'biuf' or not np.all(np.isfinite(m)) or
            not isinstance(S, np.ndarray) or S.shape != (d, d) or
            S.dtype.kind not in 'biuf' or not np.all(np.isfinite(S)) or
            not np.allclose(S, S.T)):
        return None
    try:
        factor = np.linalg.cholesky(S)
        centered = np.linalg.solve(factor, (X.astype(float) - m).T)
    except np.linalg.LinAlgError:
        return None
    indices = np.arange(d)
    log_det = 2 * np.sum(np.log(factor[indices, indices]))
    exponent = np.sum(centered ** 2, axis=0)
    log_density = -0.5 * (d * np.log(2 * np.pi) + log_det + exponent)
    return np.maximum(np.exp(log_density), 1e-300)

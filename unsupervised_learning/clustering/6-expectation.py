#!/usr/bin/env python3
"""Calculate Gaussian mixture posterior probabilities."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Return responsibilities and total log likelihood, or two Nones."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            X.dtype.kind not in 'biuf' or not np.all(np.isfinite(X)) or
            not isinstance(pi, np.ndarray) or pi.ndim != 1 or
            pi.size == 0 or pi.dtype.kind not in 'biuf' or
            not np.all(np.isfinite(pi)) or np.any(pi < 0) or
            not np.isclose(pi.sum(), 1)):
        return None, None
    k, d = pi.size, X.shape[1]
    if (not isinstance(m, np.ndarray) or m.shape != (k, d) or
            not isinstance(S, np.ndarray) or S.shape != (k, d, d)):
        return None, None
    weighted = np.empty((k, X.shape[0]))
    for cluster in range(k):
        density = pdf(X, m[cluster], S[cluster])
        if density is None:
            return None, None
        weighted[cluster] = pi[cluster] * density
    total = weighted.sum(axis=0)
    return weighted / total, np.sum(np.log(total))

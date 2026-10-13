#!/usr/bin/env python3
"""Select a Gaussian mixture size with Bayesian information criterion."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Return the best cluster count, its parameters, likelihoods and BICs."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            X.dtype.kind not in 'biuf' or not np.all(np.isfinite(X)) or
            type(kmin) is not int or kmin <= 0 or
            type(iterations) is not int or iterations <= 0 or
            type(tol) is not float or not np.isfinite(tol) or tol < 0 or
            type(verbose) is not bool):
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or not kmin <= kmax <= X.shape[0]:
        return None, None, None, None
    likelihoods = np.empty(kmax - kmin + 1)
    scores = np.empty(kmax - kmin + 1)
    best_k, best_result, best_score = None, None, np.inf
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S, _, likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None
        parameters = k * (d + d * (d + 1) / 2) + k - 1
        score = parameters * np.log(n) - 2 * likelihood
        likelihoods[k - kmin] = likelihood
        scores[k - kmin] = score
        if score < best_score:
            best_k, best_result, best_score = k, (pi, m, S), score
    return best_k, best_result, likelihoods, scores

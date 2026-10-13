#!/usr/bin/env python3
"""Fit a Gaussian mixture using expectation maximization."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Return mixture parameters, responsibilities, and log likelihood."""
    if (type(iterations) is not int or iterations <= 0 or
            type(tol) is not float or not np.isfinite(tol) or tol < 0 or
            type(verbose) is not bool):
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None
    g, likelihood = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None
    if verbose:
        print('Log Likelihood after 0 iterations: {}'.format(
            round(likelihood, 5)))
    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None
        g, updated = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None
        converged = abs(updated - likelihood) <= tol
        likelihood = updated
        if verbose and (i % 10 == 0 or converged or i == iterations):
            print('Log Likelihood after {} iterations: {}'.format(
                i, round(likelihood, 5)))
        if converged:
            break
    return pi, m, S, g, likelihood

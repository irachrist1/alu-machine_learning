#!/usr/bin/env python3
"""Stationary probabilities for regular Markov chains."""
import numpy as np


def regular(P):
    """Return a stationary row vector only if some power of P is positive."""
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] == 0 or P.shape[0] != P.shape[1] or
            P.dtype.kind not in 'fiu' or not np.all(np.isfinite(P)) or
            np.any(P < 0) or not np.allclose(P.sum(axis=1), 1)):
        return None
    n = P.shape[0]
    edges = P > 0
    reachable = edges.copy()
    # Wielandt's bound avoids guessing a numerical convergence cutoff.
    for exponent in range(1, (n - 1) ** 2 + 2):
        if np.all(reachable):
            break
        reachable = np.matmul(reachable, edges)
    else:
        return None
    system = np.vstack((P.T - np.eye(n), np.ones((1, n))))
    target = np.append(np.zeros(n), 1)
    stationary = np.linalg.lstsq(system, target, rcond=None)[0]
    stationary = np.maximum(stationary, 0)
    return (stationary / stationary.sum()).reshape(1, n)

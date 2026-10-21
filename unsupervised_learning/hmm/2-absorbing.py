#!/usr/bin/env python3
"""Reachability test for absorbing Markov chains."""
import numpy as np


def absorbing(P):
    """Return whether every state can reach an absorbing state."""
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] == 0 or P.shape[0] != P.shape[1] or
            P.dtype.kind not in 'fiu' or not np.all(np.isfinite(P)) or
            np.any(P < 0) or not np.allclose(P.sum(axis=1), 1)):
        return False
    reached = np.diag(P) == 1
    if not np.any(reached):
        return False
    for _ in range(P.shape[0]):
        previous = reached.copy()
        reached |= np.any((P > 0) & reached[np.newaxis, :], axis=1)
        if np.array_equal(previous, reached):
            break
    return bool(np.all(reached))

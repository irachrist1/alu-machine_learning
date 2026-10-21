#!/usr/bin/env python3
"""Backward likelihood and suffix probabilities."""
import numpy as np


def _valid(Observation, Emission, Transition, Initial):
    """Validate the discrete HMM dimensions and probability distributions."""
    if (not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            min(Emission.shape) == 0):
        return False
    n, m = Emission.shape
    arrays = (Emission, Transition, Initial)
    for array in arrays:
        if (not isinstance(array, np.ndarray) or
                array.dtype.kind not in 'fiu' or
                not np.all(np.isfinite(array)) or np.any(array < 0)):
            return False
    if Transition.shape != (n, n) or Initial.shape != (n, 1):
        return False
    if (not np.allclose(Emission.sum(axis=1), 1) or
            not np.allclose(Transition.sum(axis=1), 1) or
            not np.isclose(Initial.sum(), 1)):
        return False
    return (isinstance(Observation, np.ndarray) and
            Observation.ndim == 1 and Observation.size > 0 and
            Observation.dtype.kind in 'iu' and
            np.all(Observation >= 0) and np.all(Observation < m))


def backward(Observation, Emission, Transition, Initial):
    """Return observation likelihood and the backward trellis."""
    if not _valid(Observation, Emission, Transition, Initial):
        return None, None
    B = np.ones((Emission.shape[0], Observation.size))
    for t in range(Observation.size - 2, -1, -1):
        suffix = Emission[:, Observation[t + 1]] * B[:, t + 1]
        B[:, t] = np.matmul(Transition, suffix)
    probability = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return probability, B

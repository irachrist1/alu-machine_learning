#!/usr/bin/env python3
"""Forward likelihood and unscaled joint prefix probabilities."""
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


def forward(Observation, Emission, Transition, Initial):
    """Sum over hidden paths to return observation likelihood and trellis."""
    if not _valid(Observation, Emission, Transition, Initial):
        return None, None
    F = np.zeros((Emission.shape[0], Observation.size))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, Observation.size):
        F[:, t] = np.matmul(F[:, t - 1], Transition)
        F[:, t] *= Emission[:, Observation[t]]
    return np.sum(F[:, -1]), F

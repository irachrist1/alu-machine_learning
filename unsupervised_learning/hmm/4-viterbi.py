#!/usr/bin/env python3
"""Log-space Viterbi decoding with exact zero-probability edges."""
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


def viterbi(Observation, Emission, Transition, Initial):
    """Return the most likely hidden path and its joint probability."""
    if not _valid(Observation, Emission, Transition, Initial):
        return None, None
    n, length = Emission.shape[0], Observation.size
    with np.errstate(divide='ignore'):
        log_a = np.log(Transition)
        log_b = np.log(Emission)
        scores = np.log(Initial[:, 0]) + log_b[:, Observation[0]]
    previous = np.zeros((n, length), dtype=int)
    for t in range(1, length):
        candidates = scores[:, np.newaxis] + log_a
        previous[:, t] = np.argmax(candidates, axis=0)
        scores = np.max(candidates, axis=0) + log_b[:, Observation[t]]
    state = int(np.argmax(scores))
    probability = float(np.exp(scores[state]))
    path = [state]
    for t in range(length - 1, 0, -1):
        state = int(previous[state, t])
        path.append(state)
    return path[::-1], probability

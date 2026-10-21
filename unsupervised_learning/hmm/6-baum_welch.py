#!/usr/bin/env python3
"""Scaled Baum-Welch estimation for a discrete hidden Markov model."""
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Estimate transitions/emissions with the supplied fixed prior."""
    if (not _valid(Observations, Emission, Transition, Initial) or
            isinstance(iterations, (bool, np.bool_)) or
            not isinstance(iterations, (int, np.integer)) or iterations <= 0):
        return None, None
    a = Transition.astype(float).copy()
    b = Emission.astype(float).copy()
    n, length = a.shape[0], Observations.size
    old_likelihood = None
    for _ in range(iterations):
        alpha = np.zeros((n, length))
        scales = np.zeros(length)
        alpha[:, 0] = Initial[:, 0] * b[:, Observations[0]]
        scales[0] = alpha[:, 0].sum()
        if scales[0] <= 0:
            return None, None
        alpha[:, 0] /= scales[0]
        for t in range(1, length):
            alpha[:, t] = np.matmul(alpha[:, t - 1], a)
            alpha[:, t] *= b[:, Observations[t]]
            scales[t] = alpha[:, t].sum()
            if scales[t] <= 0:
                return None, None
            alpha[:, t] /= scales[t]
        likelihood = np.log(scales).sum()
        if (old_likelihood is not None and
                abs(likelihood - old_likelihood) < 1e-8):
            break
        old_likelihood = likelihood
        beta = np.ones((n, length))
        for t in range(length - 2, -1, -1):
            beta[:, t] = np.matmul(
                a, b[:, Observations[t + 1]] * beta[:, t + 1])
            beta[:, t] /= scales[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=0, keepdims=True)
        counts = np.zeros_like(a)
        for t in range(length - 1):
            pair = alpha[:, t, np.newaxis] * a
            pair *= (b[:, Observations[t + 1]] *
                     beta[:, t + 1])[np.newaxis, :]
            counts += pair / pair.sum()
        departures = counts.sum(axis=1)
        occupied = departures > 0
        a[occupied] = counts[occupied] / departures[occupied, np.newaxis]
        occupancy = gamma.sum(axis=1)
        occupied = occupancy > 0
        for symbol in range(b.shape[1]):
            counts_b = gamma[:, Observations == symbol].sum(axis=1)
            b[occupied, symbol] = counts_b[occupied] / occupancy[occupied]
    return a, b

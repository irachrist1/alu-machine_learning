#!/usr/bin/env python3
"""One-hot encoding."""
import numpy as np


def one_hot_encode(Y, classes):
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None

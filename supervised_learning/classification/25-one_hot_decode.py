#!/usr/bin/env python3
"""One-hot decoding."""
import numpy as np


def one_hot_decode(one_hot):
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None

#!/usr/bin/env python3
"""NumPy matrix concatenation helper."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Return a new array concatenated along a specific axis."""
    return np.concatenate((mat1, mat2), axis=axis)

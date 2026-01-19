#!/usr/bin/env python3
"""2D matrix concatenation helper."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Return a new matrix concatenated along a specific axis."""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    return None

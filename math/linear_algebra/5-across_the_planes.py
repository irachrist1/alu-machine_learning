#!/usr/bin/env python3
"""2D matrix element-wise addition."""


def add_matrices2D(mat1, mat2):
    """Return a new matrix with element-wise sums of two 2D matrices."""
    if len(mat1) != len(mat2):
        return None
    result = []
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
        result.append([item1 + item2 for item1, item2 in zip(row1, row2)])
    return result

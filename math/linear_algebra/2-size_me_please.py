#!/usr/bin/env python3
"""Matrix shape utility."""


def matrix_shape(matrix):
    """Return the shape of a matrix as a list of integers."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape

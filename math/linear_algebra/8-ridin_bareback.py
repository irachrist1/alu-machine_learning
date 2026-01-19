#!/usr/bin/env python3
"""Matrix multiplication helper."""


def mat_mul(mat1, mat2):
    """Return the product of two 2D matrices."""
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for row in mat1:
        new_row = []
        for col_idx in range(len(mat2[0])):
            total = 0
            for k in range(len(mat2)):
                total += row[k] * mat2[k][col_idx]
            new_row.append(total)
        result.append(new_row)
    return result

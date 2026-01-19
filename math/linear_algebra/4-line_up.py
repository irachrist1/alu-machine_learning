#!/usr/bin/env python3
"""Element-wise array addition."""


def add_arrays(arr1, arr2):
    """Return a new list with element-wise sums of two arrays."""
    if len(arr1) != len(arr2):
        return None
    return [item1 + item2 for item1, item2 in zip(arr1, arr2)]

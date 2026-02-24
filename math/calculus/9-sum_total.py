#!/usr/bin/env python3
"""Module that provides a summation function."""


def summation_i_squared(n):
    """Calculate the sum of squared values from 1 to n."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6

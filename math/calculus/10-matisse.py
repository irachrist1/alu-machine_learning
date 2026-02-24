#!/usr/bin/env python3
"""Module that provides polynomial derivative functionality."""


def poly_derivative(poly):
    """Calculate and return the derivative of a polynomial."""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not all(isinstance(c, (int, float)) and not isinstance(c, bool)
               for c in poly):
        return None

    if len(poly) == 1:
        return [0]

    derivative = [poly[i] * i for i in range(1, len(poly))]

    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    if all(value == 0 for value in derivative):
        return [0]

    return derivative

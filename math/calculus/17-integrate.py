#!/usr/bin/env python3
"""Module that provides polynomial integral functionality."""


def poly_integral(poly, C=0):
    """Calculate and return the integral of a polynomial."""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not isinstance(C, int) or isinstance(C, bool):
        return None

    if not all(isinstance(c, (int, float)) and not isinstance(c, bool)
               for c in poly):
        return None

    integral = [C]
    for i, coeff in enumerate(poly):
        value = coeff / (i + 1)
        if value.is_integer():
            value = int(value)
        integral.append(value)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral

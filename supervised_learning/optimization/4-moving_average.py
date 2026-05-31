#!/usr/bin/env python3
"""Moving average module."""


def moving_average(data, beta):
    """Calculate the weighted moving average of a data set.

    Args:
        data: list of data to calculate the moving average of.
        beta: weight used for the moving average.

    Returns:
        A list containing the moving averages of data.
    """
    averages = []
    weighted_average = 0
    for t, value in enumerate(data, 1):
        weighted_average = beta * weighted_average + (1 - beta) * value
        averages.append(weighted_average / (1 - beta ** t))
    return averages

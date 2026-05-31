#!/usr/bin/env python3
"""Learning rate decay module."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Update the learning rate using inverse time decay.

    Args:
        alpha: original learning rate.
        decay_rate: weight that determines the decay rate.
        global_step: number of gradient descent passes elapsed.
        decay_step: passes before alpha is decayed further.

    Returns:
        The updated learning rate.
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))

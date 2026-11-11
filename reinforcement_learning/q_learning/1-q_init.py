#!/usr/bin/env python3
"""Initialize action values for a discrete Frozen Lake environment."""

import numpy as np


def q_init(env):
    """Return one zero action value for each state and available action."""
    return np.zeros((env.observation_space.n, env.action_space.n))

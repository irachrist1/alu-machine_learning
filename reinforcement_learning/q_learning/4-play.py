#!/usr/bin/env python3
"""Run a greedy policy from a learned Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Display the initial and subsequent boards and return episode reward."""
    state = env.reset()
    total_reward = 0
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            break
    return total_reward

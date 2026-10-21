#!/usr/bin/env python3
"""TD(lambda) prediction with accumulating eligibility traces."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Propagate each TD error through the episode's state traces."""
    for _ in range(episodes):
        state = env.reset()
        eligibility = np.zeros_like(V, dtype=float)
        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(policy(state))
            target = reward if done else reward + gamma * V[next_state]
            error = target - V[state]
            eligibility[state] += 1
            V += alpha * error * eligibility
            eligibility *= gamma * lambtha
            state = next_state
            if done:
                break
    return V

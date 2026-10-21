#!/usr/bin/env python3
"""On-policy SARSA(lambda) control with accumulating traces."""
import numpy as np


def _epsilon_greedy(Q, state, epsilon):
    """Sample an exploratory action or use the first maximizing action."""
    if np.random.uniform() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[state]))


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Update Q using the next action sampled by the behavior policy."""
    initial_epsilon = epsilon
    for episode in range(episodes):
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode)
        state = env.reset()
        action = _epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q, dtype=float)
        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            if done:
                error = reward - Q[state, action]
            else:
                next_action = _epsilon_greedy(Q, next_state, epsilon)
                error = (reward + gamma * Q[next_state, next_action] -
                         Q[state, action])
            eligibility[state, action] += 1
            Q += alpha * error * eligibility
            eligibility *= gamma * lambtha
            if done:
                break
            state, action = next_state, next_action
    return Q

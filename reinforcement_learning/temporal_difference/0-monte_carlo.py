#!/usr/bin/env python3
"""First-visit Monte Carlo prediction for a fixed policy."""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Update V from discounted episode returns using the Gym 0.7 API."""
    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        first_visit = {}
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            if state not in first_visit:
                first_visit[state] = step
            trajectory.append((state, reward))
            state = next_state
            if done:
                break
        total = 0.0
        for step in range(len(trajectory) - 1, -1, -1):
            state, reward = trajectory[step]
            total = reward + gamma * total
            if first_visit[state] == step:
                V[state] += alpha * (total - V[state])
    return V

#!/usr/bin/env python3
"""Learn a Frozen Lake action-value table using tabular Q-learning."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Update Q in place and return it with the total reward of each episode.

    Falling into a hole receives a reward of -1. Terminal transitions use
    only their immediate reward, since there is no subsequent action.
    Exploration decays exponentially after each episode.
    """
    initial_epsilon = epsilon
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            if done and reward == 0:
                reward = -1
            target = reward
            if not done:
                target += gamma * np.max(Q[next_state])
            Q[state, action] += alpha * (target - Q[state, action])
            episode_reward += reward
            state = next_state
            if done:
                break
        total_rewards.append(episode_reward)
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode)
    return Q, total_rewards

#!/usr/bin/env python3
"""Hand-calculated trajectory checks independent of stochastic Gym versions."""
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
monte_carlo = importlib.import_module('0-monte_carlo').monte_carlo
td_lambtha = importlib.import_module('1-td_lambtha').td_lambtha
sarsa_module = importlib.import_module('2-sarsa_lambtha')
sarsa_lambtha = sarsa_module.sarsa_lambtha


class ScriptedEnv:
    """Replay explicit rewards and transitions with the required Gym API."""

    def __init__(self, episodes):
        self.episodes = episodes
        self.episode = -1
        self.actions = []

    def reset(self):
        """Start the next prescribed trajectory."""
        self.episode += 1
        self.step_index = 0
        self.done = False
        return self.episodes[self.episode][0]

    def step(self, action):
        """Return the next transition; reject stepping after termination."""
        if self.done:
            raise AssertionError('A terminal environment must be reset')
        self.actions.append(action)
        transition = self.episodes[self.episode][1][self.step_index]
        state, reward, self.done = transition
        self.step_index += 1
        return state, reward, self.done, {}


class TemporalDifferenceTests(unittest.TestCase):
    """Distinguish correct returns and traces from common errors."""

    def test_monte_carlo_uses_first_visit_and_includes_terminal_reward(self):
        env = ScriptedEnv([(0, [(1, 1, False), (0, 2, False), (2, 3, True)])])
        values = np.array([0., 0., 99.])
        result = monte_carlo(env, values, lambda s: 0,
                             episodes=1, alpha=1, gamma=.5)
        self.assertIs(result, values)
        np.testing.assert_allclose(result, [2.75, 3.5, 99])

    def test_monte_carlo_learning_rate_and_episode_horizon(self):
        env = ScriptedEnv([(0, [(1, 2, False), (2, 100, True)])])
        result = monte_carlo(env, np.array([1., 0., 0.]), lambda s: 0,
                             episodes=1, max_steps=1, alpha=.25)
        np.testing.assert_allclose(result, [1.25, 0, 0])
        self.assertEqual(len(env.actions), 1)

    def test_td_trace_decay_and_lambda_zero(self):
        for lambtha, expected in [(0, 0), (.5, .45), (1, .9)]:
            env = ScriptedEnv([(0, [(1, 0, False), (2, 1, True)])])
            result = td_lambtha(env, np.array([0., 0., 99.]), lambda s: 0,
                                lambtha, episodes=1, alpha=1, gamma=.9)
            np.testing.assert_allclose(result, [expected, 1, 99])

    def test_td_accumulating_traces_on_revisit(self):
        env = ScriptedEnv([(0, [(0, 1, False), (1, 2, True)])])
        result = td_lambtha(env, np.zeros(2), lambda s: 0, .5,
                            episodes=1, alpha=.2, gamma=.5)
        np.testing.assert_allclose(result, [.65, 0])

    def test_td_resets_traces_between_episodes(self):
        env = ScriptedEnv([(0, [(2, 1, True)]), (1, [(2, 2, True)])])
        result = td_lambtha(env, np.zeros(3), lambda s: 0, 1,
                            episodes=2, alpha=1)
        np.testing.assert_allclose(result, [1, 2, 0])

    def test_sarsa_uses_sampled_next_action_instead_of_maximum(self):
        env = ScriptedEnv([(0, [(1, 0, False)])])
        q = np.array([[0., 0.], [10., 2.]])
        with patch.object(sarsa_module, '_epsilon_greedy',
                          side_effect=[0, 1]) as choose:
            result = sarsa_lambtha(env, q, .5, episodes=1,
                                   max_steps=1, alpha=1, gamma=.9)
        self.assertIs(result, q)
        self.assertEqual(choose.call_count, 2)
        self.assertAlmostEqual(q[0, 0], 1.8)
        self.assertEqual(env.actions, [0])

    def test_sarsa_terminal_reward_and_trace_reset(self):
        env = ScriptedEnv([(0, [(2, 1, True)]), (1, [(2, 2, True)])])
        q = np.array([[0., 0.], [0., 0.], [99., 99.]])
        with patch.object(sarsa_module, '_epsilon_greedy', return_value=0):
            result = sarsa_lambtha(env, q, 1, episodes=2, alpha=1)
        np.testing.assert_allclose(result, [[1, 0], [2, 0], [99, 99]])

    def test_sarsa_propagates_error_through_state_action_traces(self):
        env = ScriptedEnv([(0, [(1, 0, False), (2, 1, True)])])
        with patch.object(sarsa_module, '_epsilon_greedy',
                          side_effect=[0, 1]):
            result = sarsa_lambtha(env, np.zeros((3, 2)), .5,
                                   episodes=1, alpha=1, gamma=.9)
        np.testing.assert_allclose(result, [[.45, 0], [0, 1], [0, 0]])

    def test_epsilon_starts_at_requested_value_and_decays(self):
        env = ScriptedEnv([(0, [(1, 1, True)])] * 3)
        with patch.object(sarsa_module, '_epsilon_greedy',
                          return_value=0) as choose:
            sarsa_lambtha(env, np.zeros((2, 1)), 0, episodes=3,
                          epsilon=.9, min_epsilon=.1, epsilon_decay=.5)
        observed = [call.args[2] for call in choose.call_args_list]
        np.testing.assert_allclose(observed,
                                   .1 + .8 * np.exp(-.5 * np.arange(3)))


if __name__ == '__main__':
    unittest.main()

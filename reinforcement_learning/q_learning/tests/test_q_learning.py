#!/usr/bin/env python3
"""Check legacy Gym examples and independently computed Bellman targets."""

from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
load_frozen_lake = importlib.import_module('0-load_env').load_frozen_lake
q_init = importlib.import_module('1-q_init').q_init
epsilon_greedy = importlib.import_module('2-epsilon_greedy').epsilon_greedy
training = importlib.import_module('3-q_learning')
play = importlib.import_module('4-play').play
MAP = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]


class OneStepEnv:
    """Expose one explicit transition and reject an extra step."""

    def __init__(self, reward, done):
        self.reward = reward
        self.done = done
        self.resets = 0
        self.steps = 0

    def reset(self):
        """Reset the one-step episode."""
        self.resets += 1
        self.steps = 0
        return 0

    def step(self, action):
        """Return a known transition to state one."""
        self.steps += 1
        if self.steps > 1:
            raise AssertionError('The episode should have stopped')
        return 1, self.reward, self.done, {}


class QLearningTests(unittest.TestCase):
    """Exercise map loading, stochastic decisions, updates, and evaluation."""

    def test_random_map_matches_supplied_seeded_example(self):
        np.random.seed(0)
        env = load_frozen_lake()
        rows = ['SFFFFFFH', 'HFFFFHFF', 'FHFHHFFF', 'FFFHFFFF',
                'FFFFFFHF', 'FFFFFFFF', 'FFFFHFFF', 'FFFFFFFG']
        expected = np.asarray([list(row) for row in rows], dtype='c')
        np.testing.assert_array_equal(env.desc, expected)
        self.assertEqual(env.P[0][0], [(1.0, 0, 0.0, False)])

    def test_custom_named_and_slippery_environments(self):
        custom = load_frozen_lake(desc=MAP)
        named = load_frozen_lake(map_name='4x4')
        slippery = load_frozen_lake(desc=MAP, is_slippery=True)
        self.assertEqual(custom.desc.shape, (3, 3))
        self.assertEqual(named.desc.shape, (4, 4))
        self.assertEqual(len(slippery.P[0][0]), 3)
        self.assertAlmostEqual(sum(t[0] for t in slippery.P[0][0]), 1)
        for env in (custom, named, slippery):
            q = q_init(env)
            self.assertEqual(q.shape, (env.observation_space.n, 4))
            np.testing.assert_array_equal(q, np.zeros(q.shape))

    def test_epsilon_greedy_matches_both_seeded_examples(self):
        q = np.zeros((9, 4))
        q[7] = [.5, .7, 1, -1]
        np.random.seed(0)
        self.assertEqual(epsilon_greedy(q, 7, .5), 2)
        np.random.seed(1)
        self.assertEqual(epsilon_greedy(q, 7, .5), 0)

    def test_nonterminal_update_uses_maximum_next_action_value(self):
        env = OneStepEnv(3, False)
        q = np.array([[0., 1.], [2., 5.]])
        with patch.object(training, 'epsilon_greedy', return_value=1):
            result, rewards = training.train(env, q, episodes=1,
                                             max_steps=1, alpha=.5, gamma=.9)
        self.assertIs(result, q)
        self.assertAlmostEqual(q[0, 1], 4.25)
        self.assertEqual(rewards, [3])
        self.assertEqual(env.steps, 1)

    def test_goal_does_not_bootstrap_terminal_values(self):
        env = OneStepEnv(1, True)
        q = np.array([[0., 0.], [99., 99.]])
        with patch.object(training, 'epsilon_greedy', return_value=0):
            _, rewards = training.train(env, q, episodes=2, alpha=1)
        self.assertEqual(q[0, 0], 1)
        self.assertEqual(rewards, [1, 1])
        self.assertEqual(env.resets, 2)

    def test_hole_reward_is_negative_one(self):
        env = OneStepEnv(0, True)
        q = np.zeros((2, 2))
        with patch.object(training, 'epsilon_greedy', return_value=0):
            _, rewards = training.train(env, q, episodes=1, alpha=1)
        self.assertEqual(q[0, 0], -1)
        self.assertEqual(rewards, [-1])

    def test_training_matches_supplied_table_and_reward_batches(self):
        np.random.seed(0)
        env = load_frozen_lake(desc=MAP)
        q, rewards = training.train(env, q_init(env))
        expected = [[.96059593, .970299, .95098488, .96059396],
                    [.96059557, -.77123208, .0094072, .37627228],
                    [.18061285, -.1, 0, 0],
                    [.97029877, .9801, -.99999988, .96059583],
                    [0, 0, 0, 0], [0, 0, 0, 0],
                    [.98009763, .98009933, .99, .9702983],
                    [.98009922, .98999782, 1, -.99999952],
                    [0, 0, 0, 0]]
        np.testing.assert_allclose(q, expected, atol=5e-9, rtol=0)
        means = [np.mean(batch) for batch in np.split(np.array(rewards), 10)]
        expected_means = [.812, .88, .9, .9, .88, .844, .892, .896, .852, .928]
        np.testing.assert_allclose(means, expected_means)

    def test_greedy_play_reaches_goal_and_renders(self):
        env = load_frozen_lake(desc=MAP)
        q = q_init(env)
        for state, action in [(0, 1), (3, 1), (6, 2), (7, 2)]:
            q[state, action] = 1
        output = io.StringIO()
        with redirect_stdout(output):
            reward = play(env, q)
        self.assertEqual(reward, 1)
        self.assertIn('Down', output.getvalue())
        self.assertIn('Right', output.getvalue())
        self.assertEqual(output.getvalue().count('G'), 5)


if __name__ == '__main__':
    unittest.main()

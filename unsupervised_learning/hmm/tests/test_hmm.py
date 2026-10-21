#!/usr/bin/env python3
"""Compare HMM algorithms against enumeration and independent invariants."""
import importlib
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
markov_chain = importlib.import_module('0-markov_chain').markov_chain
regular = importlib.import_module('1-regular').regular
absorbing = importlib.import_module('2-absorbing').absorbing
forward = importlib.import_module('3-forward').forward
viterbi = importlib.import_module('4-viterbi').viterbi
backward = importlib.import_module('5-backward').backward
baum_welch = importlib.import_module('6-baum_welch').baum_welch


def paths(observations, emission, transition, initial):
    """Enumerate path probabilities independently for small models."""
    result = {}
    for path in itertools.product(range(len(initial)),
                                  repeat=len(observations)):
        probability = initial[path[0], 0] * emission[path[0], observations[0]]
        for t in range(1, len(observations)):
            probability *= transition[path[t - 1], path[t]]
            probability *= emission[path[t], observations[t]]
        result[path] = probability
    return result


class HMMTests(unittest.TestCase):
    """Check numerical outputs, topology, underflow, and invalid inputs."""

    def setUp(self):
        self.a = np.array([[.6, .4], [.5, .5]])
        self.b = np.array([[.2, .4, .4], [.5, .4, .1]])
        self.pi = np.array([[.8], [.2]])
        self.o = np.array([2, 0, 2])

    def test_previous_ice_cream_exercise(self):
        expected = paths(self.o, self.b, self.a, self.pi)
        self.assertAlmostEqual(sum(expected.values()), .028562)
        self.assertAlmostEqual(expected[(0, 0, 0)], .009216)
        path, probability = viterbi(self.o, self.b, self.a, self.pi)
        self.assertEqual(path, [0, 1, 0])
        self.assertAlmostEqual(probability, .0128)
        for algorithm in (forward, backward):
            self.assertAlmostEqual(
                algorithm(self.o, self.b, self.a, self.pi)[0], .028562)

    def test_random_models_against_all_paths(self):
        rng = np.random.RandomState(12)
        for _ in range(30):
            a = rng.dirichlet(np.ones(3), 3)
            b = rng.dirichlet(np.ones(2), 3)
            pi = rng.dirichlet(np.ones(3)).reshape(3, 1)
            o = rng.randint(0, 2, 5)
            expected = paths(o, b, a, pi)
            probability, f = forward(o, b, a, pi)
            reverse_probability, backward_table = backward(o, b, a, pi)
            path, best = viterbi(o, b, a, pi)
            self.assertAlmostEqual(probability, sum(expected.values()))
            self.assertAlmostEqual(reverse_probability, probability)
            self.assertAlmostEqual(best, max(expected.values()))
            self.assertAlmostEqual(expected[tuple(path)], best)
            np.testing.assert_allclose(
                (f * backward_table).sum(axis=0), probability)

    def test_markov_evolution_and_stationary_distribution(self):
        a = np.array([[.6, .4], [.3, .7]])
        s = np.array([[1., 0.]])
        np.testing.assert_array_equal(markov_chain(a, s, 0), s)
        np.testing.assert_allclose(markov_chain(a, s, 2), [[.48, .52]])
        stationary = regular(a)
        np.testing.assert_allclose(stationary, [[3 / 7, 4 / 7]])
        np.testing.assert_allclose(stationary @ a, stationary)

    def test_regular_requires_irreducibility_and_aperiodicity(self):
        self.assertIsNone(regular(np.eye(2)))
        self.assertIsNone(regular(np.array([[0., 1.], [1., 0.]])))
        a = np.array([[.5, .5, 0], [0, 0, 1], [1, 0, 0]])
        stationary = regular(a)
        np.testing.assert_allclose(stationary @ a, stationary)

    def test_absorption_depends_on_reachability(self):
        self.assertTrue(absorbing(np.eye(2)))
        self.assertFalse(absorbing(self.a))
        reachable = np.array([[1., 0, 0], [.2, .8, 0], [0, .2, .8]])
        self.assertTrue(absorbing(reachable))
        reachable[2] = [0, 0, 1]
        self.assertTrue(absorbing(reachable))
        closed_class = np.array([[1., 0, 0], [0, 0, 1], [0, 1, 0]])
        self.assertFalse(absorbing(closed_class))

    def test_single_observation_and_impossible_sequence(self):
        o = np.array([0])
        for algorithm in (forward, backward):
            self.assertAlmostEqual(algorithm(o, self.b, self.a, self.pi)[0],
                                   .26)
        b = np.array([[1., 0], [1., 0]])
        for algorithm in (forward, backward, viterbi):
            result = algorithm(np.array([1]), b, self.a, self.pi)
            self.assertEqual(result[1] if algorithm == viterbi else result[0],
                             0)
        self.assertEqual(baum_welch(np.array([1]), self.a, b, self.pi),
                         (None, None))

    def test_viterbi_retains_path_after_probability_underflows(self):
        a = np.eye(2)
        b = np.array([[.6, .4], [.4, .6]])
        o = np.zeros(2000, dtype=int)
        path, probability = viterbi(o, b, a, np.array([[.5], [.5]]))
        self.assertEqual(path, [0] * 2000)
        self.assertEqual(probability, 0)

    def test_baum_welch_matches_course_example_and_preserves_inputs(self):
        np.random.seed(1)
        b = np.array([[.9, .1, 0], [.4, .5, .1]])
        a = np.array([[.6, .4], [.3, .7]])
        pi = np.array([[.5], [.5]])
        hidden = [np.random.choice(2, p=pi[:, 0])]
        for _ in range(364):
            hidden.append(np.random.choice(2, p=a[hidden[-1]]))
        o = np.array([np.random.choice(3, p=b[s]) for s in hidden])
        a0 = np.ones((2, 2)) / 2
        b0 = np.abs(np.random.randn(2, 3))
        b0 /= b0.sum(axis=1, keepdims=True)
        a_copy, b_copy = a0.copy(), b0.copy()
        fitted_a, fitted_b = baum_welch(o, a0, b0, pi)
        np.testing.assert_allclose(np.round(fitted_a, 2),
                                   [[.81, .19], [.28, .72]])
        np.testing.assert_allclose(np.round(fitted_b, 2),
                                   [[.82, .18, 0], [.26, .58, .16]])
        np.testing.assert_array_equal(a0, a_copy)
        np.testing.assert_array_equal(b0, b_copy)

    def test_baum_welch_improves_likelihood_and_handles_long_sequences(self):
        o = np.tile(self.o, 500)
        a, b = baum_welch(o, self.a, self.b, self.pi, iterations=4)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue(np.all(np.isfinite(b)))
        np.testing.assert_allclose(a.sum(axis=1), 1)
        np.testing.assert_allclose(b.sum(axis=1), 1)
        a, b = self.a, self.b
        previous = forward(self.o, b, a, self.pi)[0]
        for _ in range(8):
            a, b = baum_welch(self.o, a, b, self.pi, iterations=1)
            current = forward(self.o, b, a, self.pi)[0]
            self.assertGreaterEqual(current + 1e-12, previous)
            previous = current

    def test_invalid_models_and_observations_fail_cleanly(self):
        for invalid in (np.array([]), np.array([-.1]), np.array([-1]),
                        np.array([3]), np.array([[0]]), np.array([True])):
            for algorithm in (forward, backward, viterbi):
                self.assertEqual(algorithm(invalid, self.b, self.a, self.pi),
                                 (None, None))
        for bad in (np.array([[np.nan, 0], [0, 1]]), np.eye(2) * 2,
                    np.array([[1.1, -.1], [0, 1]]), np.zeros((0, 0))):
            self.assertIsNone(markov_chain(bad, np.array([[1., 0.]])))
            self.assertIsNone(regular(bad))
            self.assertFalse(absorbing(bad))
        self.assertIsNone(markov_chain(self.a, np.array([[1., 0.]]), True))
        self.assertEqual(baum_welch(self.o, self.a, self.b, self.pi, 0),
                         (None, None))


if __name__ == '__main__':
    unittest.main()

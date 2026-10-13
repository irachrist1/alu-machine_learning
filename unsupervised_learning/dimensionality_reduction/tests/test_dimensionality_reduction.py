"""Independent checks for PCA projections and input preservation."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pca


weights = load('0-pca')
project = load('1-pca')


class PCATests(unittest.TestCase):
    def test_course_centered_example_retains_three_directions(self):
        rng = np.random.RandomState(0)
        a, b, c = (rng.normal(size=50) for _ in range(3))
        X = np.array([a, b, c, 2 * a, -5 * b, 10 * c]).T
        X -= X.mean(axis=0)
        original = X.copy()
        W = weights(X)
        self.assertEqual(W.shape, (6, 3))
        np.testing.assert_allclose(W.T @ W, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(X @ W @ W.T, X, atol=1e-12)
        np.testing.assert_array_equal(X, original)

    def test_weights_fraction_changes_number_of_components(self):
        X = np.diag([8., 2., 1.])
        self.assertEqual(weights(X, .7).shape, (3, 1))
        self.assertEqual(weights(X, .9).shape, (3, 2))
        self.assertEqual(weights(X, 1.).shape, (3, 3))

    def test_projection_is_centered_and_translation_invariant(self):
        X = np.random.RandomState(12).normal(size=(30, 5))
        original = X.copy()
        T = project(X, 3)
        shifted = project(X + np.arange(5) * 20, 3)
        self.assertEqual(T.shape, (30, 3))
        np.testing.assert_allclose(T.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(T @ T.T, shifted @ shifted.T, atol=1e-11)
        np.testing.assert_array_equal(X, original)

    def test_projection_matches_independent_covariance_eigenspace(self):
        X = np.random.RandomState(9).normal(size=(40, 4))
        centered = X - X.mean(axis=0)
        _, eigenvectors = np.linalg.eigh(centered.T @ centered)
        expected = centered @ eigenvectors[:, -2:]
        result = project(X, 2)
        np.testing.assert_allclose(result @ result.T,
                                   expected @ expected.T, atol=1e-12)

    def test_full_projection_preserves_centered_distances(self):
        X = np.array([[2., 3., 4.], [4., 9., 3.],
                      [7., 1., 8.], [0., 5., 6.]])
        T = project(X, 3)
        before = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        after = np.sum((T[:, None] - T[None, :]) ** 2, axis=-1)
        np.testing.assert_allclose(after, before, atol=1e-12)


if __name__ == '__main__':
    unittest.main()

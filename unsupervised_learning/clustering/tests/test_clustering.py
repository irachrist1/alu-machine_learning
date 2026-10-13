#!/usr/bin/env python3
"""Regression checks for the Clustering project."""
import ast
import contextlib
import importlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
initialize = importlib.import_module('0-initialize').initialize
kmeans = importlib.import_module('1-kmeans').kmeans
variance = importlib.import_module('2-variance').variance
optimum_k = importlib.import_module('3-optimum').optimum_k
initialize_gmm = importlib.import_module('4-initialize').initialize
pdf = importlib.import_module('5-pdf').pdf
expectation = importlib.import_module('6-expectation').expectation
maximization = importlib.import_module('7-maximization').maximization
em = importlib.import_module('8-EM').expectation_maximization
bic = importlib.import_module('9-BIC').BIC
sklearn_kmeans = importlib.import_module('10-kmeans').kmeans
gmm = importlib.import_module('11-gmm').gmm
agglomerative = importlib.import_module('12-agglomerative').agglomerative


def restaurant_clusters():
    """Recreate the course's five-cluster K-means example."""
    np.random.seed(0)
    groups = [np.random.multivariate_normal(mean, 16 * np.eye(2), 50)
              for mean in ([30, 40], [10, 25], [40, 20], [60, 30], [20, 70])]
    X = np.concatenate(groups)
    np.random.shuffle(X)
    return X


def mixture_data():
    """Recreate the course's unequal-size Gaussian mixture example."""
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], 10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], 750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], 750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], 1000)
    X = np.concatenate((a, b, c, d))
    np.random.shuffle(X)
    return X


class ClusteringTests(unittest.TestCase):
    """Check numerical answers, failure cases, and algorithm invariants."""

    def test_initialize_matches_course_example(self):
        """The specified seed and uniform draw must match the example."""
        C = initialize(restaurant_clusters(), 5)
        expected = [[14.54730144, 13.46780434], [20.57098466, 33.55245039],
                    [9.55556506, 51.51143281], [48.72458008, 20.03154959],
                    [25.43826106, 60.35542243]]
        np.testing.assert_allclose(C, expected, atol=1e-8)

    def test_kmeans_matches_course_example(self):
        """The five separated clusters should recover the stated means."""
        X = restaurant_clusters()
        C, labels = kmeans(X, 5)
        expected = [[9.92511389, 25.73098987], [30.06722465, 40.41123947],
                    [39.62770705, 19.89843487], [59.22766628, 29.19796006],
                    [20.0835633, 69.81592298]]
        np.testing.assert_allclose(C, expected, atol=1e-8)
        self.assertEqual(labels.shape, (250,))

    def test_kmeans_final_labels_after_one_iteration(self):
        """Labels must correspond to the returned, updated centroids."""
        X = np.array([[0.], [1.], [4.], [9.], [10.]])
        with patch('numpy.random.uniform',
                   return_value=np.array([[0.], [1.]])):
            C, labels = kmeans(X, 2, iterations=1)
        np.testing.assert_array_equal(labels,
                                      np.argmin(abs(X - C.T), axis=1))

    def test_kmeans_reinitializes_empty_cluster(self):
        """An empty centroid gets another draw within the data bounds."""
        X = np.array([[0.], [10.], [10.]])
        draws = [np.array([[0.], [0.]]), np.array([10.])]
        with patch('numpy.random.uniform', side_effect=draws) as uniform:
            C, labels = kmeans(X, 2, iterations=1)
        self.assertEqual(uniform.call_count, 2)
        self.assertEqual(C.shape, (2, 1))
        np.testing.assert_array_equal(labels, [0, 1, 1])

    def test_invalid_datasets_and_parameters(self):
        """Invalid or empty inputs return the required failure tuples."""
        for X in (None, [], np.ones(3), np.empty((0, 2)),
                  np.empty((2, 0)), np.array([[np.nan]]),
                  np.array([['text']]), np.array([[1j]])):
            self.assertIsNone(initialize(X, 1))
            self.assertEqual(kmeans(X, 1), (None, None))
            self.assertEqual(initialize_gmm(X, 1), (None, None, None))
        X = np.array([[0.], [1.]])
        for k in (0, -1, 1.2, True, '2'):
            self.assertIsNone(initialize(X, k))
            self.assertEqual(kmeans(X, k), (None, None))
        for iterations in (0, -1, 1.2, True):
            self.assertEqual(kmeans(X, 1, iterations), (None, None))

    def test_variance_and_unsigned_values(self):
        """Variance uses squared distances without unsigned wraparound."""
        X = np.array([[0], [10]], dtype=np.uint8)
        self.assertEqual(variance(X, np.array([[9]], dtype=np.uint8)), 82)
        self.assertIsNone(variance(X, np.ones((2, 2))))

    def test_optimum_k_range_and_deltas(self):
        """All requested sizes run and use the first variance as baseline."""
        X = restaurant_clusters()
        results, changes = optimum_k(X, kmax=5)
        self.assertEqual(len(results), 5)
        self.assertEqual(changes[0], 0)
        self.assertAlmostEqual(changes[-1], 150059.18397, places=4)
        self.assertEqual(optimum_k(X, 2, 2), (None, None))
        self.assertEqual(optimum_k(X, 1, 251), (None, None))
        results, _ = optimum_k(np.array([[0.], [1.]]))
        self.assertEqual(len(results), 2)

    def test_gmm_initialization(self):
        """GMM starts with equal priors and one identity per centroid."""
        pi, m, S = initialize_gmm(mixture_data(), 4)
        np.testing.assert_allclose(pi, 0.25)
        np.testing.assert_allclose(S, np.tile(np.eye(2), (4, 1, 1)))
        np.testing.assert_allclose(m[0], [54.73711515, 31.81393242])

    def test_pdf_against_scipy(self):
        """Correlated Gaussian densities match an independent library."""
        m = np.array([12, 30, 10])
        S = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
        np.random.seed(0)
        X = np.random.multivariate_normal(m, S, 100)
        np.testing.assert_allclose(pdf(X, m, S),
                                   multivariate_normal.pdf(X, m, S))

    def test_pdf_floor_and_invalid_covariance(self):
        """Tiny densities are floored; invalid covariance is rejected."""
        np.testing.assert_array_equal(pdf(np.array([[1e6]]), np.zeros(1),
                                          np.eye(1)), [1e-300])
        X, m = np.zeros((2, 2)), np.zeros(2)
        for S in (np.zeros((2, 2)), -np.eye(2), np.ones((1, 1)),
                  np.array([[1., 2.], [0., 1.]])):
            self.assertIsNone(pdf(X, m, S))

    def test_expectation_against_scipy(self):
        """Bayes responsibilities and mixture likelihood agree with SciPy."""
        X = np.array([[-1.], [0.], [3.]])
        pi, m = np.array([0.3, 0.7]), np.array([[-1.], [2.]])
        S = np.ones((2, 1, 1))
        weighted = np.array([pi[i] * multivariate_normal.pdf(X, m[i], S[i])
                             for i in range(2)])
        g, likelihood = expectation(X, pi, m, S)
        np.testing.assert_allclose(g, weighted / weighted.sum(axis=0))
        self.assertAlmostEqual(likelihood, np.log(weighted.sum(axis=0)).sum())

    def test_expectation_validation_and_zero_prior(self):
        """Invalid priors fail; a zero-prior cluster receives zero mass."""
        X, m, S = np.array([[0.], [1.]]), np.zeros((2, 1)), np.ones((2, 1, 1))
        for pi in (np.array([0.5, 0.4]), np.array([-1., 2.]),
                   np.array([np.nan, 1.]), [0.5, 0.5]):
            self.assertEqual(expectation(X, pi, m, S), (None, None))
        g, _ = expectation(X, np.array([1., 0.]), m, S)
        np.testing.assert_array_equal(g, [[1., 1.], [0., 0.]])

    def test_maximization_hand_calculation(self):
        """Soft counts produce the specified weighted means and variance."""
        X = np.array([[0.], [2.], [4.]])
        g = np.array([[1., 0.5, 0.], [0., 0.5, 1.]])
        pi, m, S = maximization(X, g)
        np.testing.assert_allclose(pi, [0.5, 0.5])
        np.testing.assert_allclose(m[:, 0], [2 / 3, 10 / 3])
        np.testing.assert_allclose(S[:, 0, 0], [8 / 9, 8 / 9])
        self.assertEqual(maximization(X, np.zeros((2, 3))),
                         (None, None, None))
        self.assertEqual(maximization(X, np.array([[1., 1., 1.],
                                                  [0., 0., 0.]])),
                         (None, None, None))

    def test_em_matches_course_example(self):
        """EM recovers four clusters and the correct stopping step."""
        X = mixture_data()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pi, m, S, g, likelihood = em(X, 4, 150, verbose=True)
        self.assertAlmostEqual(likelihood, -94439.93889004056, places=5)
        np.testing.assert_allclose(X.shape[0] * pi,
                                   [761.03239903, 747.62391034,
                                    1005.60275934, 9985.74093129], atol=1e-4)
        np.testing.assert_allclose(g.sum(axis=0), 1)
        self.assertIn('Log Likelihood after 52 iterations:', output.getvalue())
        final_g, final_l = expectation(X, pi, m, S)
        np.testing.assert_allclose(g, final_g)
        self.assertAlmostEqual(likelihood, final_l)

    def test_maximization_signed_normalized_weights(self):
        """The course checker also evaluates the formula with signed weights."""
        X = np.array([[0.], [2.], [4.]])
        g = np.array([[-1., 0.5, 0.], [2., 0.5, 1.]])
        pi, m, S = maximization(X, g)
        np.testing.assert_allclose(pi, [-1 / 6, 7 / 6])
        np.testing.assert_allclose(m[:, 0], [-2, 10 / 7])
        np.testing.assert_allclose(S[:, 0, 0], [-8, 152 / 49])

    def test_em_iteration_limit_and_validation(self):
        """Returned posteriors match final parameters even at the limit."""
        X = mixture_data()
        pi, m, S, g, likelihood = em(X, 4, iterations=1)
        expected_g, expected_l = expectation(X, pi, m, S)
        np.testing.assert_allclose(g, expected_g)
        self.assertAlmostEqual(likelihood, expected_l)
        for options in ({'tol': -1.}, {'tol': np.nan}, {'tol': 1},
                        {'iterations': 0}, {'verbose': 1}):
            self.assertEqual(em(X, 4, **options), (None,) * 5)

    def test_em_verbose_rounding(self):
        """The single-Gaussian MLE prints rounded values without padding."""
        X = np.arange(7, dtype=float).reshape(-1, 1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pi, m, S, g, likelihood = em(X, 1, verbose=True)
        np.testing.assert_allclose(S, [[[4.]]])
        self.assertAlmostEqual(likelihood, -14.784599996352325)
        self.assertEqual(output.getvalue().splitlines()[-1],
                         'Log Likelihood after 2 iterations: -14.7846')

    def test_bic_selects_course_cluster_count(self):
        """BIC balances likelihood against full-covariance parameter count."""
        X = mixture_data()
        best_k, result, likelihoods, scores = bic(X, kmax=5)
        self.assertEqual(best_k, 4)
        self.assertEqual(result[0].shape, (4,))
        k = np.arange(1, 6)
        np.testing.assert_allclose(scores, (6 * k - 1) * np.log(len(X))
                                   - 2 * likelihoods)
        self.assertAlmostEqual(scores[3], 189096.84790787, places=4)

    def test_bic_requires_comparison_and_validates_ranges(self):
        """The course requires at least two different sizes to compare."""
        X = np.array([[0.], [1.], [2.], [3.]])
        for options in ({'kmax': 1}, {'kmin': 2, 'kmax': 1}, {'kmax': 5},
                        {'kmin': True}, {'kmin': 2, 'kmax': 2}):
            self.assertEqual(bic(X, **options), (None,) * 4)

    def test_sklearn_wrappers(self):
        """Library wrappers return the required arrays and scalar BIC."""
        X = restaurant_clusters()
        C, labels = sklearn_kmeans(X, 5)
        self.assertEqual(C.shape, (5, 2))
        self.assertEqual(labels.shape, (250,))
        pi, m, S, labels, score = gmm(X, 5)
        self.assertEqual(pi.shape, (5,))
        self.assertEqual(m.shape, (5, 2))
        self.assertEqual(S.shape, (5, 2, 2))
        self.assertEqual(labels.shape, (250,))
        self.assertTrue(np.isfinite(score))

    def test_agglomerative_plot_and_labels(self):
        """Ward linkage draws a dendrogram with five distance-cut groups."""
        X = restaurant_clusters()
        with patch.object(plt, 'show') as show:
            labels = agglomerative(X, 100)
        show.assert_called_once()
        self.assertEqual(labels.shape, (250,))
        self.assertEqual(len(np.unique(labels)), 5)
        self.assertGreater(len(plt.gca().collections), 0)
        plt.close('all')

    def test_course_source_restrictions(self):
        """Check loops, imports, executable bits, and Python 3.5 syntax."""
        loop_limits = [0, 2, 0, 2, 0, 0, 1, 1, 1, 1]
        for file in PROJECT.glob('[0-9]*.py'):
            source = file.read_text()
            tree = ast.parse(source, feature_version=(3, 5))
            number = int(file.name.split('-')[0])
            self.assertTrue(source.startswith('#!/usr/bin/env python3\n'))
            self.assertTrue(source.endswith('\n'))
            self.assertTrue(os.access(str(file), os.X_OK))
            self.assertTrue(ast.get_docstring(tree))
            functions = [n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef)]
            self.assertTrue(all(ast.get_docstring(n) for n in functions))
            imports = [alias.name for n in ast.walk(tree)
                       if isinstance(n, ast.Import) for alias in n.names]
            allowed = (['numpy'] if number < 10 else
                       ['sklearn.cluster'] if number == 10 else
                       ['sklearn.mixture'] if number == 11 else
                       ['scipy.cluster.hierarchy', 'matplotlib.pyplot'])
            self.assertEqual(imports, allowed)
            if number < 10:
                loops = [n for n in ast.walk(tree)
                         if isinstance(n, (ast.For, ast.While,
                                           ast.comprehension))]
                self.assertLessEqual(len(loops), loop_limits[number])
            if number in (0, 1):
                draws = [n for n in ast.walk(tree)
                         if isinstance(n, ast.Call) and
                         isinstance(n.func, ast.Attribute) and
                         n.func.attr == 'uniform']
                self.assertEqual(len(draws), 1 if number == 0 else 2)
            if number == 5:
                self.assertFalse(any(isinstance(n, ast.Attribute) and
                                     n.attr in ('diag', 'diagonal')
                                     for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()

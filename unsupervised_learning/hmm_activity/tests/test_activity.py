"""Adversarial checks use synthetic data, never coursework measurement claims."""

import csv
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_hmm import GaussianHMM
from features import (activity_metrics, fit_scaler, read_recording, resample,
                      spectral_features, window_features)
from run_activity import run


class ActivityTests(unittest.TestCase):
    def test_forward_viterbi_and_posterior_against_every_path(self):
        model = GaussianHMM([0.6, 0.4], [[0.7, 0.3], [0.2, 0.8]],
                            [[-1], [1]], [[0.5], [0.8]])
        x = np.array([[-0.8], [0.2], [1.3]])
        b = model._emissions(x)
        paths, scores = [], []
        for path in product(range(2), repeat=3):
            value = np.log(model.initial[path[0]]) + b[0, path[0]]
            for t in range(1, 3):
                value += np.log(model.transition[path[t - 1], path[t]]) + b[t, path[t]]
            paths.append(path)
            scores.append(value)
        np.testing.assert_allclose(model.score(x), np.logaddexp.reduce(scores))
        path, score = model.decode(x)
        np.testing.assert_array_equal(path, paths[np.argmax(scores)])
        self.assertAlmostEqual(score, max(scores))
        _, gamma, xi = model._expectation(x)
        probabilities = np.exp(np.array(scores) - np.logaddexp.reduce(scores))
        expected = np.zeros_like(gamma)
        for path, probability in zip(paths, probabilities):
            for t, state in enumerate(path):
                expected[t, state] += probability
        np.testing.assert_allclose(gamma, expected)
        self.assertAlmostEqual(xi.sum(), 2)

    def test_baum_welch_likelihood_and_recording_boundaries(self):
        rng = np.random.RandomState(7)
        seq = [rng.normal(center, 0.25, size=(20, 2))
               for center in (-3, -3, 3, 3)]
        y = [np.full(20, i // 2, dtype=int) for i in range(4)]
        model = GaussianHMM.from_labels(seq, y, n_states=2).fit(seq)
        self.assertTrue(model.converged)
        self.assertTrue(np.all(np.diff(model.history) >= -1e-7))
        # Concatenating files would add a spurious -3 to +3 transition.
        self.assertLess(model.transition[0, 1], 1e-6)
        np.testing.assert_array_equal(model.decode(seq[0])[0], np.zeros(20))
        np.testing.assert_array_equal(model.decode(seq[-1])[0], np.ones(20))

    def test_single_sample_sequences_keep_transition_rows(self):
        model = GaussianHMM([0.5, 0.5], [[0.8, 0.2], [0.1, 0.9]],
                            [[-1], [1]], [[1], [1]])
        previous = model.transition.copy()
        model.fit([np.array([[-1.0]]), np.array([[1.0]])])
        np.testing.assert_array_equal(model.transition, previous)
        self.assertTrue(np.isfinite(model.score(np.ones((3000, 1)))))

    def test_constant_features_and_fft_known_frequency(self):
        hz = 100
        signal = np.sin(2 * np.pi * 3 * np.arange(200) / hz)
        frequency, energy = spectral_features(signal, hz)
        self.assertAlmostEqual(frequency, 3)
        self.assertAlmostEqual(energy, 0.5, places=3)
        values = np.ones((100, 6))
        x, starts = window_features(values, 20)
        self.assertEqual(x.shape, (4, 11))
        mean, scale = fit_scaler([x])
        np.testing.assert_allclose((x - mean) / scale, 0, atol=1e-12)
        np.testing.assert_array_equal(starts, [0, 1, 2, 3])
        self.assertTrue(np.isfinite(x).all())

    def test_resampling_suppresses_alias_and_preserves_low_frequency(self):
        t = np.arange(1000) / 100
        low = np.sin(2 * np.pi * 2 * t)
        high = np.sin(2 * np.pi * 32 * t)
        grid, values = resample(t, np.column_stack((low, high)), 20)
        self.assertLess(np.std(values[10:-10, 1]), 0.03)
        self.assertGreater(np.std(values[10:-10, 0]), 0.65)
        self.assertAlmostEqual(np.median(np.diff(grid)), 0.05)
        with self.assertRaises(ValueError):
            resample(t, np.column_stack((low, high)), 200)

    def test_metrics_use_correct_denominators_and_missing_class_is_undefined(self):
        m = activity_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(m['overall_accuracy'], 0.75)
        self.assertEqual(m['per_activity'][0]['sensitivity'], 0.5)
        self.assertEqual(m['per_activity'][0]['specificity'], 1)
        self.assertIsNone(m['per_activity'][2]['sensitivity'])
        self.assertEqual(m['per_activity'][1]['specificity'], 0.5)

    def test_bad_timestamps_and_duplicate_test_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [[i / 20, 0, 0, 9.8, 0, 0, 0] for i in range(120)]
            for split in ('train', 'test'):
                for activity in ('standing', 'walking', 'jumping', 'still'):
                    path = root / split / activity / 'sample.csv'
                    path.parent.mkdir(parents=True)
                    self.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, 'Duplicate sensor content'):
                run(root, root / 'results')
            self.assertFalse((root / 'results').exists())
            rows[30][0] = rows[29][0]
            self.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, 'Timestamps'):
                read_recording(path)

    def test_complete_pipeline_writes_measured_outputs_for_unseen_files(self):
        rng = np.random.RandomState(83)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            t = np.arange(120) / 20
            for split in ('train', 'test'):
                for label, activity in enumerate(('standing', 'walking', 'jumping', 'still')):
                    for index in range(2):
                        wave = np.sin(2 * np.pi * (label + 1) * t)
                        x = rng.normal(0, 0.01, (len(t), 6))
                        x[:, 2] += 9.8 + (label + 1) * 0.3 * wave
                        x[:, 5] += label + 0.2 * wave
                        path = root / split / activity / '{}.csv'.format(index)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        self.write_csv(path, np.column_stack((t, x)))
            result = run(root, root / 'results')
            self.assertTrue(result['training_converged'])
            self.assertEqual(result['train_files'], 8)
            self.assertEqual(result['test_files'], 8)
            self.assertGreater(result['overall_accuracy'], 0.9)
            for name in ('metrics.json', 'model.npz', 'predictions.csv',
                         'transition.png', 'emissions.png', 'confusion.png',
                         'convergence.png', 'recording_examples.png', 'decoded_007.png'):
                self.assertGreater((root / 'results' / name).stat().st_size, 0)

    @staticmethod
    def write_csv(path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
            writer.writerows(rows)


if __name__ == '__main__':
    unittest.main()

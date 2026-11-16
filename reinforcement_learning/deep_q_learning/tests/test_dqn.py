#!/usr/bin/env python3
"""Run real Breakout transitions, gradient updates, and HDF5 round trips."""

import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from keras import backend as K
from keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
AtariProcessor = importlib.import_module('dqn_utils').AtariProcessor


class IntegrationTests(unittest.TestCase):
    """Check real runtime behavior rather than substituted dependencies."""

    def test_processing_preserves_evaluation_rewards(self):
        processor = AtariProcessor()
        frame = processor.process_observation(
            np.full((210, 160, 3), 255, dtype=np.uint8))
        self.assertEqual(frame.shape, (84, 84))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(processor.process_reward(7), 1)
        self.assertEqual(AtariProcessor(False).process_reward(7), 7)
        batch = processor.process_state_batch(np.array([[frame] * 4]))
        self.assertEqual(batch.dtype, np.float32)
        self.assertTrue(np.all(batch == 1))

    def test_real_training_save_reload_and_greedy_play(self):
        training = importlib.import_module('train')
        playing = importlib.import_module('play')
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'policy.h5')
            agent = training.train(steps=64, output=path, warmup=16,
                                   memory_limit=1000)
            self.assertIsInstance(agent, DQNAgent)
            self.assertIsInstance(agent.memory, SequentialMemory)
            self.assertIsInstance(agent.policy.inner_policy, EpsGreedyQPolicy)
            self.assertIsInstance(agent.test_policy, GreedyQPolicy)
            self.assertGreater(agent.memory.nb_entries, 32)
            self.assertGreater(
                K.get_value(agent.trainable_model.optimizer.iterations), 0)
            restored = load_model(path, compile=False)
            batch = np.zeros((1, 4, 84, 84), dtype=np.float32)
            np.testing.assert_allclose(agent.model.predict(batch),
                                       restored.predict(batch))
            scores = playing.play(path, episodes=1, visualize=False,
                                  max_steps=64)
            self.assertEqual(len(scores), 1)
            self.assertTrue(np.isfinite(scores[0]))


if __name__ == '__main__':
    unittest.main()

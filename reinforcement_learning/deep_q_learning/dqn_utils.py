#!/usr/bin/env python3
"""Shared pixel processing and network construction for Atari DQN.

The architecture and preprocessing follow keras-rl's Atari example.
"""

import h5py
import numpy as np
from PIL import Image
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


class AtariProcessor(Processor):
    """Store grayscale bytes and normalize only sampled minibatches."""

    def __init__(self, clip_rewards=True):
        """Clip rewards during learning, preserve game scores in evaluation."""
        self.clip_rewards = clip_rewards

    def process_observation(self, observation):
        """Resize one RGB screen to an 84-by-84 grayscale image."""
        image = Image.fromarray(observation).resize((84, 84)).convert('L')
        return np.asarray(image, dtype=np.uint8)

    def process_state_batch(self, batch):
        """Normalize four-frame replay states to the unit interval."""
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clip learning rewards without changing reported game scores."""
        return np.clip(reward, -1.0, 1.0) if self.clip_rewards else reward


def build_model(nb_actions):
    """Return a convolutional action-value network for four Atari frames."""
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                     data_format='channels_last'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu',
                     data_format='channels_last'))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def save_policy(model, path):
    """Save a full policy with byte metadata readable by legacy Keras.

    HDF5 3 returns variable-length string attributes as text; Keras 2.2.5
    expects bytes. Fixed byte attributes also work with the course's h5py.
    """
    model.save(path, include_optimizer=False)
    with h5py.File(path, 'r+') as policy:
        groups = [policy]
        policy.visititems(lambda name, item: groups.append(item)
                          if isinstance(item, h5py.Group) else None)
        for group in groups:
            for key in list(group.attrs):
                value = group.attrs[key]
                if isinstance(value, str):
                    del group.attrs[key]
                    group.attrs[key] = np.bytes_(value.encode('utf-8'))


def build_agent(model, nb_actions, policy, memory_limit=100000,
                warmup=50000, clip_rewards=True):
    """Compile DQN with replay, a fixed target network, and greedy testing."""
    memory = SequentialMemory(limit=memory_limit, window_length=4)
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                     policy=policy, test_policy=GreedyQPolicy(),
                     processor=AtariProcessor(clip_rewards),
                     nb_steps_warmup=warmup,
                     target_model_update=10000, train_interval=4,
                     gamma=0.99, delta_clip=1.0)
    agent.compile(Adam(lr=0.00025), metrics=['mae'])
    return agent

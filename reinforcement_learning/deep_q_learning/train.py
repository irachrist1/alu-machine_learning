#!/usr/bin/env python3
"""Train a keras-rl DQN to play Breakout and save its policy network."""

import argparse
import gym
import numpy as np
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from dqn_utils import build_agent, build_model, save_policy


def train(steps=1750000, output='policy.h5', seed=123, warmup=50000,
          memory_limit=100000, env_name='BreakoutDeterministic-v4'):
    """Learn from Atari transitions and save the full final Keras model."""
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    try:
        model = build_model(env.action_space.n)
        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=0.1,
            value_test=0.0, nb_steps=1000000)
        agent = build_agent(model, env.action_space.n, policy,
                            memory_limit, warmup)
        callbacks = [FileLogger(output + '.training.json', interval=100),
                     ModelIntervalCheckpoint(output + '.checkpoint-{step}',
                                             interval=250000)]
        agent.fit(env, nb_steps=steps, visualize=False, verbose=2,
                  callbacks=callbacks)
        save_policy(model, output)
        return agent
    finally:
        env.close()


def main():
    """Read reproducible training parameters from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1750000)
    parser.add_argument('--output', default='policy.h5')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--warmup', type=int, default=50000)
    parser.add_argument('--memory-limit', type=int, default=100000)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0 or args.memory_limit < 6:
        parser.error('steps and replay capacity must be positive')
    train(args.steps, args.output, args.seed, args.warmup, args.memory_limit)


if __name__ == '__main__':
    main()

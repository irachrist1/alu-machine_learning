#!/usr/bin/env python3
"""Load the saved policy and display greedy Atari Breakout episodes."""

import argparse
import json
import gym
import numpy as np
from keras.models import load_model
from rl.policy import GreedyQPolicy
from dqn_utils import build_agent


def play(policy_path='policy.h5', episodes=10, seed=123, visualize=True,
         env_name='BreakoutDeterministic-v4', max_steps=10000):
    """Evaluate the stored network and report original, unclipped scores."""
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    try:
        model = load_model(policy_path, compile=False)
        if model.output_shape[-1] != env.action_space.n:
            raise ValueError('The saved policy has the wrong action count')
        agent = build_agent(model, env.action_space.n, GreedyQPolicy(),
                            memory_limit=6, warmup=0, clip_rewards=False)
        history = agent.test(env, nb_episodes=episodes, visualize=visualize,
                             nb_max_episode_steps=max_steps, verbose=2,
                             nb_max_start_steps=30,
                             start_step_policy=lambda observation: 1)
        scores = [float(value) for value in history.history['episode_reward']]
        print(json.dumps({'scores': scores, 'mean': float(np.mean(scores)),
                          'maximum': max(scores)}, sort_keys=True))
        return scores
    finally:
        env.close()


def main():
    """Show play by default; permit headless evaluation for verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', default='policy.h5')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error('episodes and maximum episode steps must be positive')
    play(args.policy, args.episodes, args.seed, not args.headless,
         max_steps=args.max_steps)


if __name__ == '__main__':
    main()

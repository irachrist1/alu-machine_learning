#!/usr/bin/env python3
"""Load the Frozen Lake environment used in the Q-learning exercises."""

import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


def _random_map():
    """Generate an 8-by-8 map with a safe route from start to goal."""
    while True:
        board = np.random.choice(['F', 'H'], (8, 8), p=[0.8, 0.2])
        board[0, 0], board[-1, -1] = 'S', 'G'
        frontier = [(0, 0)]
        visited = set()
        while frontier:
            row, col = frontier.pop()
            if (row, col) in visited:
                continue
            visited.add((row, col))
            if board[row, col] == 'G':
                return board
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] != 'H':
                    frontier.append((nr, nc))


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Return a Frozen Lake environment with the given map and dynamics."""
    if desc is None and map_name is None:
        desc = _random_map()
    return FrozenLakeEnv(desc=desc, map_name=map_name,
                         is_slippery=is_slippery)

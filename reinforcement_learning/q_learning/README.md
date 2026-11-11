# Tabular Q-learning

Frozen Lake exercises for the ALU Machine Learning Techniques II intranet.
The assignment targets Python 3.5, NumPy 1.15, and Gym 0.7's four-value
`step` API and `FrozenLake-v0` environment.

Gym 0.7's `gym.make` does not accept map keyword arguments, and its
`FrozenLakeEnv` requires an explicit map. The loader constructs that class
directly and generates a reachable random 8-by-8 map when neither map
argument is supplied, as required by the task.

| File | Responsibility |
| --- | --- |
| `0-load_env.py` | Load a custom, named, or random Frozen Lake map. |
| `1-q_init.py` | Allocate zero values for all state-action pairs. |
| `2-epsilon_greedy.py` | Sample exploration or choose the highest Q-value. |
| `3-q_learning.py` | Update Q-values with the Bellman target and decay exploration. |
| `4-play.py` | Render an episode while choosing actions greedily. |

For a nonterminal transition, training applies
`Q[s, a] += alpha * (reward + gamma * max(Q[next_s]) - Q[s, a])`.
For terminal transitions the target is the immediate reward. Holes receive
-1 during training; evaluation reports the environment's original rewards.

After episode `i`, exploration becomes
`min_epsilon + (initial_epsilon - min_epsilon) * exp(-epsilon_decay * i)`.
The table is updated in place, and training returns the rewards for each
episode so learning progress can be evaluated separately from one lucky run.

The implementation was prepared with AI assistance. Review the functions and
their learning assumptions before using this work in a course assessment.

# Temporal Difference

ALU intranet project 2418, due 27 October 2026 at 23:58.

| File | Algorithm | Update target |
|---|---|---|
| `0-monte_carlo.py` | First-visit Monte Carlo prediction | Discounted rewards through the recorded episode horizon |
| `1-td_lambtha.py` | TD(lambda) with accumulating state traces | Reward plus discounted next-state value |
| `2-sarsa_lambtha.py` | SARSA(lambda) with accumulating state-action traces | Reward plus discounted value of the next sampled action |

All functions update the supplied floating-point value table in place and
return it. They use the specified Gym 0.7 environment interface: `reset()`
returns a state and `step(action)` returns `(state, reward, done, info)`.

Terminal transitions use the reward alone. An arbitrary initial terminal
value must not become an extra reward. Traces reset between episodes and
decay by `gamma * lambtha`. At lambda zero only the current state or
state-action pair receives the update. SARSA uses the action sampled by its
epsilon-greedy behavior policy rather than a maximum over all next actions.
Epsilon decays exponentially between episodes, starting at the requested
value in episode zero.

Monte Carlo updates only the first occurrence of a state in each episode.
A trajectory stopped at `max_steps` uses its observed finite-horizon return.
TD and SARSA still bootstrap at a nonterminal cutoff.

```sh
python3 -m unittest discover -s reinforcement_learning/temporal_difference/tests -v
python3 -m pycodestyle reinforcement_learning/temporal_difference/*.py
```

The tests use deterministic environments with hand-calculated expected
returns and updates. They cover revisits, terminal values, trace decay and
reset, and on-policy action selection. The published stochastic FrozenLake
examples have not been reproduced in the original Gym 0.7 / NumPy 1.15
environment; manual platform checklist items remain unchecked. These files
are prepared locally and have not been pushed or submitted.

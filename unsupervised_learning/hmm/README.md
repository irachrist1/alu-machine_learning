# Hidden Markov Models

NumPy implementations for ALU project 2331. Each numbered file exposes the
function and return shape specified by the assignment.

| File | Method | Reasoning |
|---|---|---|
| `0-markov_chain.py` | `s @ P**t` (matrix power) | Each multiplication advances the distribution by one transition. |
| `1-regular.py` | Positive-power reachability and stationary linear system | A stationary vector alone does not establish regularity; periodic and disconnected chains must be rejected. |
| `2-absorbing.py` | Reverse reachability from absorbing states | Every state must have a positive-probability route to an absorbing state. |
| `3-forward.py` | Sum-product recurrence | Sum all predecessor paths to obtain the observation likelihood. |
| `4-viterbi.py` | Log-space max-product recurrence and backtracking | Keep the best predecessor per state, then trace the best complete path. |
| `5-backward.py` | Reverse sum-product recurrence | Sum all possible suffixes from each state. |
| `6-baum_welch.py` | Scaled forward/backward EM | Expected transition and emission counts update the model without requiring observed hidden states. |

The forward trellis stores joint probabilities `P(X[:t+1], Z[t]=i)`, not a
normalized posterior. The backward trellis stores
`P(X[t+1:] | Z[t]=i)` conditional on the current state. Multiplying corresponding
entries and summing over states gives the same full-sequence likelihood at
every time step.

The public forward and backward arrays remain unscaled to match the grader's
specified outputs. Their probabilities can underflow on very long sequences.
Viterbi uses logarithms so its path survives underflow, although its returned
probability can still round to zero. Baum-Welch normalizes each forward step,
uses the same scaling in the backward pass, and stops when the change in
log-likelihood is below `1e-8` or the iteration cap is reached. The supplied
initial distribution stays fixed because this task returns only transitions
and emissions. Unvisited states retain their previous probability rows.

Run the regression suite from the repository root:

```sh
python3 -m unittest discover -s unsupervised_learning/hmm/tests -v
python3 -m pycodestyle unsupervised_learning/hmm/*.py
```

Tests enumerate all hidden paths for small random HMMs, reproduce the course's
Baum-Welch output, check forward/backward consistency, cover periodic and
disconnected chains, and exercise zero-probability and underflow cases.
The earlier ice-cream exercise is preserved as a regression case:
`P(3,1,3)=0.028562`, `P(3,1,3,HHH)=0.009216`, best path `HCH`, probability
`0.0128`.

These files are prepared locally. The intranet project deadline was
20 October 2026 at 23:58; local test success does not establish a submitted
grade or late-work acceptance.

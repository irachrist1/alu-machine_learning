"""Diagonal Gaussian HMM, with independent recording boundaries."""

from itertools import permutations

import numpy as np


class GaussianHMM:
    """Fit continuous emissions with log-domain Baum–Welch and decode Viterbi."""

    def __init__(self, initial, transition, means, variances, floor=1e-3):
        self.initial = np.array(initial, dtype=float, copy=True)
        self.transition = np.array(transition, dtype=float, copy=True)
        self.means = np.array(means, dtype=float, copy=True)
        self.variances = np.array(variances, dtype=float, copy=True)
        self.floor = float(floor)
        k = self.initial.size
        if (self.initial.shape != (k,) or k < 1
                or self.transition.shape != (k, k)
                or self.means.ndim != 2 or self.means.shape[0] != k
                or self.means.shape[1] < 1
                or self.variances.shape != self.means.shape
                or not np.isfinite(self.floor) or self.floor <= 0):
            raise ValueError("Incompatible HMM dimensions or variance floor")
        for value in (self.initial, self.transition, self.means, self.variances):
            if not np.isfinite(value).all():
                raise ValueError("HMM parameters must be finite")
        if (np.any(self.initial < 0) or np.any(self.transition < 0)
                or not np.isclose(self.initial.sum(), 1)
                or not np.allclose(self.transition.sum(axis=1), 1)
                or np.any(self.variances < self.floor)):
            raise ValueError("Invalid probabilities or variances")
        self.history = []
        self.converged = False

    @classmethod
    def from_labels(cls, sequences, labels, n_states=4, floor=1e-3):
        """Initialize activity identities from training labels only."""
        x = np.vstack(sequences)
        y = np.concatenate(labels)
        if (len(sequences) != len(labels)
                or any(len(a) != len(b) for a, b in zip(sequences, labels))
                or set(np.unique(y)) != set(range(n_states))):
            raise ValueError("Training data must cover every activity")
        means = np.array([x[y == s].mean(axis=0) for s in range(n_states)])
        var = np.array([x[y == s].var(axis=0) for s in range(n_states)])
        initial = np.full(n_states, 0.01)
        transitions = np.full((n_states, n_states), 0.01)
        for sequence in labels:
            initial[sequence[0]] += 1
            # Separate files are not evidence of a transition between activities.
            np.add.at(transitions, (sequence[:-1], sequence[1:]), 1)
        return cls(initial / initial.sum(),
                   transitions / transitions.sum(axis=1, keepdims=True),
                   means, np.maximum(var, floor), floor)

    def _emissions(self, x):
        x = np.asarray(x, dtype=float)
        if (x.ndim != 2 or x.shape[0] == 0
                or x.shape[1] != self.means.shape[1]
                or not np.isfinite(x).all()):
            raise ValueError("Expected a nonempty finite time-by-feature array")
        return -0.5 * (np.log(2 * np.pi * self.variances).sum(axis=1)
                       + ((x[:, None] - self.means) ** 2
                          / self.variances).sum(axis=2))

    def _expectation(self, x):
        b = self._emissions(x)
        with np.errstate(divide="ignore"):
            a = np.log(self.transition)
            initial = np.log(self.initial)
        forward = np.empty_like(b)
        forward[0] = initial + b[0]
        for t in range(1, len(b)):
            forward[t] = b[t] + np.logaddexp.reduce(
                forward[t - 1, :, None] + a, axis=0)
        score = float(np.logaddexp.reduce(forward[-1]))
        if not np.isfinite(score):
            raise ValueError("Sequence has no finite likelihood")
        backward = np.zeros_like(b)
        for t in range(len(b) - 2, -1, -1):
            backward[t] = np.logaddexp.reduce(
                a + b[t + 1] + backward[t + 1], axis=1)
        log_gamma = forward + backward
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        transitions = np.zeros_like(self.transition)
        for t in range(len(b) - 1):
            log_xi = (forward[t, :, None] + a + b[t + 1]
                      + backward[t + 1])
            transitions += np.exp(log_xi - np.logaddexp.reduce(log_xi.ravel()))
        return score, gamma, transitions

    def score(self, x):
        """Return log likelihood, avoiding probability underflow."""
        return self._expectation(x)[0]

    def fit(self, sequences, max_iter=200, tolerance=1e-6):
        """Maximize joint likelihood across files without joining boundaries."""
        sequences = [np.asarray(x, dtype=float) for x in sequences]
        if not sequences or max_iter < 1 or tolerance <= 0:
            raise ValueError("Training requires sequences and positive limits")
        self.history = []
        self.converged = False
        for iteration in range(max_iter + 1):
            expected = [self._expectation(x) for x in sequences]
            score = sum(item[0] for item in expected)
            if self.history:
                change = score - self.history[-1]
                if change < -1e-7 * (1 + abs(self.history[-1])):
                    raise ArithmeticError("Baum–Welch decreased log likelihood")
                if abs(change) <= tolerance * (1 + abs(self.history[-1])):
                    self.history.append(score)
                    self.converged = True
                    break
            self.history.append(score)
            if iteration == max_iter:
                break
            gamma = np.vstack([item[1] for item in expected])
            x = np.vstack(sequences)
            weight = gamma.sum(axis=0)
            live = weight > 1e-12
            means = self.means.copy()
            means[live] = (gamma[:, live].T @ x) / weight[live, None]
            second = (gamma[:, live].T @ (x ** 2)) / weight[live, None]
            self.variances[live] = np.maximum(
                second - means[live] ** 2, self.floor)
            self.means = means
            starts = sum(item[1][0] for item in expected)
            self.initial = starts / starts.sum()
            counts = sum(item[2] for item in expected)
            totals = counts.sum(axis=1)
            used = totals > 1e-12
            self.transition[used] = counts[used] / totals[used, None]
        return self

    def decode(self, x):
        """Return the most likely complete state path and its log density."""
        b = self._emissions(x)
        with np.errstate(divide="ignore"):
            a = np.log(self.transition)
            values = np.log(self.initial) + b[0]
        back = np.zeros(b.shape, dtype=int)
        for t in range(1, len(b)):
            candidates = values[:, None] + a
            back[t] = candidates.argmax(axis=0)
            values = candidates.max(axis=0) + b[t]
        path = np.empty(len(b), dtype=int)
        path[-1] = values.argmax()
        score = float(values[path[-1]])
        for t in range(len(b) - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        return path, score

    def align_to_training_labels(self, sequences, labels):
        """Resolve state permutation using training data, never evaluation labels."""
        k = len(self.initial)
        counts = np.zeros((k, k), dtype=int)
        for x, y in zip(sequences, labels):
            np.add.at(counts, (y, self.decode(x)[0]), 1)
        order = max(permutations(range(k)),
                    key=lambda p: sum(counts[i, p[i]] for i in range(k)))
        order = np.array(order)
        self.initial = self.initial[order]
        self.transition = self.transition[np.ix_(order, order)]
        self.means = self.means[order]
        self.variances = self.variances[order]
        return order

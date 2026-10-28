"""Run a recording-level train/test experiment after real data is collected."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features import (ACTIVITIES, FEATURES, activity_metrics, fit_scaler,
                      read_recording, resample, window_features)
from gaussian_hmm import GaussianHMM


def load_split(root, split, hz, time_unit):
    """Keep each independently recorded file as a separate HMM sequence."""
    records = []
    for label, activity in enumerate(ACTIVITIES):
        for path in sorted((root / split / activity).glob("*.csv")):
            t, raw, quality = read_recording(path, time_unit)
            grid, sampled = resample(t, raw, hz)
            x, starts = window_features(sampled, hz)
            records.append({"path": str(path.relative_to(root)), "label": label,
                            "x": x, "starts": starts, "raw_t": grid,
                            "raw": sampled, **quality})
    if not records:
        raise ValueError("No CSV recordings in {}".format(root / split))
    return records


def plot_matrix(matrix, labels, title, path):
    """Save a labelled matrix with readable counts or probabilities."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, "{:.3g}".format(matrix[i, j]), ha="center")
    ax.set_title(title)
    ax.set_xlabel("Next / predicted activity")
    ax.set_ylabel("Current / true activity")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(data_root, output, hz=20, time_unit="s", max_iter=200):
    """Train, decode unseen files, and write measured results without placeholders."""
    data_root, output = Path(data_root), Path(output)
    train = load_split(data_root, "train", hz, time_unit)
    test = load_split(data_root, "test", hz, time_unit)
    if set(r["label"] for r in train) != set(range(4)):
        raise ValueError("Training needs all four activities")
    if len(test) < 2:
        raise ValueError("At least two genuinely unseen test recordings are required")
    digests = [r["digest"] for r in train + test]
    if len(set(digests)) != len(digests):
        raise ValueError("Duplicate sensor content: inspect repeated files and leakage")
    mean, scale = fit_scaler([r["x"] for r in train])
    train_x = [(r["x"] - mean) / scale for r in train]
    train_y = [np.full(len(r["x"]), r["label"], dtype=int) for r in train]
    model = GaussianHMM.from_labels(train_x, train_y)
    model.fit(train_x, max_iter=max_iter)
    if not model.converged:
        raise RuntimeError("Baum–Welch reached its limit; inspect convergence before reporting")
    model.align_to_training_labels(train_x, train_y)
    truth, predictions, decoded = [], [], []
    for record in test:
        states, log_density = model.decode((record["x"] - mean) / scale)
        predictions.extend(states.tolist())
        truth.extend([record["label"]] * len(states))
        decoded.append((record, states, log_density))
    metrics = activity_metrics(truth, predictions)
    metrics.update({"test_files": len(test), "train_files": len(train),
                    "rate_hz": hz, "window_seconds": 2, "hop_seconds": 1,
                    "training_converged": model.converged,
                    "training_log_likelihood": model.history,
                    "feature_names": list(FEATURES),
                    "file_quality": [{k: v for k, v in r.items()
                                      if k not in {"x", "starts", "raw_t", "raw"}}
                                     for r in train + test]})
    # A fresh directory prevents a rerun from silently replacing earlier evidence.
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    np.savez(output / "model.npz", initial=model.initial, transition=model.transition,
             means=model.means, variances=model.variances,
             scaler_mean=mean, scaler_scale=scale, activities=ACTIVITIES,
             features=FEATURES, rate_hz=hz)
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "window_start_s", "true_activity", "predicted_activity"])
        for record, states, _ in decoded:
            for start, state in zip(record["starts"], states):
                writer.writerow([record["path"], start,
                                 ACTIVITIES[record["label"]], ACTIVITIES[state]])
    plot_matrix(model.transition, ACTIVITIES, "Learned transition probabilities",
                output / "transition.png")
    plot_matrix(np.array(metrics["confusion_matrix"]), ACTIVITIES,
                "Unseen-file confusion matrix (windows)", output / "confusion.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, dimension in zip(axes, [1, 4]):
        grid = np.linspace(model.means[:, dimension].min() - 3,
                           model.means[:, dimension].max() + 3, 500)
        for state, activity in enumerate(ACTIVITIES):
            variance = model.variances[state, dimension]
            density = np.exp(-0.5 * (grid - model.means[state, dimension]) ** 2
                             / variance) / np.sqrt(2 * np.pi * variance)
            ax.plot(grid, density, label=activity)
        ax.set_title(FEATURES[dimension] + " emission density")
        ax.set_xlabel("Training-standardized feature")
        ax.set_ylabel("Probability density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "emissions.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(model.history)
    ax.set(xlabel="EM iteration", ylabel="Training log likelihood",
           title="Baum–Welch convergence")
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(9, 8))
    for state, ax in enumerate(axes):
        record = next(r for r in train if r["label"] == state)
        ax.plot(record["raw_t"], np.linalg.norm(record["raw"][:, :3], axis=1))
        ax.set(title=ACTIVITIES[state], xlabel="Seconds", ylabel="Acceleration magnitude")
    fig.tight_layout()
    fig.savefig(output / "recording_examples.png", dpi=150)
    plt.close(fig)
    for index, (record, states, _) in enumerate(decoded):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.step(record["starts"], states, where="post", label="Viterbi prediction")
        ax.axhline(record["label"], linestyle="--", color="black", label="Recorded activity")
        ax.set_yticks(range(4), ACTIVITIES)
        ax.set(xlabel="Window start (seconds)", title=record["path"])
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "decoded_{:03d}.png".format(index), dpi=150)
        plt.close(fig)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path, help="A new output directory")
    parser.add_argument("--hz", type=float, default=20)
    parser.add_argument("--time-unit", choices=["s", "ms", "us", "ns"], default="s")
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()
    result = run(args.data_root, args.output, args.hz, args.time_unit, args.max_iter)
    print(json.dumps({k: result[k] for k in ("overall_accuracy", "train_files", "test_files")}, indent=2))

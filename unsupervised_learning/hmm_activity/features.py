"""Validate timestamped sensor files, resample, and extract window features."""

import csv
import hashlib

import numpy as np

ACTIVITIES = ("standing", "walking", "jumping", "still")
SENSORS = ("ax", "ay", "az", "gx", "gy", "gz")
FEATURES = ("acc_mean", "acc_std", "acc_rms", "gyro_mean", "gyro_std",
            "gyro_rms", "acc_sma", "acc_peak_hz", "acc_spectral_energy",
            "gyro_peak_hz", "gyro_spectral_energy")


def read_recording(path, time_unit="s"):
    """Read a synchronized CSV; reject unusable timing instead of hiding it."""
    units = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if time_unit not in units:
        raise ValueError("Specify s, ms, us, or ns for numeric timestamps")
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        columns = ("timestamp",) + SENSORS
        if not reader.fieldnames or not set(columns) <= set(reader.fieldnames):
            raise ValueError("CSV needs timestamp,ax,ay,az,gx,gy,gz columns")
        rows = np.array([[float(row[c]) for c in columns] for row in reader])
    if rows.ndim != 2 or len(rows) < 3 or not np.isfinite(rows).all():
        raise ValueError("Recording must contain at least three finite rows")
    t = (rows[:, 0] - rows[0, 0]) * units[time_unit]
    dt = np.diff(t)
    median = float(np.median(dt))
    if median <= 0 or np.any(dt <= 0) or dt.max() > 3 * median:
        raise ValueError("Timestamps must increase without large missing intervals")
    duration = float(t[-1] + median)
    if duration < 5 - 1e-6 or duration > 10.1:
        raise ValueError("Record 5–10 seconds per labelled file")
    fingerprint = hashlib.sha256(np.round(
        np.column_stack((t, rows[:, 1:])), 8).tobytes()).hexdigest()
    return t, rows[:, 1:], {"source_hz": 1 / median, "duration_s": duration,
                           "digest": fingerprint,
                           "jitter_fraction": float(np.std(dt) / median)}


def resample(t, values, target_hz):
    """Use an anti-alias FIR before reducing a near-uniform recording's rate."""
    dt = float(np.median(np.diff(t)))
    source_hz = 1 / dt
    if (not np.isfinite(target_hz) or target_hz <= 0
            or target_hz > source_hz * 1.01):
        raise ValueError("Target rate must be positive and no higher than source")
    if np.std(np.diff(t)) / dt > 0.1:
        raise ValueError("Sampling jitter exceeds 10%; inspect or recollect this file")
    regular_t = np.arange(int(np.floor(t[-1] / dt + 1e-7)) + 1) * dt
    regular = np.column_stack([np.interp(regular_t, t, v) for v in values.T])
    if target_hz < source_hz * 0.95:
        half = int(np.ceil(4 * source_hz / target_hz))
        if 2 * half + 1 > len(regular):
            raise ValueError("Recording too short for anti-alias filter")
        offsets = np.arange(-half, half + 1)
        cutoff = 0.45 * target_hz / source_hz
        kernel = 2 * cutoff * np.sinc(2 * cutoff * offsets) * np.hamming(len(offsets))
        kernel /= kernel.sum()
        regular = np.column_stack([
            np.convolve(np.pad(v, half, mode="edge"), kernel, mode="valid")
            for v in regular.T])
    grid = np.arange(int(np.floor(t[-1] * target_hz + 1e-7)) + 1) / target_hz
    return grid, np.column_stack([np.interp(grid, regular_t, v) for v in regular.T])


def spectral_features(signal, hz):
    """Return a non-DC peak frequency and integrated Hann-windowed PSD."""
    signal = np.asarray(signal, dtype=float)
    window = np.hanning(len(signal))
    power = np.abs(np.fft.rfft((signal - signal.mean()) * window)) ** 2
    power /= hz * np.sum(window ** 2)
    if len(signal) % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2
    frequencies = np.fft.rfftfreq(len(signal), d=1 / hz)
    peak = (float(frequencies[1 + np.argmax(power[1:])])
            if power[1:].max() > 1e-20 else 0.0)
    return peak, float(power.sum() * hz / len(signal))


def window_features(values, hz, seconds=2, hop_seconds=1):
    """Produce 7 time-domain and 4 FFT features; omit incomplete tail windows."""
    width, step = int(round(seconds * hz)), int(round(hop_seconds * hz))
    if width < 4 or step < 1 or len(values) < width:
        raise ValueError("Need at least one complete window of four samples")
    rows, starts = [], []
    for start in range(0, len(values) - width + 1, step):
        x = values[start:start + width]
        acc, gyro = np.linalg.norm(x[:, :3], axis=1), np.linalg.norm(x[:, 3:], axis=1)
        time_features = [v for signal in (acc, gyro)
                         for v in (signal.mean(), signal.std(),
                                   np.sqrt(np.mean(signal ** 2)))]
        sma = np.abs(x[:, :3]).sum(axis=1).mean()
        rows.append(time_features + [sma] + list(spectral_features(acc, hz))
                    + list(spectral_features(gyro, hz)))
        starts.append(start / hz)
    return np.array(rows), np.array(starts)


def fit_scaler(sequences):
    """Estimate normalization from training windows only, keeping constant axes safe."""
    x = np.vstack(sequences)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    return mean, scale


def activity_metrics(truth, prediction, n_states=4):
    """Return confusion counts and one-vs-rest metrics; undefined ratios are None."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("Metrics require matching nonempty vectors")
    if (not np.isin(truth, np.arange(n_states)).all()
            or not np.isin(prediction, np.arange(n_states)).all()):
        raise ValueError("Unknown activity label")
    counts = np.zeros((n_states, n_states), dtype=int)
    np.add.at(counts, (truth, prediction), 1)
    ratio = lambda a, b: float(a / b) if b else None
    rows = []
    for s in range(n_states):
        tp = int(counts[s, s])
        fn, fp = int(counts[s].sum() - tp), int(counts[:, s].sum() - tp)
        tn = int(counts.sum()) - tp - fn - fp
        rows.append({"activity": ACTIVITIES[s], "windows": tp + fn,
                     "sensitivity": ratio(tp, tp + fn),
                     "specificity": ratio(tn, tn + fp),
                     "one_vs_rest_accuracy": ratio(tp + tn, len(truth))})
    return {"confusion_matrix": counts.tolist(), "per_activity": rows,
            "overall_accuracy": float(np.trace(counts) / len(truth))}

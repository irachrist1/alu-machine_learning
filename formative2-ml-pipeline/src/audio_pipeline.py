"""Audio augmentation and MFCC-based feature extraction."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg"}


def load_audio(path: Path, sample_rate: int = 16_000) -> tuple[np.ndarray, int]:
    """Decode common phone audio formats to mono float audio using ffmpeg."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"No audio samples were decoded from {path}")
    peak = float(np.max(np.abs(audio)))
    if peak > 1:
        audio = audio / peak
    return audio, sample_rate


def augment_audio(
    audio: np.ndarray, sample_rate: int, seed: int = 42
) -> dict[str, np.ndarray]:
    """Make pitch, time-stretch, and background-noise variants."""
    rng = np.random.default_rng(seed)
    pitch_factor = 2 ** (2 / 12)  # shift upward by about two semitones
    pitched = resample(audio, max(1, int(len(audio) / pitch_factor))).astype(np.float32)
    stretched = resample(audio, max(1, int(len(audio) / 0.90))).astype(np.float32)
    noise_scale = max(float(np.std(audio)) * 0.02, 1e-4)
    noisy = np.clip(
        audio + rng.normal(0, noise_scale, len(audio)), -1, 1
    ).astype(np.float32)
    return {
        "original": audio.astype(np.float32),
        "pitch_shift_2": pitched,
        "time_stretch_0_90": stretched,
        "background_noise": noisy,
    }


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595 * np.log10(1 + np.asarray(frequency) / 700)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700 * (10 ** (np.asarray(mel) / 2595) - 1)


def _power_spectrogram(
    audio: np.ndarray, sample_rate: int, n_fft: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    frame_length = int(0.025 * sample_rate)
    frame_step = int(0.010 * sample_rate)
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))
    frame_count = 1 + int(np.ceil((len(audio) - frame_length) / frame_step))
    padded_length = (frame_count - 1) * frame_step + frame_length
    padded = np.pad(audio, (0, max(0, padded_length - len(audio))))
    indices = (
        np.arange(frame_length)[None, :]
        + np.arange(frame_count)[:, None] * frame_step
    )
    frames = padded[indices] * np.hamming(frame_length)
    spectrum = np.fft.rfft(frames, n=n_fft)
    power = (np.abs(spectrum) ** 2) / n_fft
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    return power, frequencies


def extract_audio_features(
    audio: np.ndarray, sample_rate: int, n_mfcc: int = 13
) -> dict[str, float]:
    """Extract MFCC means/stds, roll-off, RMS energy, and zero crossings."""
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    power, frequencies = _power_spectrogram(emphasized, sample_rate)
    n_fft_bins = power.shape[1]

    mel_points = np.linspace(
        _hz_to_mel(0), _hz_to_mel(sample_rate / 2), 28
    )
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((512 + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft_bins - 1)
    filters = np.zeros((26, n_fft_bins))
    for index in range(1, 27):
        left, center, right = bins[index - 1 : index + 2]
        if center > left:
            filters[index - 1, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filters[index - 1, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    filter_energies = np.maximum(power @ filters.T, np.finfo(float).eps)
    mfcc = dct(np.log(filter_energies), type=2, axis=1, norm="ortho")[:, :n_mfcc]
    features: dict[str, float] = {}
    for index in range(n_mfcc):
        features[f"mfcc_{index + 1:02d}_mean"] = float(mfcc[:, index].mean())
        features[f"mfcc_{index + 1:02d}_std"] = float(mfcc[:, index].std())

    cumulative = np.cumsum(power, axis=1)
    thresholds = 0.85 * cumulative[:, -1]
    rolloff_indices = np.argmax(cumulative >= thresholds[:, None], axis=1)
    features["spectral_rolloff_mean"] = float(frequencies[rolloff_indices].mean())
    features["rms_energy"] = float(np.sqrt(np.mean(audio**2)))
    features["zero_crossing_rate"] = float(
        np.mean(np.abs(np.diff(np.signbit(audio))).astype(float))
    )
    features["duration_seconds"] = float(len(audio) / sample_rate)
    return features


def process_audio_dataset(
    input_dir: Path = Path("data/audio"),
    output_csv: Path = Path("data/processed/audio_features.csv"),
    augmented_dir: Path = Path("data/audio/augmented"),
) -> pd.DataFrame:
    """Process member audio folders and save all required feature rows."""
    rows: list[dict] = []
    augmented_dir.mkdir(parents=True, exist_ok=True)
    member_dirs = [
        folder
        for folder in sorted(input_dir.iterdir())
        if folder.is_dir() and folder.name != augmented_dir.name
    ] if input_dir.exists() else []

    for member_dir in member_dirs:
        for source_path in sorted(member_dir.iterdir()):
            if source_path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            audio, sample_rate = load_audio(source_path)
            variants = augment_audio(audio, sample_rate)
            for variant_name, variant in variants.items():
                output_member_dir = augmented_dir / member_dir.name
                output_member_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_member_dir / (
                    f"{source_path.stem}__{variant_name}.wav"
                )
                wavfile.write(output_path, sample_rate, variant.astype(np.float32))
                rows.append(
                    {
                        "member": member_dir.name,
                        "source_file": str(source_path),
                        "variant": variant_name,
                        "processed_file": str(output_path),
                        **extract_audio_features(variant, sample_rate),
                    }
                )

    features = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/audio"))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("data/processed/audio_features.csv")
    )
    args = parser.parse_args()
    features = process_audio_dataset(args.input_dir, args.output_csv)
    print(f"Saved {len(features)} audio feature rows to {args.output_csv}")


if __name__ == "__main__":
    main()


"""Image augmentation, visualization support, and histogram features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def augment_image(image: Image.Image) -> dict[str, Image.Image]:
    """Return the original plus three required augmentation variants."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    return {
        "original": rgb,
        "rotated_15deg": rgb.rotate(15, resample=Image.Resampling.BILINEAR),
        "horizontal_flip": ImageOps.mirror(rgb),
        "grayscale": ImageOps.grayscale(rgb).convert("RGB"),
    }


def extract_image_features(image: Image.Image, bins: int = 16) -> dict[str, float]:
    """Create normalized RGB/gray histograms and a small pixel embedding."""
    rgb = ImageOps.fit(image.convert("RGB"), (128, 128))
    array = np.asarray(rgb, dtype=np.float32)
    features: dict[str, float] = {}

    for channel_index, channel_name in enumerate(["red", "green", "blue"]):
        histogram, _ = np.histogram(
            array[:, :, channel_index], bins=bins, range=(0, 256), density=False
        )
        histogram = histogram / max(histogram.sum(), 1)
        for index, value in enumerate(histogram):
            features[f"{channel_name}_hist_{index:02d}"] = float(value)

    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    gray_histogram, _ = np.histogram(gray, bins=bins, range=(0, 256))
    gray_histogram = gray_histogram / max(gray_histogram.sum(), 1)
    for index, value in enumerate(gray_histogram):
        features[f"gray_hist_{index:02d}"] = float(value)

    # This 16x16 representation acts as a basic face appearance embedding.
    embedding = np.asarray(
        rgb.convert("L").resize((16, 16), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    for index, value in enumerate(embedding.ravel()):
        features[f"pixel_embedding_{index:03d}"] = float(value)

    features["brightness_mean"] = float(gray.mean())
    features["brightness_std"] = float(gray.std())
    return features


def process_image_dataset(
    input_dir: Path = Path("data/images"),
    output_csv: Path = Path("data/processed/image_features.csv"),
    augmented_dir: Path = Path("data/images/augmented"),
) -> pd.DataFrame:
    """Process member folders and save one row for every image variant."""
    rows: list[dict] = []
    augmented_dir.mkdir(parents=True, exist_ok=True)

    member_dirs = [
        folder
        for folder in sorted(input_dir.iterdir())
        if folder.is_dir() and folder.name != augmented_dir.name
    ] if input_dir.exists() else []

    for member_dir in member_dirs:
        for source_path in sorted(member_dir.iterdir()):
            if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            with Image.open(source_path) as image:
                variants = augment_image(image)
            for variant_name, variant in variants.items():
                output_member_dir = augmented_dir / member_dir.name
                output_member_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_member_dir / (
                    f"{source_path.stem}__{variant_name}.png"
                )
                variant.save(output_path)
                rows.append(
                    {
                        "member": member_dir.name,
                        "source_file": str(source_path),
                        "variant": variant_name,
                        "processed_file": str(output_path),
                        **extract_image_features(variant),
                    }
                )

    features = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/images"))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("data/processed/image_features.csv")
    )
    args = parser.parse_args()
    features = process_image_dataset(args.input_dir, args.output_csv)
    print(f"Saved {len(features)} image feature rows to {args.output_csv}")


if __name__ == "__main__":
    main()


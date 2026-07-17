from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.audio_pipeline import extract_audio_features, load_audio
from src.image_pipeline import augment_image, extract_image_features
from src.auth_model import split_sources_by_identity


def test_image_augmentations_and_feature_count():
    image_path = Path("data/images/yvette_uwimpaye/mi.png")
    with Image.open(image_path) as image:
        variants = augment_image(image)
        features = extract_image_features(variants["original"])

    assert set(variants) == {
        "original",
        "rotated_15deg",
        "horizontal_flip",
        "grayscale",
    }
    assert len(features) == 322
    assert np.isclose(
        sum(features[f"red_hist_{index:02d}"] for index in range(16)), 1
    )


def test_audio_decoding_and_required_features():
    audio_path = Path("data/audio/mahlet_tilahun/voice_010858.m4a")
    audio, sample_rate = load_audio(audio_path)
    features = extract_audio_features(audio, sample_rate)

    assert sample_rate == 16_000
    assert len(audio) > 0
    assert "mfcc_01_mean" in features
    assert "mfcc_13_std" in features
    assert features["spectral_rolloff_mean"] > 0
    assert features["rms_energy"] > 0


def test_single_source_identity_stays_in_training():
    rows = []
    for source in ["christian_1", "christian_2"]:
        rows.append({"member": "christian", "source_file": source, "value": 1})
    rows.append({"member": "yvette", "source_file": "yvette_1", "value": 2})
    train, test = split_sources_by_identity(pd.DataFrame(rows))

    assert "yvette" in set(train["member"])
    assert "yvette" not in set(test["member"])
    assert test["source_file"].nunique() == 1

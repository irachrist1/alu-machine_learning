# Multimodal User Authentication and Product Recommendation

This project combines customer transaction data, social media data, facial images, and voice samples. The command-line flow first checks the face, runs the product model, then checks the voice before showing the result.

## Data sources

- [Customer social profiles](https://docs.google.com/spreadsheets/d/10up-WdC0a6egYaXLKiMQUotpOvaKZRZvYuWFQx4-RPQ/edit?gid=862127784)
- [Customer transactions](https://docs.google.com/spreadsheets/d/1s4WOVm49lmLQ8d9QbbbdgAcRTNh3m0KCH_5ciiaRZw0/edit?gid=1844409263)

## Required group media

Every member needs their own lowercase folder in both `data/images` and `data/audio`. Use the same member name in both locations, for example `christian_tonny_gentil_iradukunda`.

Each image folder must have:

- `neutral.jpg`
- `smiling.jpg`
- `surprised.jpg`

Each audio folder must have:

- `yes_approve.m4a` saying “Yes, approve”
- `confirm_transaction.m4a` saying “Confirm transaction”

Also put one outsider image and outsider voice in `data/unauthorized` for the denied-access demo. The outsider must give permission for the sample to be used.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

FFmpeg is needed for phone audio such as M4A files.

## Run the pipeline

```bash
python -m src.tabular_pipeline
python -m src.image_pipeline
python -m src.audio_pipeline
python -m src.train_auth_models
```

The scripts create:

- `data/processed/merged_dataset.csv`
- `data/processed/image_features.csv`
- `data/processed/audio_features.csv`
- evaluation JSON files with accuracy, macro F1-score, and log loss
- model files in `models/`

The feature extraction scripts save three augmentations for every original sample. Image variants are rotation, horizontal flip, and grayscale. Audio variants are pitch shift, time stretch, and background noise.

## Run the system demonstration

```bash
python -m src.system_simulation \
  --face data/images/christian_tonny_gentil_iradukunda/neutral.jpeg \
  --audio data/audio/christian_tonny_gentil_iradukunda/yes_approve.m4a \
  --customer-id 150 \
  --purchase-amount 250 \
  --purchase-date 2024-04-15
```

For the unauthorized test, use the outsider image or outsider audio and show the “Access denied” result.

## Tests

```bash
python -m pytest -q
```

The model split keeps every augmented version of one original sample on the same side. This avoids training on an original and testing on its flipped or noisy copy.

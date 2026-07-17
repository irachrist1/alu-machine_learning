"""Build the assignment notebook with nbformat."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "formative2_multimodal_preprocessing.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook.metadata.language_info = {"name": "python", "version": "3"}

notebook.cells = [
    markdown(
        """
# Formative 2: Multimodal Data Preprocessing

This notebook shows how we cleaned and merged the customer datasets, processed images and sound, and built the three-model system.
"""
    ),
    markdown(
        """
## tl;dr

The social dataset had 155 rows, including 5 exact duplicates. The transaction dataset had 150 rows and 10 missing ratings. After cleaning social profiles and doing a validated many-to-one inner join, 117 transactions remained and no transaction was duplicated.

The product model got 0.167 accuracy, 0.177 macro F1-score, and 1.746 log loss on 30 held-out rows. This is a weak result, so it should be treated as a classroom simulation and not a real shopping system.

The image and audio pipelines each produced 16 feature rows after augmentation. The image model got 1.000 accuracy, 1.000 macro F1-score, and 0.088 log loss, but only Christian had enough original images for a held-out evaluation. The audio model got 1.000 accuracy, 1.000 macro F1-score, and 0.096 log loss across Christian and Mahlet. These high authentication scores come from a very small dataset and should not be generalized.
"""
    ),
    markdown(
        """
## Context & Methods

The required flow is face recognition, product prediction, voice approval, then display the predicted product. Images are augmented using rotation, horizontal flip, and grayscale. Audio is augmented with pitch shift, time stretch, and background noise.

### Key Assumptions

- IDs such as A178 and 178 represent the same customer.
- Repeated social observations are useful, so they are aggregated to customer-level averages and maximums.
- Exact duplicate social rows are removed first.
- An inner join is used because the recommendation model needs features from both sources.
- Missing customer ratings are filled with the median for the cleaned dataset, but rating is not used to predict product because it is collected after purchase.
- All augmentations from one original sample stay in the same model split to reduce leakage.
"""
    ),
    code(
        """
from pathlib import Path
import json
import os
import sys

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from PIL import Image
from scipy.signal import spectrogram
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.audio_pipeline import load_audio, process_audio_dataset
from src.auth_model import train_identity_model
from src.image_pipeline import process_image_dataset
from src.tabular_pipeline import (
    MODEL_FEATURES,
    RANDOM_STATE,
    TARGET,
    make_product_pipeline,
    merge_and_engineer,
)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)
"""
    ),
    markdown("## Data"),
    markdown("### 1. Load and inspect the two tabular sources"),
    code(
        """
social = pd.read_csv("data/raw/customer_social_profiles.csv")
transactions = pd.read_csv("data/raw/customer_transactions.csv")

print("Social shape:", social.shape)
print("Transaction shape:", transactions.shape)
display(social.head())
display(transactions.head())
"""
    ),
    code(
        """
quality_table = pd.DataFrame({
    "social_dtype": social.dtypes.astype(str),
    "social_nulls": social.isna().sum(),
}).fillna("-")
transaction_quality = pd.DataFrame({
    "transaction_dtype": transactions.dtypes.astype(str),
    "transaction_nulls": transactions.isna().sum(),
})
display(quality_table)
display(transaction_quality)
print("Exact social duplicates:", social.duplicated().sum())
print("Exact transaction duplicates:", transactions.duplicated().sum())
"""
    ),
    markdown("### 2. Exploratory data analysis"),
    code(
        """
plt.figure(figsize=(7, 4))
sns.countplot(
    data=transactions,
    x="product_category",
    order=transactions["product_category"].value_counts().index,
    color="#4C78A8",
)
plt.title("Transactions by Product Category")
plt.xlabel("Product category")
plt.ylabel("Number of transactions")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
plt.figure(figsize=(7, 4))
sns.histplot(transactions["purchase_amount"], bins=12, kde=True, color="#F58518")
plt.title("Distribution of Purchase Amount")
plt.xlabel("Purchase amount")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
plt.figure(figsize=(8, 4))
sns.boxplot(
    data=transactions,
    x="product_category",
    y="purchase_amount",
    color="#72B7B2",
)
plt.title("Purchase Amount Outliers by Product Category")
plt.xlabel("Product category")
plt.ylabel("Purchase amount")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
social_numeric = social[
    ["engagement_score", "purchase_interest_score"]
].copy()
transaction_numeric = transactions[
    ["purchase_amount", "customer_rating"]
].copy()
eda_numeric = pd.concat(
    [social_numeric.reset_index(drop=True), transaction_numeric.reset_index(drop=True)],
    axis=1,
)
plt.figure(figsize=(6, 4))
sns.heatmap(eda_numeric.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between Numeric Variables")
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
The categories are fairly balanced, with Sports having the most rows. Purchase amounts cover almost the full range for every category, and the numeric correlations are weak. This helps explain why product prediction is difficult.
"""
    ),
    markdown("### 3. Clean, aggregate, and validate the merge"),
    code(
        """
merged, merge_audit = merge_and_engineer(social, transactions)
display(pd.json_normalize(merge_audit, sep=".").T.rename(columns={0: "value"}))
print("Merged shape:", merged.shape)
display(merged.head())
"""
    ),
    code(
        """
print("Null values after merge:")
display(merged.isna().sum().to_frame("null_count"))
print("Duplicate transaction IDs:", merged["transaction_id"].duplicated().sum())
print("Product distribution after merge:")
display(merged[TARGET].value_counts().to_frame("rows"))
"""
    ),
    markdown("## Results"),
    markdown("### 4. Product recommendation model"),
    code(
        """
x = merged[MODEL_FEATURES]
y = merged[TARGET]
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)
product_model = make_product_pipeline()
product_model.fit(x_train, y_train)
product_predictions = product_model.predict(x_test)
product_probabilities = product_model.predict_proba(x_test)

from sklearn.metrics import accuracy_score, f1_score, log_loss

product_metrics = {
    "accuracy": accuracy_score(y_test, product_predictions),
    "macro_f1": f1_score(y_test, product_predictions, average="macro"),
    "log_loss": log_loss(
        y_test,
        product_probabilities,
        labels=product_model.named_steps["classifier"].classes_,
    ),
}
display(pd.Series(product_metrics).round(4).to_frame("score"))
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    product_predictions,
    cmap="Blues",
    xticks_rotation=25,
    ax=ax,
)
plt.title("Product Model Confusion Matrix")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
preprocessor = product_model.named_steps["preprocessing"]
feature_names = preprocessor.get_feature_names_out()
importance = pd.Series(
    product_model.named_steps["classifier"].feature_importances_,
    index=feature_names,
).sort_values(ascending=False).head(12)

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh", color="#54A24B")
plt.title("Top Product Model Feature Importances")
plt.xlabel("Random Forest importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
The product model does not separate the five classes well. Its accuracy is below the majority-class baseline in this split. The data appears generated for practicing preprocessing, so a low score is not surprising. More customer history and a real preference signal would be needed to improve it.
"""
    ),
    markdown("### 5. Image samples, augmentation, and features"),
    code(
        """
image_paths = sorted(
    path
    for path in Path("data/images").glob("*/*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    and "augmented" not in path.parts
)
print("Original images found:", len(image_paths))

if image_paths:
    fig, axes = plt.subplots(1, len(image_paths), figsize=(4 * len(image_paths), 4))
    axes = np.atleast_1d(axes)
    for ax, path in zip(axes, image_paths):
        with Image.open(path) as image:
            ax.imshow(image)
        ax.set_title(f"{path.parent.name}: {path.stem}")
        ax.axis("off")
    plt.suptitle("Original Face Samples")
    plt.tight_layout()
    plt.show()
else:
    print("No image samples are available yet.")
"""
    ),
    code(
        """
image_features = process_image_dataset()
if image_features.empty:
    print("Image feature extraction is waiting for group files.")
else:
    image_summary = (
        image_features.groupby(["member", "variant"])
        .size()
        .rename("rows")
        .reset_index()
    )
    display(image_summary)
    display(image_features.iloc[:4, :12])
    print("Image feature file shape:", image_features.shape)
"""
    ),
    markdown(
        """
The image CSV contains RGB and grayscale histograms, brightness statistics, and a small 16 by 16 pixel embedding. Every original gets three augmented copies.
"""
    ),
    markdown("### 6. Audio waveforms, spectrograms, augmentation, and features"),
    code(
        """
audio_paths = sorted(
    path
    for path in Path("data/audio").glob("*/*")
    if path.suffix.lower() in {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg"}
    and "augmented" not in path.parts
)
print("Original audio files found:", len(audio_paths))

for member in sorted({path.parent.name for path in audio_paths}):
    path = next(path for path in audio_paths if path.parent.name == member)
    audio, sample_rate = load_audio(path)
    times = np.arange(len(audio)) / sample_rate
    frequencies, spec_times, power = spectrogram(audio, fs=sample_rate)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(times, audio, color="#4C78A8", linewidth=0.7)
    axes[0].set_title(f"{member} Waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[1].pcolormesh(
        spec_times,
        frequencies,
        10 * np.log10(power + 1e-10),
        shading="auto",
        cmap="magma",
    )
    axes[1].set_ylim(0, 8000)
    axes[1].set_title(f"{member} Spectrogram")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        """
audio_features = process_audio_dataset()
if audio_features.empty:
    print("Audio feature extraction is waiting for group files.")
else:
    audio_summary = (
        audio_features.groupby(["member", "variant"])
        .size()
        .rename("rows")
        .reset_index()
    )
    display(audio_summary)
    display(audio_features.iloc[:4, :12])
    print("Audio feature file shape:", audio_features.shape)
"""
    ),
    markdown(
        """
The audio CSV saves 13 MFCC means, 13 MFCC standard deviations, spectral roll-off, RMS energy, zero-crossing rate, and duration. The Christian and Mahlet recordings have visible speech bursts, and most energy is concentrated in lower frequencies.
"""
    ),
    markdown("### 7. Facial recognition and voiceprint models"),
    code(
        """
auth_results = {}
for kind, features in {"image": image_features, "audio": audio_features}.items():
    metadata_columns = {"member", "source_file", "variant", "processed_file"}
    feature_columns = [
        column for column in features.columns if column not in metadata_columns
    ]
    _, metrics = train_identity_model(
        features,
        feature_columns,
        Path(f"models/{kind}_recognition_model.joblib"),
        Path(f"data/processed/{kind}_model_metrics.json"),
    )
    auth_results[kind] = metrics

auth_summary = pd.DataFrame(
    {
        kind: {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["f1_macro"],
            "log_loss": metrics["log_loss"],
            "trained_identities": len(metrics["trained_identities"]),
            "evaluated_identities": len(metrics["evaluated_identities"]),
        }
        for kind, metrics in auth_results.items()
    }
).T
display(auth_summary.round(4))
print("Image identity coverage:", auth_results["image"]["original_samples_per_identity"])
print("Audio identity coverage:", auth_results["audio"]["original_samples_per_identity"])
"""
    ),
    markdown(
        """
Both authentication models worked on the held-out original samples. The audio evaluation covers both available speakers. The image evaluation only covers Christian because Yvette supplied one original image, so the perfect image score must be read with that limitation. Augmented copies from one original were never split across training and testing.
"""
    ),
    markdown("## Takeaways"),
    markdown(
        """
- Cleaning and aggregation prevented a many-to-many merge from duplicating transactions.
- The product model is functional, but its current test performance is weak and should not be presented as production quality.
- The image and audio processors create the required augmentations and feature CSV files, and both authentication models run successfully.
- The valid CLI transaction approved Christian and predicted Clothing. A non-face image was rejected at the face step, and a Yvette-face/Mahlet-voice combination was denied because the identities did not match.
- The shared folder is still missing Hassan's media, Mahlet's images, and most of Yvette's media. The GitHub and demonstration-video links must also be inserted before submission.
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)

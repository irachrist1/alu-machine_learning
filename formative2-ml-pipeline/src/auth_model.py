"""Shared training helpers for face and voice identity classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss


def split_sources_by_identity(
    features: pd.DataFrame,
    identity_column: str = "member",
    source_column: str = "source_file",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep all augmentations from one source in the same split."""
    unique_sources = features[[identity_column, source_column]].drop_duplicates()
    rng = np.random.default_rng(random_state)
    test_sources: list[str] = []

    identities_with_holdout: list[str] = []
    for identity, group in unique_sources.groupby(identity_column):
        sources = group[source_column].astype(str).to_numpy().copy()
        # A singleton identity can be learned, but it cannot provide a fair test
        # sample. Keep it in training and report that coverage limitation.
        if len(sources) < 2:
            continue
        rng.shuffle(sources)
        test_sources.append(str(sources[0]))
        identities_with_holdout.append(str(identity))

    if not test_sources:
        raise ValueError(
            "At least one member needs 2 original samples before model evaluation."
        )

    test_mask = features[source_column].astype(str).isin(test_sources)
    return features.loc[~test_mask].copy(), features.loc[test_mask].copy()


def train_identity_model(
    features: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
    metrics_path: Path,
) -> tuple[RandomForestClassifier, dict]:
    """Train a closed-set identity model and save evaluation outputs."""
    if features["member"].nunique() < 2:
        raise ValueError("At least 2 different members are needed to train this model.")

    train, test = split_sources_by_identity(features)
    evaluation_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    evaluation_model.fit(train[feature_columns], train["member"])
    predictions = evaluation_model.predict(test[feature_columns])
    probabilities = evaluation_model.predict_proba(test[feature_columns])

    original_counts = (
        features[["member", "source_file"]]
        .drop_duplicates()
        .groupby("member")
        .size()
        .to_dict()
    )
    evaluated_identities = sorted(test["member"].astype(str).unique())

    metrics = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_original_samples": int(train["source_file"].nunique()),
        "test_original_samples": int(test["source_file"].nunique()),
        "trained_identities": sorted(features["member"].astype(str).unique()),
        "evaluated_identities": evaluated_identities,
        "original_samples_per_identity": {
            str(key): int(value) for key, value in original_counts.items()
        },
        "all_identities_evaluated": len(evaluated_identities)
        == features["member"].nunique(),
        "accuracy": float(accuracy_score(test["member"], predictions)),
        "f1_macro": float(f1_score(test["member"], predictions, average="macro")),
        "log_loss": float(
            log_loss(
                test["member"], probabilities, labels=evaluation_model.classes_
            )
        ),
        "classification_report": classification_report(
            test["member"], predictions, output_dict=True, zero_division=0
        ),
    }

    # Refit on every row after evaluation so the saved classifier can use all
    # original and augmented samples during the command-line demonstration.
    final_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(features[feature_columns], features["member"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final_model, "feature_columns": feature_columns}, model_path
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return final_model, metrics


def predict_identity(
    model_bundle: dict,
    feature_row: pd.DataFrame,
    confidence_threshold: float = 0.65,
) -> tuple[str, float, bool]:
    """Predict a member and reject low-confidence samples as unauthorized."""
    model = model_bundle["model"]
    columns = model_bundle["feature_columns"]
    probabilities = model.predict_proba(feature_row[columns])[0]
    best_index = int(np.argmax(probabilities))
    identity = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    return identity, confidence, confidence >= confidence_threshold

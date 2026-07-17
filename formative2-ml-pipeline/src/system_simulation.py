"""Command-line simulation of face, product, and voice model logic."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from PIL import Image

from src.audio_pipeline import extract_audio_features, load_audio
from src.auth_model import predict_identity
from src.image_pipeline import extract_image_features
from src.tabular_pipeline import MODEL_FEATURES


def product_input(customer_id: int, amount: float, date: str) -> pd.DataFrame:
    """Combine a known social profile with current transaction details."""
    merged = pd.read_csv("data/processed/merged_dataset.csv")
    profile_columns = [
        "engagement_score_mean",
        "engagement_score_max",
        "purchase_interest_score_mean",
        "purchase_interest_score_max",
        "social_profile_count",
        "platform_count",
        "review_sentiment",
        "primary_social_platform",
    ]
    matches = merged.loc[merged["customer_id"] == customer_id, profile_columns]
    if matches.empty:
        raise ValueError(f"Customer {customer_id} does not have a merged social profile.")

    row = matches.iloc[0].to_dict()
    purchase_date = pd.Timestamp(date)
    row.update(
        {
            "purchase_amount": amount,
            "purchase_month": purchase_date.month,
            "purchase_day_of_week": purchase_date.dayofweek,
            "purchase_quarter": purchase_date.quarter,
        }
    )
    return pd.DataFrame([row])[MODEL_FEATURES]


def simulate(args: argparse.Namespace) -> int:
    required_models = {
        "face": Path("models/image_recognition_model.joblib"),
        "voice": Path("models/audio_recognition_model.joblib"),
        "product": Path("models/product_recommendation_model.joblib"),
    }
    missing = [name for name, path in required_models.items() if not path.exists()]
    if missing:
        print("Models missing: " + ", ".join(missing))
        print("Run the preprocessing and training scripts first.")
        return 2

    face_bundle = joblib.load(required_models["face"])
    with Image.open(args.face) as image:
        face_row = pd.DataFrame([extract_image_features(image)])
    face_member, face_confidence, face_allowed = predict_identity(
        face_bundle, face_row, args.confidence_threshold
    )
    print(f"Face result: {face_member} ({face_confidence:.1%})")
    if not face_allowed:
        print("Access denied: face was not recognized with enough confidence.")
        return 1

    product_model = joblib.load(required_models["product"])
    try:
        candidate = product_model.predict(
            product_input(args.customer_id, args.purchase_amount, args.purchase_date)
        )[0]
    except ValueError as error:
        print(f"Access denied: {error}")
        return 1
    print("Product model ran. Voice approval is still required.")

    voice_bundle = joblib.load(required_models["voice"])
    audio, sample_rate = load_audio(args.audio)
    voice_row = pd.DataFrame([extract_audio_features(audio, sample_rate)])
    voice_member, voice_confidence, voice_allowed = predict_identity(
        voice_bundle, voice_row, args.confidence_threshold
    )
    print(f"Voice result: {voice_member} ({voice_confidence:.1%})")

    if not voice_allowed or voice_member != face_member:
        print("Access denied: voice was not approved or identities did not match.")
        return 1

    print(f"Approved user: {face_member}")
    print(f"Predicted product: {candidate}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--customer-id", type=int, required=True)
    parser.add_argument("--purchase-amount", type=float, required=True)
    parser.add_argument("--purchase-date", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    args = parser.parse_args()
    raise SystemExit(simulate(args))


if __name__ == "__main__":
    main()

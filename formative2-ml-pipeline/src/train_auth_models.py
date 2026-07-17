"""Create face and voice classifiers from saved feature CSV files."""

from pathlib import Path

import pandas as pd

from src.auth_model import train_identity_model


def train_one(kind: str) -> None:
    feature_path = Path(f"data/processed/{kind}_features.csv")
    if not feature_path.exists():
        print(f"{kind}: skipped because {feature_path} is missing")
        return

    features = pd.read_csv(feature_path)
    metadata = {"member", "source_file", "variant", "processed_file"}
    feature_columns = [column for column in features.columns if column not in metadata]
    try:
        _, metrics = train_identity_model(
            features,
            feature_columns,
            Path(f"models/{kind}_recognition_model.joblib"),
            Path(f"data/processed/{kind}_model_metrics.json"),
        )
    except ValueError as error:
        print(f"{kind}: not trained yet - {error}")
        return

    print(
        f"{kind}: accuracy={metrics['accuracy']:.3f}, "
        f"macro_f1={metrics['f1_macro']:.3f}, loss={metrics['log_loss']:.3f}"
    )


def main() -> None:
    train_one("image")
    train_one("audio")


if __name__ == "__main__":
    main()


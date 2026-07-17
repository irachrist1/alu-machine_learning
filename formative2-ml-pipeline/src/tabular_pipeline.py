"""Clean, merge, and model the customer tabular datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "purchase_amount",
    "engagement_score_mean",
    "engagement_score_max",
    "purchase_interest_score_mean",
    "purchase_interest_score_max",
    "social_profile_count",
    "platform_count",
    "purchase_month",
    "purchase_day_of_week",
    "purchase_quarter",
]

CATEGORICAL_FEATURES = ["primary_social_platform", "review_sentiment"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "product_category"


def _mode(series: pd.Series) -> str:
    """Return a stable mode for grouped categorical values."""
    modes = series.dropna().mode()
    return "Unknown" if modes.empty else str(sorted(modes.astype(str))[0])


def clean_social_profiles(social: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean IDs and collapse repeated social observations per customer."""
    original_rows = len(social)
    exact_duplicates = int(social.duplicated().sum())
    social = social.drop_duplicates().copy()

    social["customer_id"] = pd.to_numeric(
        social["customer_id_new"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    social["engagement_score"] = pd.to_numeric(
        social["engagement_score"], errors="coerce"
    )
    social["purchase_interest_score"] = pd.to_numeric(
        social["purchase_interest_score"], errors="coerce"
    )
    social = social.dropna(subset=["customer_id"]).copy()
    social["customer_id"] = social["customer_id"].astype(int)

    # Pick the platform from the row where that customer engaged the most.
    primary_platform = (
        social.sort_values(
            ["customer_id", "engagement_score", "social_media_platform"],
            ascending=[True, False, True],
        )
        .drop_duplicates("customer_id")
        .set_index("customer_id")["social_media_platform"]
        .rename("primary_social_platform")
    )

    aggregated = social.groupby("customer_id", as_index=True).agg(
        engagement_score_mean=("engagement_score", "mean"),
        engagement_score_max=("engagement_score", "max"),
        purchase_interest_score_mean=("purchase_interest_score", "mean"),
        purchase_interest_score_max=("purchase_interest_score", "max"),
        social_profile_count=("customer_id", "size"),
        platform_count=("social_media_platform", "nunique"),
        review_sentiment=("review_sentiment", _mode),
    )
    aggregated = aggregated.join(primary_platform).reset_index()

    audit = {
        "original_rows": original_rows,
        "exact_duplicates_removed": exact_duplicates,
        "rows_after_deduplication": len(social),
        "unique_customers": int(aggregated["customer_id"].nunique()),
    }
    return aggregated, audit


def clean_transactions(transactions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fix transaction types and handle the missing rating values."""
    transactions = transactions.drop_duplicates().copy()
    transactions = transactions.rename(columns={"customer_id_legacy": "customer_id"})
    transactions["customer_id"] = pd.to_numeric(
        transactions["customer_id"], errors="coerce"
    )
    transactions["purchase_amount"] = pd.to_numeric(
        transactions["purchase_amount"], errors="coerce"
    )
    transactions["customer_rating"] = pd.to_numeric(
        transactions["customer_rating"], errors="coerce"
    )
    transactions["purchase_date"] = pd.to_datetime(
        transactions["purchase_date"], errors="coerce"
    )

    missing_ratings = int(transactions["customer_rating"].isna().sum())
    rating_median = float(transactions["customer_rating"].median())
    transactions["customer_rating"] = transactions["customer_rating"].fillna(
        rating_median
    )
    transactions = transactions.dropna(
        subset=["customer_id", "transaction_id", "purchase_date", TARGET]
    ).copy()
    transactions["customer_id"] = transactions["customer_id"].astype(int)

    audit = {
        "rows_after_cleaning": len(transactions),
        "missing_ratings_imputed": missing_ratings,
        "rating_median_used": rating_median,
        "duplicate_rows_removed": int(
            len(transactions) - len(transactions.drop_duplicates())
        ),
    }
    return transactions, audit


def merge_and_engineer(
    social: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Make an inner m:1 merge and add date features for modeling."""
    social_clean, social_audit = clean_social_profiles(social)
    transactions_clean, transaction_audit = clean_transactions(transactions)

    merged = transactions_clean.merge(
        social_clean,
        on="customer_id",
        how="inner",
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("The inner merge produced an unexpected unmatched row.")
    merged = merged.drop(columns="_merge")

    merged["purchase_month"] = merged["purchase_date"].dt.month
    merged["purchase_day_of_week"] = merged["purchase_date"].dt.dayofweek
    merged["purchase_quarter"] = merged["purchase_date"].dt.quarter
    merged = merged.sort_values("transaction_id").reset_index(drop=True)

    if merged["transaction_id"].duplicated().any():
        raise ValueError("A transaction was duplicated during the merge.")

    audit = {
        "social": social_audit,
        "transactions": transaction_audit,
        "merge": {
            "join_type": "inner",
            "relationship": "many_to_one",
            "rows_before_merge": len(transactions_clean),
            "rows_after_merge": len(merged),
            "transactions_without_social_profile": int(
                (~transactions_clean["customer_id"].isin(social_clean["customer_id"])).sum()
            ),
            "duplicate_transaction_ids_after_merge": int(
                merged["transaction_id"].duplicated().sum()
            ),
        },
    }
    return merged, audit


def make_product_pipeline() -> Pipeline:
    """Build preprocessing and classifier steps as one reusable object."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def train_product_model(merged: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Train and evaluate the product recommendation classifier."""
    x = merged[MODEL_FEATURES]
    y = merged[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_product_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    classes = model.named_steps["classifier"].classes_

    metrics = {
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    return model, metrics


def run_pipeline(
    social_path: Path,
    transactions_path: Path,
    output_dir: Path,
    model_dir: Path,
) -> tuple[dict, dict]:
    """Run the full tabular pipeline and save its reproducible outputs."""
    social = pd.read_csv(social_path)
    transactions = pd.read_csv(transactions_path)
    merged, audit = merge_and_engineer(social, transactions)
    model, metrics = train_product_model(merged)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "merged_dataset.csv", index=False)
    (output_dir / "merge_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (output_dir / "product_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    joblib.dump(model, model_dir / "product_recommendation_model.joblib")
    return audit, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--social",
        type=Path,
        default=Path("data/raw/customer_social_profiles.csv"),
    )
    parser.add_argument(
        "--transactions",
        type=Path,
        default=Path("data/raw/customer_transactions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    audit, metrics = run_pipeline(
        args.social, args.transactions, args.output_dir, args.model_dir
    )
    print(json.dumps({"merge_audit": audit, "model_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()


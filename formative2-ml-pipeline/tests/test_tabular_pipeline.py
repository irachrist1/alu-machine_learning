import pandas as pd

from src.tabular_pipeline import MODEL_FEATURES, merge_and_engineer, train_product_model


def load_sources():
    social = pd.read_csv("data/raw/customer_social_profiles.csv")
    transactions = pd.read_csv("data/raw/customer_transactions.csv")
    return social, transactions


def test_merge_is_many_to_one_without_transaction_duplication():
    social, transactions = load_sources()
    merged, audit = merge_and_engineer(social, transactions)

    assert len(merged) == 117
    assert merged["transaction_id"].is_unique
    assert audit["social"]["exact_duplicates_removed"] == 5
    assert audit["transactions"]["missing_ratings_imputed"] == 10
    assert audit["merge"]["transactions_without_social_profile"] == 33


def test_engineered_model_features_are_available():
    social, transactions = load_sources()
    merged, _ = merge_and_engineer(social, transactions)

    assert set(MODEL_FEATURES).issubset(merged.columns)
    assert merged["purchase_month"].between(1, 12).all()
    assert merged["purchase_day_of_week"].between(0, 6).all()


def test_product_model_trains_and_predicts_known_labels():
    social, transactions = load_sources()
    merged, _ = merge_and_engineer(social, transactions)
    model, metrics = train_product_model(merged)
    prediction = model.predict(merged[MODEL_FEATURES].head(3))

    assert len(prediction) == 3
    assert set(prediction).issubset(set(merged["product_category"]))
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_macro"] <= 1
    assert metrics["log_loss"] >= 0

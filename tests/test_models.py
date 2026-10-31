import numpy as np

from src.data.load import load_raw
from src.data.schema import TARGET
from src.models.evaluate import classification_metrics, precision_recall_at_k
from src.models.train import (
    build_logistic_regression_pipeline,
    compute_scale_pos_weight,
    split_data,
    xy,
)
from tests.conftest import requires_real_dataset


def test_split_data_is_stratified_and_disjoint(sample_df):
    train_df, val_df, test_df = split_data(sample_df)

    assert len(train_df) + len(val_df) + len(test_df) == len(sample_df)
    assert set(train_df.index) & set(val_df.index) == set()
    assert set(train_df.index) & set(test_df.index) == set()
    assert set(val_df.index) & set(test_df.index) == set()

    overall_rate = sample_df[TARGET].mean()
    for split in (train_df, val_df, test_df):
        assert abs(split[TARGET].mean() - overall_rate) < 0.08  # small fixture, looser tolerance


def test_logistic_regression_pipeline_fits_and_predicts_probabilities(sample_df):
    train_df, val_df, _ = split_data(sample_df)
    X_train, y_train = xy(train_df)
    X_val, y_val = xy(val_df)

    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_val)[:, 1]

    assert prob.shape[0] == len(val_df)
    assert (prob >= 0).all() and (prob <= 1).all()


def test_compute_scale_pos_weight_matches_class_ratio():
    y = np.array([0, 0, 0, 1])
    assert compute_scale_pos_weight(y) == 3.0


def test_classification_metrics_shape():
    y_true = np.array([0, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.4, 0.9, 0.6, 0.3])
    m = classification_metrics(y_true, y_prob)
    assert 0 <= m["roc_auc"] <= 1
    assert 0 <= m["pr_auc"] <= 1
    assert m["confusion_matrix"]["tp"] + m["confusion_matrix"]["fn"] == 2


def test_precision_recall_at_k_perfect_ranking():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.9, 0.8])  # perfectly ranks positives first
    result = precision_recall_at_k(y_true, y_prob, k_fractions=(0.4,))
    row = result.iloc[0]
    assert row["precision_at_k"] == 1.0
    assert row["recall_at_k"] == 1.0


@requires_real_dataset
def test_split_data_prevalence_close_to_overall_on_real_dataset():
    df = load_raw("data/raw/fraud_oracle.csv")
    train_df, val_df, test_df = split_data(df)
    overall_rate = df[TARGET].mean()
    for split in (train_df, val_df, test_df):
        assert abs(split[TARGET].mean() - overall_rate) < 0.01

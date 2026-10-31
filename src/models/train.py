"""Stratified splitting and model pipeline construction.

Split strategy: 60% train / 20% validation / 20% test, stratified on the target to
preserve the ~6% fraud-flag prevalence in every split. The test split is held out and
must only be scored once, at final evaluation (see docs/model_evaluation.md).
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.schema import TARGET
from src.features.build_features import build_preprocessor, get_feature_frame

RANDOM_STATE = 42


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
    """Return (train_df, val_df, test_df), each containing features + target,
    stratified on FraudFound_P. val_size is the fraction of the *original* data.
    """
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[TARGET],
        random_state=RANDOM_STATE,
    )
    return train_df, val_df, test_df


def xy(df: pd.DataFrame):
    X = get_feature_frame(df)
    y = df[TARGET].to_numpy()
    return X, y


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(scale_pos_weight: float) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=False)),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="aucpr",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def compute_scale_pos_weight(y_train) -> float:
    """XGBoost's scale_pos_weight = negative_count / positive_count on the TRAINING fold only."""
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    return negatives / positives

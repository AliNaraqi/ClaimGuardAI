import pandas as pd

from src.data.schema import IDENTIFIER_COLUMNS, TARGET
from src.features.build_features import (
    build_preprocessor,
    engineer_features,
    get_feature_frame,
)


def test_engineer_features_recodes_age_sentinel(sample_df):
    n_sentinel = (sample_df["Age"] == 0).sum()
    assert n_sentinel > 0  # fixture deliberately injects this quirk

    engineered = engineer_features(sample_df)
    assert engineered["Age"].isna().sum() == n_sentinel
    assert engineered["AgeUnknown"].sum() == n_sentinel
    assert set(engineered["AgeUnknown"].unique()) <= {0, 1}


def test_get_feature_frame_excludes_identifiers_and_target(sample_df):
    features = get_feature_frame(sample_df)
    for col in IDENTIFIER_COLUMNS + [TARGET]:
        assert col not in features.columns


def test_preprocessor_fits_and_transforms_without_nan_leak(sample_df):
    X = get_feature_frame(sample_df)
    pre = build_preprocessor(scale_numeric=True)
    transformed = pre.fit_transform(X)
    # dense or sparse — either way, no NaNs should remain after imputation
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not pd.isna(arr).any()
    assert arr.shape[0] == len(sample_df)


def test_preprocessor_handles_unknown_categories_at_transform_time(sample_df):
    """A category never seen in training (e.g. a new Make) must not crash scoring."""
    X = get_feature_frame(sample_df)
    train_X, score_X = X.iloc[:100].copy(), X.iloc[100:].copy()
    score_X = score_X.copy()
    score_X["Make"] = "SomeBrandNewMakeNeverSeenInTraining"

    pre = build_preprocessor(scale_numeric=True)
    pre.fit(train_X)
    transformed = pre.transform(score_X)  # should not raise
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert arr.shape[0] == len(score_X)

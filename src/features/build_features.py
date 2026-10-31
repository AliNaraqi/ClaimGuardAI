"""Feature engineering and the train-only preprocessing pipeline.

Design notes (see docs/data_dictionary.md and notebooks/01_eda.ipynb):
- `PolicyNumber`/`RepNumber` are identifiers and are dropped before modeling.
- `Age == 0` is a known sentinel (always co-occurs with AgeOfPolicyHolder == '16 to 17'),
  not a literal age. `engineer_features` recodes it to NaN plus an explicit `AgeUnknown`
  flag; this recoding is a deterministic per-row fix (no aggregate statistic), so it is
  safe to apply before the train/test split without leaking information.
- All imputation/scaling/encoding statistics are fit only on the training fold via the
  returned scikit-learn pipeline, never on validation/test data.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import (
    AGE_SENTINEL,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)

ENGINEERED_NUMERIC_COLUMNS = NUMERIC_COLUMNS + ["AgeUnknown"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply row-wise, split-safe feature fixes. Returns a new dataframe."""
    out = df.copy()
    out["AgeUnknown"] = (out["Age"] == AGE_SENTINEL).astype(int)
    out.loc[out["Age"] == AGE_SENTINEL, "Age"] = pd.NA
    out["Age"] = out["Age"].astype("float64")
    return out


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers/target, keep only model features, after engineering."""
    engineered = engineer_features(df)
    return engineered[ENGINEERED_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build a ColumnTransformer. All statistics are fit only when .fit() is called
    on the training split — never on validation/test data.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, ENGINEERED_NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )

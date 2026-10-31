import os

import pandas as pd
import pytest

from src.data.schema import validate_schema

FIXTURE_PATH = "tests/fixtures/sample_claims.csv"
REAL_DATA_PATH = "data/raw/fraud_oracle.csv"

requires_real_dataset = pytest.mark.skipif(
    not os.path.exists(REAL_DATA_PATH),
    reason="requires the licensed Kaggle fraud_oracle.csv at data/raw/ (not committed to the repo)",
)

requires_trained_model = pytest.mark.skipif(
    not os.path.exists("models/selected_pipeline_xgboost.joblib"),
    reason="requires a trained model artifact from notebooks/02_baseline_modeling.ipynb",
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small, fully synthetic, schema-conformant fixture (not derived from the real
    dataset) used so unit tests run in CI without the licensed Kaggle CSV.
    """
    df = pd.read_csv(FIXTURE_PATH)
    validate_schema(df)
    return df

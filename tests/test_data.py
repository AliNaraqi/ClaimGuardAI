import pytest

from src.data.load import load_raw
from src.data.schema import TARGET, validate_schema
from tests.conftest import requires_real_dataset


def test_validate_schema_accepts_sample_fixture(sample_df):
    validate_schema(sample_df)  # should not raise


def test_validate_schema_rejects_missing_target(sample_df):
    broken = sample_df.drop(columns=[TARGET])
    with pytest.raises(AssertionError):
        validate_schema(broken)


def test_validate_schema_rejects_non_binary_target(sample_df):
    broken = sample_df.copy()
    broken.loc[broken.index[0], TARGET] = 2
    with pytest.raises(AssertionError):
        validate_schema(broken)


def test_validate_schema_rejects_duplicate_policy_numbers(sample_df):
    broken = sample_df.copy()
    broken.loc[broken.index[1], "PolicyNumber"] = broken.loc[broken.index[0], "PolicyNumber"]
    with pytest.raises(AssertionError):
        validate_schema(broken)


@requires_real_dataset
def test_load_raw_on_real_dataset_matches_known_shape():
    df = load_raw("data/raw/fraud_oracle.csv")
    assert len(df) == 15420
    assert set(df[TARGET].unique()) <= {0, 1}

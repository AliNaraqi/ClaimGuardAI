"""Load and validate fraud_oracle.csv."""

import pandas as pd

from src.data.schema import validate_schema


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_schema(df)
    return df

"""Schema definition for fraud_oracle.csv, derived from docs/data_dictionary.md."""

TARGET = "FraudFound_P"

# Pure row/entity identifiers — never legitimate predictive features.
IDENTIFIER_COLUMNS = ["PolicyNumber", "RepNumber"]

NUMERIC_COLUMNS = [
    "WeekOfMonth",
    "WeekOfMonthClaimed",
    "Age",
    "Deductible",
    "DriverRating",
    "Year",
]

CATEGORICAL_COLUMNS = [
    "Month",
    "DayOfWeek",
    "Make",
    "AccidentArea",
    "DayOfWeekClaimed",
    "MonthClaimed",
    "Sex",
    "MaritalStatus",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "VehiclePrice",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    "AgeOfPolicyHolder",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "NumberOfSuppliments",
    "AddressChange_Claim",
    "NumberOfCars",
    "BasePolicy",
]

ALL_FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

EXPECTED_COLUMNS = set(ALL_FEATURE_COLUMNS + IDENTIFIER_COLUMNS + [TARGET])

# Sentinel values observed in EDA (notebooks/01_eda.ipynb) that are not literal values.
CLAIMED_DATE_SENTINEL = "0"
AGE_SENTINEL = 0


def validate_schema(df):
    """Raise AssertionError if the dataframe doesn't match the expected fraud_oracle schema."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing expected columns: {sorted(missing)}"
    assert set(df[TARGET].unique()) <= {0, 1}, "FraudFound_P must be binary 0/1"
    assert df["PolicyNumber"].is_unique, "PolicyNumber is expected to be a unique row identifier"

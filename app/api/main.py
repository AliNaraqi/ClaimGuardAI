"""ClaimGuard AI — optional scoring API.

Decision-support only: `/score` returns a model risk score for human-reviewed triage.
It never returns a fraud determination and must not be wired into any automated
claim-denial workflow. See docs/model_evaluation.md for scope and limitations.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS  # noqa: E402

MODEL_PATH = ROOT / "models" / "selected_pipeline_xgboost.joblib"

app = FastAPI(
    title="ClaimGuard AI Scoring API",
    description=(
        "Returns a model-estimated risk score for a claim, for analyst triage only. "
        "Not a fraud determination; always requires human review."
    ),
    version="0.1.0",
)

_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None and MODEL_PATH.exists():
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


class ClaimInput(BaseModel):
    Month: str
    WeekOfMonth: int
    DayOfWeek: str
    Make: str
    AccidentArea: str
    DayOfWeekClaimed: str
    MonthClaimed: str
    WeekOfMonthClaimed: int
    Sex: str
    MaritalStatus: str
    Age: int = Field(..., ge=0, le=120)
    Fault: str
    PolicyType: str
    VehicleCategory: str
    VehiclePrice: str
    Deductible: int
    DriverRating: int
    Days_Policy_Accident: str
    Days_Policy_Claim: str
    PastNumberOfClaims: str
    AgeOfVehicle: str
    AgeOfPolicyHolder: str
    PoliceReportFiled: str
    WitnessPresent: str
    AgentType: str
    NumberOfSuppliments: str
    AddressChange_Claim: str
    NumberOfCars: str
    Year: int
    BasePolicy: str


class ScoreResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    risk_score: float
    model_version: str
    disclaimer: str = (
        "This is a model estimate for analyst triage, not a fraud determination. "
        "Human review is required before any action is taken on this claim."
    )


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health():
    pipeline = _load_pipeline()
    return HealthResponse(status="ok", model_loaded=pipeline is not None)


@app.post("/score", response_model=ScoreResponse)
def score(claim: ClaimInput):
    pipeline = _load_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model artifact found at models/selected_pipeline_xgboost.joblib. "
                "Run notebooks/02_baseline_modeling.ipynb first."
            ),
        )

    row = {col: getattr(claim, col) for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS}
    row["AgeUnknown"] = 1 if row["Age"] == 0 else 0
    if row["AgeUnknown"] == 1:
        row["Age"] = None

    X = pd.DataFrame([row])
    risk_score = float(pipeline.predict_proba(X)[0, 1])

    return ScoreResponse(risk_score=risk_score, model_version="xgboost_week2_baseline")

from fastapi.testclient import TestClient

from app.api.main import app
from tests.conftest import requires_trained_model

client = TestClient(app)

VALID_PAYLOAD = {
    "Month": "Jan",
    "WeekOfMonth": 2,
    "DayOfWeek": "Monday",
    "Make": "Honda",
    "AccidentArea": "Urban",
    "DayOfWeekClaimed": "Tuesday",
    "MonthClaimed": "Jan",
    "WeekOfMonthClaimed": 2,
    "Sex": "Male",
    "MaritalStatus": "Married",
    "Age": 35,
    "Fault": "Policy Holder",
    "PolicyType": "Sedan - Collision",
    "VehicleCategory": "Sedan",
    "VehiclePrice": "20000 to 29000",
    "Deductible": 400,
    "DriverRating": 2,
    "Days_Policy_Accident": "more than 30",
    "Days_Policy_Claim": "more than 30",
    "PastNumberOfClaims": "none",
    "AgeOfVehicle": "3 years",
    "AgeOfPolicyHolder": "31 to 35",
    "PoliceReportFiled": "No",
    "WitnessPresent": "No",
    "AgentType": "External",
    "NumberOfSuppliments": "none",
    "AddressChange_Claim": "no change",
    "NumberOfCars": "1 vehicle",
    "Year": 1995,
    "BasePolicy": "Collision",
}


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_rejects_missing_field():
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["Age"]
    r = client.post("/score", json=bad_payload)
    assert r.status_code == 422


@requires_trained_model
def test_score_returns_risk_score_and_disclaimer():
    r = client.post("/score", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert "not a fraud determination" in body["disclaimer"]

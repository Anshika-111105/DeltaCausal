from fastapi.testclient import TestClient
import pytest
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "DeltaCausal" in data["service"]


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "Criteo" in data["primary_dataset"]
    assert "x_learner" in data["supported_models"]


def test_validate_endpoint():
    payload = {
        "dataset_type": "synthetic",
        "sample_size": 1000,
    }
    response = client.post("/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["total_rows"] == 1000


def test_policy_simulate_endpoint():
    payload = {
        "budget": 5000.0,
        "cost_per_intervention": 2.0,
        "margin_per_conversion": 40.0,
    }
    response = client.post("/policy/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["budget_allocated"] == 5000.0
    assert len(data["comparison_table"]) > 0


def test_assistant_ask_endpoint():
    payload = {
        "question": "What assumptions are needed for causal inference?",
    }
    response = client.post("/assistant/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "SUTVA" in " ".join(data["causal_assumptions"])
    assert len(data["cited_sources"]) > 0

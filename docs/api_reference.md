# REST API Reference: Counterfactual Uplift Engine

The application exposes a production-grade FastAPI service compliant with OpenAPI 3.1.
Interactive Swagger UI is available at `http://localhost:8000/docs`.

---

## Base Endpoints

### 1. `GET /health`
Returns service availability status and UTC timestamp.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "Counterfactual Uplift Modeling Engine",
  "version": "1.0.0",
  "timestamp_utc": "2026-10-30T13:30:00.000000"
}
```

---

### 2. `GET /metadata`
Returns system metadata, supported algorithms, and dataset details.

---

## Validation & Training

### 3. `POST /validate`
Executes data quality checks, schema conformance, balance tests, and target leakage detection.

**Request Body:**
```json
{
  "dataset_type": "synthetic",
  "sample_size": 5000
}
```

**Response (200 OK):**
```json
{
  "is_valid": true,
  "total_rows": 5000,
  "total_columns": 27,
  "duplicate_count": 0,
  "outcome_rate_treated": 0.0521,
  "outcome_rate_control": 0.0084,
  "issues": []
}
```

---

### 4. `POST /train`
Triggers full end-to-end training pipeline and persists versioned model artifacts to `artifacts/{experiment_id}`.

**Request Body:**
```json
{
  "dataset_type": "synthetic",
  "sample_size": 25000,
  "budget": 10000.0,
  "cost_per_intervention": 2.50,
  "margin_per_conversion": 50.0
}
```

---

## Inference & Policy Simulation

### 5. `POST /uplift/predict`
Calculates estimated CATE $\hat{\tau}(x)$, counterfactual probabilities $P(Y(0)), P(Y(1))$, and recommended customer archetype.

**Request Body:**
```json
{
  "features": [
    {
      "f0": 1.25, "f1": -0.40, "f2": 0.10, "f3": 0.00,
      "f4": 0.85, "f5": -0.20, "f6": 0.05, "f7": 0.10,
      "f8": -0.30, "f9": 0.15, "f10": 0.00, "f11": -0.05
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "experiment_id": "exp_20261030_133039",
  "model_used": "t_learner",
  "predictions": [
    {
      "index": 0,
      "estimated_uplift": 0.08412,
      "predicted_p0_baseline": 0.0620,
      "predicted_p1_treated": 0.1461,
      "recommended_segment": "Persuadables",
      "should_target": true
    }
  ]
}
```

---

### 6. `POST /policy/simulate`
Simulates knapsack targeting policy under custom budget, intervention cost, and margin constraints.

---

### 7. `POST /assistant/ask`
Interacts with the Grounded Causal Decision Assistant via semantic RAG and live experiment metric retrieval.

**Request Body:**
```json
{
  "question": "What is the expected ROI of increasing the marketing budget to $15,000?"
}
```


"""Pydantic Request and Response Schemas for FastAPI REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp_utc: str


class MetadataResponse(BaseModel):
    project_name: str
    version: str
    primary_dataset: str
    supported_models: List[str]
    causal_outcomes: List[str]


class ValidateRequest(BaseModel):
    dataset_type: str = Field(default="synthetic", description="'synthetic' or 'criteo'")
    file_path: Optional[str] = Field(default=None, description="Path to CSV file")
    sample_size: int = Field(default=5000, description="Records to validate")


class ValidateResponse(BaseModel):
    is_valid: bool
    total_rows: int
    total_columns: int
    duplicate_count: int
    outcome_rate_treated: float
    outcome_rate_control: float
    issues: List[Dict[str, Any]]


class TrainRequest(BaseModel):
    dataset_type: str = Field(default="synthetic")
    sample_size: int = Field(default=10000)
    models: Optional[List[str]] = Field(default=None, description="Subset of models to train")
    budget: float = Field(default=10000.0)
    cost_per_intervention: float = Field(default=2.50)
    margin_per_conversion: float = Field(default=50.0)


class TrainResponse(BaseModel):
    experiment_id: str
    best_model_name: str
    primary_metric: str
    models_trained: List[str]
    n_train_samples: int
    status: str


class PredictRequest(BaseModel):
    experiment_id: Optional[str] = Field(default=None, description="Experiment ID to load")
    model_name: Optional[str] = Field(default=None, description="Model to use (defaults to best model)")
    features: List[Dict[str, float]] = Field(
        description="List of feature dicts with keys f0 through f11",
        examples=[[{"f0": 0.5, "f1": -0.2, "f2": 0.1, "f3": 0.0, "f4": 1.2, "f5": -0.3, "f6": 0.0, "f7": 0.1, "f8": -0.4, "f9": 0.2, "f10": 0.0, "f11": -0.1}]],
    )


class UpliftPredictionItem(BaseModel):
    index: int
    estimated_uplift: float
    predicted_p0_baseline: float
    predicted_p1_treated: float
    recommended_segment: str
    should_target: bool


class PredictResponse(BaseModel):
    experiment_id: str
    model_used: str
    predictions: List[UpliftPredictionItem]


class PolicySimulateRequest(BaseModel):
    budget: float = Field(default=10000.0)
    cost_per_intervention: float = Field(default=2.50)
    margin_per_conversion: float = Field(default=50.0)
    experiment_id: Optional[str] = None


class PolicySimulateResponse(BaseModel):
    budget_allocated: float
    budget_spent: float
    customers_targeted: int
    expected_incremental_conversions: float
    expected_incremental_revenue: float
    expected_incremental_profit: float
    roi_percentage: float
    comparison_table: List[Dict[str, Any]]


class AssistantAskRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural language causal or business question",
        examples=["Which customer segment delivers the highest incremental ROI?"],
    )
    experiment_id: Optional[str] = None


class AssistantAskResponse(BaseModel):
    question: str
    direct_answer: str
    validated_metrics: Dict[str, Any]
    causal_assumptions: List[str]
    cited_sources: List[Dict[str, str]]
    next_recommended_experiment: str


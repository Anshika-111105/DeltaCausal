"""Health and metadata routes."""

from datetime import datetime
from fastapi import APIRouter
from ..schemas import HealthResponse, MetadataResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(
        status="healthy",
        service="DeltaCausal: Counterfactual Uplift Modeling Engine",
        version="1.0.0",
        timestamp_utc=datetime.utcnow().isoformat(),
    )


@router.get("/metadata", response_model=MetadataResponse)
def get_metadata() -> MetadataResponse:
    """Return platform metadata and capabilities."""
    return MetadataResponse(
        project_name="DeltaCausal: Counterfactual Business Decision & Uplift Engine",
        version="1.0.0",
        primary_dataset="Criteo Uplift Prediction Dataset v2.1",
        supported_models=[
            "s_learner",
            "t_learner",
            "x_learner",
            "doubly_robust",
            "two_model_naive",
            "response_propensity",
            "random",
        ],
        causal_outcomes=["conversion", "visit"],
    )


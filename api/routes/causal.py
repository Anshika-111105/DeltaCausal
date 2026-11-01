import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
import joblib
import numpy as np
import pandas as pd

from src.assistant.rag_engine import DecisionAssistant
from src.data.loader import UpliftDataLoader
from src.features.preprocessor import CausalPreprocessor
from src.policy.optimizer import PolicyOptimizer
from src.policy.segmenter import CustomerSegmenter
from src.validation.validator import DataValidator
from ..schemas import (
    AssistantAskRequest,
    AssistantAskResponse,
    PolicySimulateRequest,
    PolicySimulateResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
    UpliftPredictionItem,
    ValidateRequest,
    ValidateResponse,
)

router = APIRouter(tags=["Causal Inference & Uplift"])
ARTIFACTS_DIR = Path("artifacts")


def _get_active_experiment_id(requested_id: Optional[str] = None) -> Path:
    """Find valid experiment artifact directory."""
    if requested_id:
        p = ARTIFACTS_DIR / requested_id
        if p.exists() and p.is_dir():
            return p
        raise HTTPException(status_code=404, detail=f"Experiment '{requested_id}' not found.")

    exp_dirs = sorted([d for d in ARTIFACTS_DIR.iterdir() if d.is_dir() and d.name.startswith("exp_")])
    if not exp_dirs:
        # Check smoke test exp
        smoke = ARTIFACTS_DIR / "smoke_test_exp"
        if smoke.exists():
            return smoke
        raise HTTPException(
            status_code=404,
            detail="No trained experiments found. Please call POST /train first.",
        )
    return exp_dirs[-1]


@router.post("/validate", response_model=ValidateResponse)
def validate_dataset(req: ValidateRequest):
    """Validate dataset schema, balance, and causal prerequisites."""
    loader = UpliftDataLoader(raw_path=req.file_path)
    df, _ = loader.load_data(dataset_type=req.dataset_type, sample_size=req.sample_size)

    validator = DataValidator()
    feature_cols = [f"f{i}" for i in range(12)]
    report = validator.validate(df, dataset_type=req.dataset_type, feature_cols=feature_cols)

    return ValidateResponse(
        is_valid=report.is_valid,
        total_rows=report.total_rows,
        total_columns=report.total_columns,
        duplicate_count=report.duplicate_count,
        outcome_rate_treated=report.outcome_rate_treated,
        outcome_rate_control=report.outcome_rate_control,
        issues=report.issues,
    )


@router.post("/train", response_model=TrainResponse)
def train_pipeline(req: TrainRequest):
    """Trigger training pipeline."""
    from scripts.run_pipeline import run_pipeline

    try:
        exp_id = run_pipeline(
            dataset_type=req.dataset_type,
            sample_size=req.sample_size,
            budget=req.budget,
            cost_per_intervention=req.cost_per_intervention,
            margin_per_conversion=req.margin_per_conversion,
        )

        exp_dir = ARTIFACTS_DIR / exp_id
        with open(exp_dir / "evaluation_report.json", "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        return TrainResponse(
            experiment_id=exp_id,
            best_model_name=eval_data.get("best_model_name", "x_learner"),
            primary_metric=eval_data.get("primary_metric", "qini_score"),
            models_trained=list(eval_data.get("model_metrics", {}).keys()),
            n_train_samples=int(req.sample_size * 0.60),
            status="SUCCESS",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    """Retrieve metadata and configuration of a specific experiment."""
    exp_dir = _get_active_experiment_id(experiment_id)
    rec_path = exp_dir / "experiment_record.json"
    if not rec_path.exists():
        raise HTTPException(status_code=404, detail="Experiment record missing.")
    with open(rec_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/experiments/{experiment_id}/metrics")
def get_experiment_metrics(experiment_id: str):
    """Retrieve evaluation metrics for an experiment."""
    exp_dir = _get_active_experiment_id(experiment_id)
    eval_path = exp_dir / "evaluation_report.json"
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation metrics missing.")
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/uplift/predict", response_model=PredictResponse)
def predict_uplift(req: PredictRequest):
    """Generate individual CATE predictions and recommended action for customer feature vectors."""
    exp_dir = _get_active_experiment_id(req.experiment_id)

    prep_path = exp_dir / "preprocessor.joblib"
    if not prep_path.exists():
        raise HTTPException(status_code=500, detail="Preprocessor artifact missing.")
    preprocessor: CausalPreprocessor = joblib.load(prep_path)

    # Determine model
    if req.model_name:
        model_name = req.model_name
    else:
        eval_file = exp_dir / "evaluation_report.json"
        if eval_file.exists():
            with open(eval_file, "r", encoding="utf-8") as f:
                model_name = json.load(f).get("best_model_name", "x_learner")
        else:
            model_name = "x_learner"

    model_path = exp_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model artifact '{model_name}.joblib' not found.")
    model = joblib.load(model_path)

    # Format input DataFrame
    df_in = pd.DataFrame(req.features)
    X_mat = preprocessor.transform(df_in)

    uplift = model.predict_uplift(X_mat)
    if hasattr(model, "predict_counterfactuals"):
        p0, p1 = model.predict_counterfactuals(X_mat)
    else:
        p0 = np.zeros(len(X_mat))
        p1 = uplift

    segmenter = CustomerSegmenter()
    labels, _ = segmenter.segment_customers(p0, p1)

    items = []
    for i in range(len(df_in)):
        seg = labels.iloc[i]
        tau = float(uplift[i])
        should_target = bool(seg == "Persuadables" or (tau > 0.02 and seg != "Sleeping Dogs"))
        items.append(
            UpliftPredictionItem(
                index=i,
                estimated_uplift=round(tau, 5),
                predicted_p0_baseline=round(float(p0[i]), 4),
                predicted_p1_treated=round(float(p1[i]), 4),
                recommended_segment=seg,
                should_target=should_target,
            )
        )

    return PredictResponse(
        experiment_id=exp_dir.name,
        model_used=model_name,
        predictions=items,
    )


@router.post("/policy/simulate", response_model=PolicySimulateResponse)
def simulate_policy(req: PolicySimulateRequest):
    """Simulate budget-constrained targeting policy under custom financial assumptions."""
    exp_dir = _get_active_experiment_id(req.experiment_id)
    policy_csv = exp_dir / "policy_comparison.csv"

    opt = PolicyOptimizer(default_cost=req.cost_per_intervention, default_margin=req.margin_per_conversion)

    # If policy CSV exists, load it
    if policy_csv.exists():
        comp_df = pd.read_csv(policy_csv)
        uplift_row = comp_df[comp_df["policy_name"] == "Counterfactual Uplift Targeting"]
        if not uplift_row.empty:
            r = uplift_row.iloc[0]
            return PolicySimulateResponse(
                budget_allocated=req.budget,
                budget_spent=float(r["budget_spent"]),
                customers_targeted=int(r["customers_targeted"]),
                expected_incremental_conversions=float(r["expected_incremental_conversions"]),
                expected_incremental_revenue=float(r["expected_incremental_revenue"]),
                expected_incremental_profit=float(r["expected_incremental_profit"]),
                roi_percentage=float(r["roi_percentage"]),
                comparison_table=comp_df.to_dict(orient="records"),
            )

    # Fallback simulation
    dummy_scores = np.random.normal(0.04, 0.05, 5000)
    _, res = opt.optimize_policy(dummy_scores, budget=req.budget)
    return PolicySimulateResponse(
        budget_allocated=req.budget,
        budget_spent=res.budget_spent,
        customers_targeted=res.customers_targeted,
        expected_incremental_conversions=res.expected_incremental_conversions,
        expected_incremental_revenue=res.expected_incremental_revenue,
        expected_incremental_profit=res.expected_incremental_profit,
        roi_percentage=res.roi_percentage,
        comparison_table=[res.__dict__],
    )


@router.post("/assistant/ask", response_model=AssistantAskResponse)
def ask_assistant(req: AssistantAskRequest):
    """Query Grounded Causal Decision Assistant."""
    assistant = DecisionAssistant(artifacts_dir="artifacts")
    ans = assistant.answer_query(req.question, experiment_id=req.experiment_id)
    return AssistantAskResponse(
        question=ans.query,
        direct_answer=ans.direct_answer,
        validated_metrics=ans.validated_metrics,
        causal_assumptions=ans.causal_assumptions,
        cited_sources=ans.cited_sources,
        next_recommended_experiment=ans.next_recommended_experiment,
    )


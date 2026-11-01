import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from src.causal.diagnostics import CausalDiagnostics
from src.data.loader import UpliftDataLoader
from src.data.splitter import CausalDataSplitter
from src.evaluation.evaluator import UpliftEvaluator
from src.models.trainer import ModelTrainer
from src.policy.optimizer import PolicyOptimizer
from src.policy.segmenter import CustomerSegmenter
from src.validation.validator import DataValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    dataset_type: str = "synthetic",
    sample_size: int = 50000,
    raw_path: str = "data/raw/criteo_sample.csv",
    budget: float = 10000.0,
    cost_per_intervention: float = 2.50,
    margin_per_conversion: float = 50.0,
    artifacts_dir: str = "artifacts",
    reports_dir: str = "reports",
    random_seed: int = 42,
    observational_bias: bool = False,
) -> str:
    """Execute end-to-end causal uplift pipeline and return experiment_id."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{ts}"
    exp_dir = Path(artifacts_dir) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting Uplift Pipeline Execution: {experiment_id} ===")

    # 1. Ingestion
    loader = UpliftDataLoader(
        raw_path=raw_path,
        synthetic_path="data/synthetic/synthetic_uplift.csv",
        random_seed=random_seed,
    )
    df, metadata = loader.load_data(
        dataset_type=dataset_type,
        sample_size=sample_size,
        observational_bias=observational_bias,
    )
    logger.info(f"Dataset loaded: {metadata.dataset_name} ({len(df):,} rows)")

    # 2. Validation
    validator = DataValidator()
    feature_cols = [f"f{i}" for i in range(12)]
    val_report = validator.validate(
        df=df,
        dataset_type=dataset_type,
        treatment_col="treatment",
        outcome_col="conversion",
        feature_cols=feature_cols,
    )
    val_report.to_json(exp_dir / "validation_report.json")
    with open(Path(reports_dir) / "data_validation_report.md", "w", encoding="utf-8") as f:
        f.write(val_report.summary_markdown())

    if not val_report.is_valid:
        logger.error("Validation failed. Halting pipeline.")
        raise ValueError("Data validation failed. See validation report for details.")

    # 3. Causal Diagnostics
    diagnostics = CausalDiagnostics(smd_threshold=0.10)
    balance_metrics = diagnostics.check_covariate_balance(df, feature_cols, "treatment")
    srm_result = diagnostics.check_sample_ratio_mismatch(df, expected_treatment_ratio=0.85, treatment_col="treatment")
    overlap_res, prop_scores = diagnostics.check_propensity_overlap(df, feature_cols, "treatment")

    diag_dict = {
        "covariate_balance": [b.__dict__ for b in balance_metrics],
        "srm_test": srm_result.__dict__,
        "propensity_overlap": overlap_res.__dict__,
    }
    with open(exp_dir / "causal_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag_dict, f, indent=2)

    # 4. Stratified Train / Val / Test Splitting
    splitter = CausalDataSplitter(train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_state=random_seed)
    splits = splitter.split(df, treatment_col="treatment", outcome_col="conversion", stratify=True)
    logger.info(f"Splits created: Train={len(splits.train_df):,}, Val={len(splits.val_df):,}, Test={len(splits.test_df):,}")

    # Persist processed partitions to data/processed/
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits.train_df.to_csv(processed_dir / "train.csv", index=False)
    splits.val_df.to_csv(processed_dir / "val.csv", index=False)
    splits.test_df.to_csv(processed_dir / "test.csv", index=False)
    with open(processed_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(splits.summary(), f, indent=2)
    logger.info(f"Persisted processed train/val/test splits to {processed_dir.resolve()}")

    # 5. Model Training on Train Set
    trainer = ModelTrainer(artifacts_root=artifacts_dir, random_seed=random_seed)
    models, preprocessor, train_record = trainer.train_all(
        train_df=splits.train_df,
        feature_cols=feature_cols,
        treatment_col="treatment",
        outcome_col="conversion",
        experiment_id=experiment_id,
    )

    # 6. Model Evaluation on Validation Set
    evaluator = UpliftEvaluator(primary_metric="qini_score", n_bootstraps=50, random_seed=random_seed)
    val_comparison, val_eval_report, val_scores = evaluator.evaluate_models(
        models=models,
        preprocessor=preprocessor,
        eval_df=splits.val_df,
        experiment_id=experiment_id,
        split_name="validation",
        margin_per_conversion=margin_per_conversion,
        cost_per_intervention=cost_per_intervention,
    )
    val_comparison.to_csv(exp_dir / "val_model_comparison.csv")
    with open(exp_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(val_eval_report.__dict__, f, indent=2, default=str)

    best_model_name = val_eval_report.best_model_name
    logger.info(f"Selected Best Uplift Model: {best_model_name}")

    # 7. Final Model Evaluation on Held-Out Test Set
    test_comparison, test_eval_report, test_scores = evaluator.evaluate_models(
        models={best_model_name: models[best_model_name], "response_propensity": models["response_propensity"], "random": models["random"]},
        preprocessor=preprocessor,
        eval_df=splits.test_df,
        experiment_id=experiment_id,
        split_name="test",
        margin_per_conversion=margin_per_conversion,
        cost_per_intervention=cost_per_intervention,
    )
    test_comparison.to_csv(exp_dir / "test_model_comparison.csv")

    # 8. Customer Segmentation
    best_model = models[best_model_name]
    X_test = preprocessor.transform(splits.test_df)
    p0_pred, p1_pred = best_model.predict_counterfactuals(X_test)
    segmenter = CustomerSegmenter()
    seg_labels, seg_summaries = segmenter.segment_customers(p0_pred, p1_pred)

    with open(exp_dir / "customer_segments.json", "w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in seg_summaries], f, indent=2)

    # 9. Budget-Constrained Policy Optimization on Test Set
    optimizer = PolicyOptimizer(default_cost=cost_per_intervention, default_margin=margin_per_conversion)
    test_best_scores = test_scores[best_model_name]
    test_prop_scores = test_scores["response_propensity"]
    true_cate_test = splits.test_df["true_cate"].values if "true_cate" in splits.test_df.columns else None

    policy_comp_df = optimizer.compare_targeting_policies(
        uplift_scores=test_best_scores,
        response_propensity_scores=test_prop_scores,
        true_cate=true_cate_test,
        budget=budget,
        cost_per_intervention=cost_per_intervention,
        margin_per_conversion=margin_per_conversion,
        random_seed=random_seed,
    )
    policy_comp_df.to_csv(exp_dir / "policy_comparison.csv", index=False)
    with open(exp_dir / "policy_comparison.json", "w", encoding="utf-8") as f:
        json.dump(policy_comp_df.to_dict(orient="records"), f, indent=2)

    # 10. Sensitivity Analysis
    sensitivity_df = optimizer.run_sensitivity_grid(
        uplift_scores=test_best_scores,
        base_budget=budget,
        base_cost=cost_per_intervention,
        base_margin=margin_per_conversion,
    )
    sensitivity_df.to_csv(exp_dir / "sensitivity_analysis.csv", index=False)

    # 11. Generate Markdown Reports
    _generate_markdown_reports(
        reports_dir=Path(reports_dir),
        experiment_id=experiment_id,
        val_comparison=val_comparison,
        test_comparison=test_comparison,
        policy_comp_df=policy_comp_df,
        seg_summaries=seg_summaries,
        val_eval_report=val_eval_report,
        diag_dict=diag_dict,
        budget=budget,
        cost=cost_per_intervention,
        margin=margin_per_conversion,
    )

    logger.info(f"=== Pipeline Finished Successfully: {experiment_id} ===")
    return experiment_id


def _generate_markdown_reports(
    reports_dir: Path,
    experiment_id: str,
    val_comparison: pd.DataFrame,
    test_comparison: pd.DataFrame,
    policy_comp_df: pd.DataFrame,
    seg_summaries: list,
    val_eval_report: Any,
    diag_dict: dict,
    budget: float,
    cost: float,
    margin: float,
):
    """Write comprehensive human-readable reports to reports/ directory."""
    # Model evaluation report
    model_eval_md = [
        f"# Model Evaluation & Benchmark Report: `{experiment_id}`",
        f"- **Primary Selection Metric**: `{val_eval_report.primary_metric}`",
        f"- **Winning Model**: `{val_eval_report.best_model_name}`",
        "",
        "## Validation Set Comparison",
        val_comparison.to_markdown(),
        "",
        "## Held-Out Test Set Performance",
        test_comparison.to_markdown(),
        "",
        "## Bootstrap Confidence Intervals (95% CI)",
    ]
    for m, ci in val_eval_report.bootstrap_intervals.items():
        model_eval_md.append(f"- **{m}**: Qini Score CI: `{ci.get('qini_score_ci')}`, AUUC CI: `{ci.get('auuc_ci')}`")

    with open(reports_dir / "model_evaluation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(model_eval_md))

    # Policy simulation report
    policy_md = [
        f"# Campaign Policy Simulation & Optimization Report: `{experiment_id}`",
        f"- **Budget**: ${budget:,.2f}",
        f"- **Intervention Cost per Customer**: ${cost:.2f}",
        f"- **Gross Margin per Conversion**: ${margin:.2f}",
        "",
        "## Targeting Policy Comparison",
        policy_comp_df.to_markdown(index=False),
        "",
        "## Customer Segmentation Breakdown",
    ]
    for seg in seg_summaries:
        policy_md.append(
            f"### {seg.segment_name} ({seg.percentage}% of cohort, {seg.count:,} customers)\n"
            f"- **Recommendation**: `{seg.targeting_recommendation}`\n"
            f"- **Avg Predicted Uplift**: {seg.avg_estimated_uplift:+.4f}\n"
            f"- **Avg Baseline Rate P(Y(0))**: {seg.avg_predicted_baseline:.4f}\n"
            f"- **Action**: {seg.action_description}\n"
        )

    with open(reports_dir / "policy_simulation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(policy_md))

    # Experiment summary report
    summary_md = [
        f"# End-to-End Experiment Summary: `{experiment_id}`",
        f"- **Run Date**: {datetime.utcnow().isoformat()} UTC",
        f"- **Best Model**: `{val_eval_report.best_model_name}`",
        f"- **Test Incremental Conversions**: {policy_comp_df.loc[policy_comp_df['policy_name']=='Counterfactual Uplift Targeting', 'expected_incremental_conversions'].values[0]:,.1f}",
        f"- **Test Incremental Profit**: ${policy_comp_df.loc[policy_comp_df['policy_name']=='Counterfactual Uplift Targeting', 'expected_incremental_profit'].values[0]:,.2f}",
        f"- **ROI**: {policy_comp_df.loc[policy_comp_df['policy_name']=='Counterfactual Uplift Targeting', 'roi_percentage'].values[0]:.1f}%",
        "",
        "## Causal Validity Summary",
        f"- SRM Test: {'PASSED' if not diag_dict['srm_test']['is_srm_detected'] else 'FLAGGED'}",
        f"- Propensity Overlap: {'ADEQUATE' if diag_dict['propensity_overlap']['is_overlap_adequate'] else 'TRIMMING REQUIRED'}",
    ]
    with open(reports_dir / "experiment_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md))


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end uplift modeling and policy optimization pipeline.")
    parser.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "criteo"], help="Dataset type")
    parser.add_argument("--sample-size", type=int, default=50000, help="Sample size for training and evaluation")
    parser.add_argument("--raw-path", type=str, default="data/raw/criteo_sample.csv", help="Path to raw dataset")
    parser.add_argument("--budget", type=float, default=10000.0, help="Marketing campaign budget")
    parser.add_argument("--cost", type=float, default=2.50, help="Intervention cost per customer")
    parser.add_argument("--margin", type=float, default=50.0, help="Margin per conversion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--observational", action="store_true", help="Simulate observational confounding")
    args = parser.parse_args()

    exp_id = run_pipeline(
        dataset_type=args.dataset_type,
        sample_size=args.sample_size,
        raw_path=args.raw_path,
        budget=args.budget,
        cost_per_intervention=args.cost,
        margin_per_conversion=args.margin,
        random_seed=args.seed,
        observational_bias=args.observational,
    )
    print(f"EXPERIMENT_SUCCESS: {exp_id}")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assistant.rag_engine import DecisionAssistant
from src.causal.diagnostics import CausalDiagnostics
from src.data.loader import UpliftDataLoader
from src.data.splitter import CausalDataSplitter
from src.evaluation.evaluator import UpliftEvaluator
from src.models.trainer import ModelTrainer
from src.policy.optimizer import PolicyOptimizer
from src.policy.segmenter import CustomerSegmenter
from src.validation.validator import DataValidator


def run_smoke_test():
    start_time = time.time()
    print("=== [SMOKE TEST] 1. Ingestion ===")
    loader = UpliftDataLoader(random_seed=42)
    df, meta = loader.load_data(dataset_type="synthetic", sample_size=2000, force_regenerate=True)
    assert len(df) == 2000, f"Expected 2000 rows, got {len(df)}"
    assert "treatment" in df.columns and "conversion" in df.columns

    print("=== [SMOKE TEST] 2. Validation ===")
    validator = DataValidator()
    feature_cols = [f"f{i}" for i in range(12)]
    val_report = validator.validate(df, dataset_type="synthetic", feature_cols=feature_cols)
    assert val_report.is_valid, f"Validation failed with issues: {val_report.issues}"

    print("=== [SMOKE TEST] 3. Causal Diagnostics ===")
    diag = CausalDiagnostics()
    smds = diag.check_covariate_balance(df, feature_cols)
    assert len(smds) == 12
    srm = diag.check_sample_ratio_mismatch(df, expected_treatment_ratio=0.85)
    overlap, prop_scores = diag.check_propensity_overlap(df, feature_cols)
    assert len(prop_scores) == 2000

    print("=== [SMOKE TEST] 4. Splitting ===")
    splitter = CausalDataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42)
    splits = splitter.split(df, stratify=True)
    assert len(splits.train_df) == 1200
    assert len(splits.val_df) == 400
    assert len(splits.test_df) == 400

    print("=== [SMOKE TEST] 5. Model Training ===")
    trainer = ModelTrainer(artifacts_root="artifacts", random_seed=42)
    models, preprocessor, record = trainer.train_all(
        train_df=splits.train_df,
        feature_cols=feature_cols,
        experiment_id="smoke_test_exp",
        model_types=["s_learner", "t_learner", "x_learner", "two_model_naive", "response_propensity", "random"],
    )
    assert "x_learner" in models
    assert preprocessor.is_fitted

    print("=== [SMOKE TEST] 6. Uplift Evaluation ===")
    evaluator = UpliftEvaluator(n_bootstraps=5, random_seed=42)
    val_df, eval_report, scores = evaluator.evaluate_models(
        models=models,
        preprocessor=preprocessor,
        eval_df=splits.val_df,
        experiment_id="smoke_test_exp",
    )
    assert len(val_df) > 0
    assert eval_report.best_model_name in models

    print("=== [SMOKE TEST] 7. Segmentation ===")
    best_m = models[eval_report.best_model_name]
    X_test = preprocessor.transform(splits.test_df)
    p0, p1 = best_m.predict_counterfactuals(X_test)
    segmenter = CustomerSegmenter()
    labels, summaries = segmenter.segment_customers(p0, p1)
    assert len(labels) == 400
    assert len(summaries) == 4

    print("=== [SMOKE TEST] 8. Policy Optimization ===")
    optimizer = PolicyOptimizer()
    pol_df = optimizer.compare_targeting_policies(
        uplift_scores=scores[eval_report.best_model_name][:400],
        response_propensity_scores=scores["response_propensity"][:400],
        budget=1000.0,
    )
    assert len(pol_df) >= 3

    print("=== [SMOKE TEST] 9. Decision Assistant ===")
    assistant = DecisionAssistant(artifacts_dir="artifacts")
    ans = assistant.answer_query("Which model performs best?", experiment_id="smoke_test_exp")
    assert len(ans.direct_answer) > 20
    assert len(ans.cited_sources) > 0

    elapsed = time.time() - start_time
    print(f"\n>>> [SMOKE TEST PASSED] All 9 stages verified successfully in {elapsed:.2f}s! <<<")


if __name__ == "__main__":
    run_smoke_test()


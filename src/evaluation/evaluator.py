"""Comprehensive Model Evaluator and Benchmark Comparison Engine.

Evaluates all candidate models on held-out validation and test sets, computes bootstrap
confidence intervals, logs results, and generates publication-quality Plotly figures.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from ..features.preprocessor import CausalPreprocessor
from .metrics import (
    calculate_auuc,
    calculate_pehe,
    calculate_qini_curve,
    calculate_qini_score,
    calculate_uplift_curve,
    evaluate_all_uplift_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    experiment_id: str
    evaluated_split: str  # "validation" or "test"
    model_metrics: Dict[str, Dict[str, float]]
    best_model_name: str
    primary_metric: str
    bootstrap_intervals: Dict[str, Dict[str, Tuple[float, float]]]


class UpliftEvaluator:
    """Evaluates and compares uplift learners against baselines."""

    def __init__(
        self,
        primary_metric: str = "qini_score",
        n_bootstraps: int = 50,
        random_seed: int = 42,
    ):
        self.primary_metric = primary_metric
        self.n_bootstraps = n_bootstraps
        self.random_seed = random_seed

    def evaluate_models(
        self,
        models: Dict[str, Any],
        preprocessor: CausalPreprocessor,
        eval_df: pd.DataFrame,
        experiment_id: str = "default_exp",
        split_name: str = "val",
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
        true_cate_col: Optional[str] = "true_cate",
        margin_per_conversion: float = 50.0,
        cost_per_intervention: float = 2.50,
    ) -> Tuple[pd.DataFrame, EvaluationReport, Dict[str, np.ndarray]]:
        """Evaluate a dictionary of models on eval_df.

        Returns
        -------
        Tuple[pd.DataFrame, EvaluationReport, Dict[str, np.ndarray]]
            Comparison DataFrame, EvaluationReport, and dict of predicted uplift scores.
        """
        X_eval = preprocessor.transform(eval_df)
        y_eval = eval_df[outcome_col].values
        t_eval = eval_df[treatment_col].values
        true_cate = eval_df[true_cate_col].values if true_cate_col in eval_df.columns else None

        scores_dict: Dict[str, np.ndarray] = {}
        metrics_dict: Dict[str, Dict[str, float]] = {}

        for name, model in models.items():
            if name == "propensity":
                continue  # Propensity is a diagnostic model, not an uplift targeting model

            logger.info(f"Evaluating model: {name} on {split_name} set...")
            uplift_scores = model.predict_uplift(X_eval)
            scores_dict[name] = uplift_scores

            res = evaluate_all_uplift_metrics(
                y_true=y_eval,
                treatment=t_eval,
                uplift_scores=uplift_scores,
                true_cate=true_cate,
                margin_per_conversion=margin_per_conversion,
                cost_per_intervention=cost_per_intervention,
            )
            metrics_dict[name] = res

        comparison_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
        comparison_df.index.name = "model"

        # Determine best model based on primary metric
        # Exclude non-uplift baselines and synthetic oracle from best uplift model selection
        uplift_model_types = {"s_learner", "t_learner", "x_learner", "doubly_robust", "two_model_naive"}
        candidate_models = [m for m in metrics_dict.keys() if m in uplift_model_types]
        if not candidate_models:
            candidate_models = [m for m in metrics_dict.keys() if m != "oracle"]
        best_model = max(candidate_models, key=lambda m: metrics_dict[m].get(self.primary_metric, -1e9))

        # Bootstrap confidence intervals for best model and key learners
        bootstrap_results: Dict[str, Dict[str, Tuple[float, float]]] = {}
        if self.n_bootstraps > 0:
            logger.info(f"Computing bootstrap confidence intervals (B={self.n_bootstraps})...")
            rng = np.random.default_rng(self.random_seed)
            n_eval = len(eval_df)

            for m_name in [best_model, "two_model_naive", "response_propensity"]:
                if m_name not in scores_dict:
                    continue
                qini_boots = []
                auuc_boots = []
                m_scores = scores_dict[m_name]

                for _ in range(self.n_bootstraps):
                    boot_idx = rng.choice(n_eval, size=n_eval, replace=True)
                    y_b = y_eval[boot_idx]
                    t_b = t_eval[boot_idx]
                    s_b = m_scores[boot_idx]

                    qini_boots.append(calculate_qini_score(y_b, t_b, s_b))
                    auuc_boots.append(calculate_auuc(y_b, t_b, s_b))

                bootstrap_results[m_name] = {
                    "qini_score_ci": (
                        round(float(np.percentile(qini_boots, 2.5)), 4),
                        round(float(np.percentile(qini_boots, 97.5)), 4),
                    ),
                    "auuc_ci": (
                        round(float(np.percentile(auuc_boots, 2.5)), 4),
                        round(float(np.percentile(auuc_boots, 97.5)), 4),
                    ),
                }

        report = EvaluationReport(
            experiment_id=experiment_id,
            evaluated_split=split_name,
            model_metrics=metrics_dict,
            best_model_name=best_model,
            primary_metric=self.primary_metric,
            bootstrap_intervals=bootstrap_results,
        )

        return comparison_df, report, scores_dict

    def plot_qini_curves(
        self,
        y_true: np.ndarray,
        treatment: np.ndarray,
        scores_dict: Dict[str, np.ndarray],
        title: str = "Qini Curves Comparison",
    ) -> go.Figure:
        """Generate interactive Plotly Qini curve visualization."""
        fig = go.Figure()

        # Add random baseline once
        first_scores = next(iter(scores_dict.values()))
        fractions, _, baseline = calculate_qini_curve(y_true, treatment, first_scores)
        fig.add_trace(
            go.Scatter(
                x=fractions,
                y=baseline,
                mode="lines",
                name="Random Baseline",
                line=dict(dash="dash", color="gray", width=2),
            )
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i, (name, scores) in enumerate(scores_dict.items()):
            _, qini_vals, _ = calculate_qini_curve(y_true, treatment, scores)
            fig.add_trace(
                go.Scatter(
                    x=fractions,
                    y=qini_vals,
                    mode="lines",
                    name=name.replace("_", " ").title(),
                    line=dict(color=colors[i % len(colors)], width=2.5),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Fraction of Targeted Customers",
            yaxis_title="Cumulative Incremental Gains (Qini)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        return fig

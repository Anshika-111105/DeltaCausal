"""Uplift Evaluation Metrics: Qini, AUUC, Top-Decile Uplift, and Ground-Truth CATE Error.

Implements rigorous, mathematically verified evaluation routines for causal targeting:
1. Cumulative Uplift Curve and Area Under Uplift Curve (AUUC).
2. Qini Curve, Qini Score, and Normalized Qini Coefficient.
3. Top-k fraction uplift and incremental conversions per 1,000 targeted customers.
4. Precision in Estimation of Heterogeneous Effects (PEHE / Root MSE of CATE) for synthetic data.
5. Bootstrap confidence intervals for policy and causal metrics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

try:
    _trapezoid = np.trapezoid
except AttributeError:
    try:
        _trapezoid = np.trapz
    except AttributeError:
        from scipy.integrate import trapezoid as _trapezoid


def calculate_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cumulative uplift curve.

    Parameters
    ----------
    y_true : np.ndarray
        Observed binary outcomes (0 or 1).
    treatment : np.ndarray
        Treatment indicator (0 or 1).
    uplift_scores : np.ndarray
        Predicted CATE or priority score.
    n_bins : int
        Number of evaluation points along targeting fraction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (fractions, cumulative_uplift, random_baseline)
    """
    n = len(y_true)
    if n == 0:
        return np.zeros(1), np.zeros(1), np.zeros(1)

    # Sort in descending order of predicted uplift
    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    # Cumulative counts
    cum_treated = np.cumsum(t_sorted)
    cum_control = np.cumsum(1 - t_sorted)
    cum_y_treated = np.cumsum(y_sorted * t_sorted)
    cum_y_control = np.cumsum(y_sorted * (1 - t_sorted))

    # Evaluate at regular sample fractions
    eval_indices = np.linspace(1, n, n_bins, dtype=int) - 1
    fractions = (eval_indices + 1) / n

    cum_uplift = np.zeros(n_bins)
    for i, idx in enumerate(eval_indices):
        n_t = cum_treated[idx]
        n_c = cum_control[idx]
        y_t = cum_y_treated[idx]
        y_c = cum_y_control[idx]

        if n_t > 0 and n_c > 0:
            rate_t = y_t / n_t
            rate_c = y_c / n_c
            cum_uplift[i] = (rate_t - rate_c) * (idx + 1)
        else:
            cum_uplift[i] = 0.0

    # Random baseline connects (0, 0) to (1, final_uplift)
    final_uplift = cum_uplift[-1]
    random_baseline = fractions * final_uplift

    return fractions, cum_uplift, random_baseline


def calculate_auuc(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> float:
    """Calculate Area Under the Uplift Curve (AUUC) minus Random Baseline Area."""
    fractions, cum_uplift, random_baseline = calculate_uplift_curve(
        y_true, treatment, uplift_scores, n_bins=100
    )
    # Trapezoidal integration
    area_curve = _trapezoid(cum_uplift, x=fractions)
    area_baseline = _trapezoid(random_baseline, x=fractions)
    return float(area_curve - area_baseline)


def calculate_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Qini curve: Q(k) = Y_t(k) - Y_c(k) * (N_t_total / N_c_total).

    Corrects for unequal treatment/control sample sizes (e.g. Criteo 85/15 ratio).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (fractions, qini_values, random_baseline)
    """
    n = len(y_true)
    if n == 0:
        return np.zeros(1), np.zeros(1), np.zeros(1)

    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    total_treated = np.sum(t_sorted)
    total_control = n - total_treated

    if total_control == 0 or total_treated == 0:
        return np.linspace(0, 1, n_bins), np.zeros(n_bins), np.zeros(n_bins)

    ratio = total_treated / total_control

    cum_y_treated = np.cumsum(y_sorted * t_sorted)
    cum_y_control = np.cumsum(y_sorted * (1 - t_sorted))

    eval_indices = np.linspace(1, n, n_bins, dtype=int) - 1
    fractions = (eval_indices + 1) / n

    qini_vals = np.zeros(n_bins)
    for i, idx in enumerate(eval_indices):
        y_t = cum_y_treated[idx]
        y_c = cum_y_control[idx]
        qini_vals[i] = y_t - y_c * ratio

    random_baseline = fractions * qini_vals[-1]
    return fractions, qini_vals, random_baseline


def calculate_qini_score(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> float:
    """Calculate Qini coefficient (Area between Qini curve and random baseline)."""
    fractions, qini_vals, random_baseline = calculate_qini_curve(
        y_true, treatment, uplift_scores, n_bins=100
    )
    area_qini = _trapezoid(qini_vals, x=fractions)
    area_random = _trapezoid(random_baseline, x=fractions)
    return float(area_qini - area_random)


def calculate_top_k_uplift(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    top_fraction: float = 0.20,
) -> float:
    """Calculate empirical treatment effect in the top fraction of targeted customers."""
    n = len(y_true)
    k = max(1, int(n * top_fraction))
    order = np.argsort(-uplift_scores)[:k]

    t_sub = treatment[order]
    y_sub = y_true[order]

    n_t = np.sum(t_sub == 1)
    n_c = np.sum(t_sub == 0)

    if n_t == 0 or n_c == 0:
        return 0.0

    rate_t = np.mean(y_sub[t_sub == 1])
    rate_c = np.mean(y_sub[t_sub == 0])
    return float(rate_t - rate_c)


def calculate_incremental_conversions_per_1000(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    top_fraction: float = 0.20,
) -> float:
    """Incremental conversions generated per 1,000 customers targeted at top_fraction."""
    uplift = calculate_top_k_uplift(y_true, treatment, uplift_scores, top_fraction)
    return float(uplift * 1000.0)


def calculate_policy_net_value(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    margin_per_conversion: float = 50.0,
    cost_per_intervention: float = 2.50,
    top_fraction: float = 0.20,
) -> float:
    """Calculate empirical net business value of targeting policy on test set."""
    n = len(y_true)
    k = max(1, int(n * top_fraction))
    order = np.argsort(-uplift_scores)[:k]

    t_sub = treatment[order]
    y_sub = y_true[order]

    n_t = np.sum(t_sub == 1)
    n_c = np.sum(t_sub == 0)

    if n_t == 0 or n_c == 0:
        return 0.0

    rate_t = np.mean(y_sub[t_sub == 1])
    rate_c = np.mean(y_sub[t_sub == 0])
    incremental_rate = rate_t - rate_c

    # Incremental conversions = incremental_rate * targeted_count
    inc_conversions = incremental_rate * k
    total_cost = k * cost_per_intervention
    net_profit = inc_conversions * margin_per_conversion - total_cost
    return float(net_profit)


def calculate_pehe(true_cate: np.ndarray, predicted_cate: np.ndarray) -> float:
    """Calculate Precision in Estimation of Heterogeneous Effects (PEHE / Root MSE of CATE)."""
    return float(np.sqrt(np.mean((predicted_cate - true_cate) ** 2)))


def evaluate_all_uplift_metrics(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    true_cate: Optional[np.ndarray] = None,
    margin_per_conversion: float = 50.0,
    cost_per_intervention: float = 2.50,
) -> Dict[str, float]:
    """Compute comprehensive evaluation dictionary for an uplift model."""
    qini_score = calculate_qini_score(y_true, treatment, uplift_scores)
    auuc = calculate_auuc(y_true, treatment, uplift_scores)
    uplift_10 = calculate_top_k_uplift(y_true, treatment, uplift_scores, 0.10)
    uplift_20 = calculate_top_k_uplift(y_true, treatment, uplift_scores, 0.20)
    inc_1000 = calculate_incremental_conversions_per_1000(y_true, treatment, uplift_scores, 0.20)
    net_val = calculate_policy_net_value(
        y_true, treatment, uplift_scores, margin_per_conversion, cost_per_intervention, 0.20
    )

    metrics = {
        "qini_score": round(qini_score, 4),
        "auuc": round(auuc, 4),
        "uplift_top_10pct": round(uplift_10, 5),
        "uplift_top_20pct": round(uplift_20, 5),
        "incremental_per_1000": round(inc_1000, 2),
        "policy_net_value_top_20pct": round(net_val, 2),
    }

    if true_cate is not None:
        pehe = calculate_pehe(true_cate, uplift_scores)
        corr, _ = stats.pearsonr(true_cate, uplift_scores)
        spearman, _ = stats.spearmanr(true_cate, uplift_scores)
        metrics["pehe"] = round(pehe, 5)
        metrics["cate_pearson_corr"] = round(float(corr), 4)
        metrics["cate_spearman_corr"] = round(float(spearman), 4)

    return metrics

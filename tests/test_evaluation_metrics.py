"""Unit tests for uplift evaluation metrics with hand-calculated verification fixtures."""

import numpy as np
import pytest
from src.evaluation.metrics import (
    calculate_auuc,
    calculate_pehe,
    calculate_qini_curve,
    calculate_qini_score,
    calculate_top_k_uplift,
)


def test_hand_calculated_pehe():
    true_cate = np.array([0.1, 0.2])
    pred_cate = np.array([0.1, 0.4])
    # Expected PEHE: sqrt(((0.1-0.1)^2 + (0.4-0.2)^2)/2) = sqrt(0.04/2) = sqrt(0.02) = 0.141421...
    expected_pehe = np.sqrt(0.02)
    computed_pehe = calculate_pehe(true_cate, pred_cate)
    assert np.isclose(computed_pehe, expected_pehe, atol=1e-5)


def test_qini_curve_shape_and_finite_values():
    n = 200
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.05, n)
    t = rng.binomial(1, 0.85, n)
    scores = rng.normal(0, 1, n)

    fractions, qini_vals, baseline = calculate_qini_curve(y, t, scores, n_bins=50)
    assert len(fractions) == 50
    assert len(qini_vals) == 50
    assert len(baseline) == 50
    assert not np.isnan(qini_vals).any()
    assert not np.isnan(baseline).any()
    assert fractions[0] > 0.0 and fractions[-1] == 1.0


def test_qini_score_positive_for_perfect_oracle():
    # Construct a case where top treated convert and top control do not
    # Treated: 50 ones, 50 zeros. Control: 50 ones, 50 zeros.
    n = 100
    t = np.array([1] * 50 + [0] * 50)
    # Perfect oracle assigns high scores to persuadables (treated responders and control non-responders)
    y = np.array([1] * 30 + [0] * 20 + [0] * 30 + [1] * 20)
    # Scores: treated who convert and control who don't get score 1.0; others get -1.0
    scores = np.array([1.0] * 30 + [-1.0] * 20 + [1.0] * 30 + [-1.0] * 20)

    qini = calculate_qini_score(y, t, scores)
    # Area above random baseline should be strictly positive
    assert qini > 0.0


def test_auuc_finite_and_computable():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    auuc = calculate_auuc(y, t, s)
    assert isinstance(auuc, float)
    assert not np.isnan(auuc)


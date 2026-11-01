import numpy as np
import pytest
from scipy import stats
from src.data.synthetic_generator import generate_synthetic_uplift_data
from src.models.uplift_learners import (
    DoublyRobustLearner,
    SLearner,
    TLearner,
    XLearner,
)


@pytest.fixture
def sample_causal_data():
    df = generate_synthetic_uplift_data(n_samples=1500, random_seed=42)
    feature_cols = [f"f{i}" for i in range(12)]
    X = df[feature_cols].values
    y = df["conversion"].values
    t = df["treatment"].values
    true_cate = df["true_cate"].values
    return X, y, t, true_cate


@pytest.mark.parametrize("model_cls", [SLearner, TLearner, XLearner, DoublyRobustLearner])
def test_uplift_models_fit_and_predict_shapes(model_cls, sample_causal_data):
    X, y, t, _ = sample_causal_data
    model = model_cls(random_state=42)
    model.fit(X, y, t)

    uplift = model.predict_uplift(X)
    assert len(uplift) == len(X)
    assert not np.isnan(uplift).any()
    assert not np.isinf(uplift).any()

    p0, p1 = model.predict_counterfactuals(X)
    assert len(p0) == len(X)
    assert len(p1) == len(X)
    assert not np.isnan(p0).any()
    assert not np.isnan(p1).any()


def test_synthetic_cate_recovery_positive_correlation(sample_causal_data):
    """Verify that an uplift learner correctly recovers heterogeneous effects with positive correlation to true CATE."""
    X, y, t, true_cate = sample_causal_data
    learner = XLearner(random_state=42)
    learner.fit(X, y, t)

    predicted_cate = learner.predict_uplift(X)
    corr, p_val = stats.pearsonr(predicted_cate, true_cate)

    # Model should correlate positively with ground truth CATE on synthetic DGP
    assert corr > 0.20, f"Expected Pearson correlation > 0.20, got {corr:.4f}"
    assert p_val < 0.01, f"Expected statistically significant correlation, got p={p_val}"


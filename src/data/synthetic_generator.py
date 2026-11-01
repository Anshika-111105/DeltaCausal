from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    z_clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def generate_synthetic_uplift_data(
    n_samples: int = 50000,
    treatment_ratio: float = 0.85,
    observational_bias: bool = False,
    random_seed: int = 42,
    base_cost: float = 2.50,
    base_margin: float = 50.0,
) -> pd.DataFrame:
    """Generate a synthetic uplift dataset with known counterfactual potential outcomes."""
    rng = np.random.default_rng(random_seed)

    # 1. Generate 12 pre-treatment features f0 to f11
    # Features represent customer behavioral metrics (e.g. recency, frequency, spend, engagement)
    mean = np.zeros(12)
    # Generate covariance matrix with moderate realistic correlation
    corr_base = np.eye(12)
    for i in range(12):
        for j in range(12):
            if i != j:
                corr_base[i, j] = 0.3 ** abs(i - j)
    X = rng.multivariate_normal(mean=mean, cov=corr_base, size=n_samples)

    # Standardize column names matching Criteo format
    feature_cols = [f"f{i}" for i in range(12)]
    df = pd.DataFrame(X, columns=feature_cols)
    df.insert(0, "customer_id", [f"cust_{i:07d}" for i in range(n_samples)])

    # 2. Baseline Outcome Probability Model: P(Y(0)=1 | X)
    # Most e-commerce baselines have conversion rates between 2% and 8%
    # Weights for baseline response
    w0 = np.array([0.45, -0.35, 0.25, -0.15, 0.50, -0.20, 0.10, -0.05, 0.30, 0.15, -0.10, 0.05])
    # Intercept tuned for realistic overall conversion rate
    base_logit = -2.80 + np.dot(X, w0) + 0.15 * (X[:, 0] ** 2) - 0.10 * (X[:, 4] * X[:, 1])
    prob_y0 = sigmoid(base_logit)

    # 3. Treatment Effect Heterogeneity Function: tau(X)
    # Define non-linear treatment effect:
    # High responsiveness for moderate baseline users with positive f0, f4
    # Reactance (sleeping dogs) for users with high f2 and negative f5 (oversaturated/sensitive)
    uplift_signal = (
        0.55 * X[:, 0]
        - 0.45 * X[:, 1]
        + 0.35 * X[:, 4]
        - 0.60 * np.maximum(0, X[:, 2] - 1.2)  # Negative effect for extreme f2 (irritated customers)
        + 0.25 * (X[:, 0] * X[:, 4])
    )
    # Shift logit under treatment
    # Intercept shift represents average uplift benefit
    treated_logit = base_logit + 0.35 + 0.50 * uplift_signal
    prob_y1 = sigmoid(treated_logit)

    # Clamp probabilities to valid [0.001, 0.999] interval
    prob_y0 = np.clip(prob_y0, 0.001, 0.999)
    prob_y1 = np.clip(prob_y1, 0.001, 0.999)

    # True CATE: difference in conversion probabilities
    true_cate = prob_y1 - prob_y0

    # 4. Treatment Assignment Mechanism: P(T=1 | X)
    if observational_bias:
        # Confounded assignment: targeted users who have higher engagement (f0, f4)
        propensity_logit = -0.5 + 0.6 * X[:, 0] + 0.5 * X[:, 4] - 0.3 * X[:, 1]
        propensity_score = sigmoid(propensity_logit)
        # Ensure positivity / common support [0.05, 0.95]
        propensity_score = np.clip(propensity_score, 0.05, 0.95)
    else:
        # Randomized trial: constant assignment probability
        propensity_score = np.full(n_samples, treatment_ratio)

    treatment = rng.binomial(n=1, p=propensity_score, size=n_samples)

    # 5. Potential Outcomes realization (Bernoulli sampling)
    potential_y0 = rng.binomial(n=1, p=prob_y0, size=n_samples)
    potential_y1 = rng.binomial(n=1, p=prob_y1, size=n_samples)

    # 6. Observed Outcome via SUTVA Consistency: Y = T*Y(1) + (1-T)*Y(0)
    conversion = np.where(treatment == 1, potential_y1, potential_y0)

    # Secondary outcome: website visit (higher rate than conversion, correlated with treatment)
    visit_prob = np.clip(
        0.15 + 0.40 * conversion + 0.10 * treatment + 0.08 * X[:, 0] - 0.05 * X[:, 1],
        0.01,
        0.98,
    )
    visit = rng.binomial(n=1, p=visit_prob, size=n_samples)

    # Exposure indicator: for treatment group, 96% exposure adherence; for control, 0%
    exposure_prob = np.where(treatment == 1, 0.96, 0.0)
    exposure = rng.binomial(n=1, p=exposure_prob, size=n_samples)

    # 7. Ground Truth Customer Segment Archetypes
    # Based on latent probabilities
    segments = []
    for p0, p1, tau_i in zip(prob_y0, prob_y1, true_cate):
        if tau_i > 0.04 and p0 < 0.25:
            segments.append("Persuadables")
        elif tau_i < -0.03:
            segments.append("Sleeping Dogs")
        elif p0 >= 0.20 and abs(tau_i) <= 0.04:
            segments.append("Sure Things")
        else:
            segments.append("Lost Causes")

    # 8. Business Financial Parameters
    # Intervention cost per customer (e.g. promo discount coupon + messaging cost)
    cost = np.clip(rng.normal(loc=base_cost, scale=0.20, size=n_samples), 1.0, 10.0)
    # Margin generated upon conversion
    margin = np.clip(rng.normal(loc=base_margin, scale=3.0, size=n_samples), 20.0, 100.0)

    # Populate final DataFrame
    df["treatment"] = treatment.astype(int)
    df["conversion"] = conversion.astype(int)
    df["visit"] = visit.astype(int)
    df["exposure"] = exposure.astype(int)
    df["propensity_score"] = np.round(propensity_score, 4)
    df["true_prob_y0"] = np.round(prob_y0, 4)
    df["true_prob_y1"] = np.round(prob_y1, 4)
    df["true_y0"] = potential_y0.astype(int)
    df["true_y1"] = potential_y1.astype(int)
    df["true_cate"] = np.round(true_cate, 5)
    df["segment_label"] = segments
    df["cost"] = np.round(cost, 2)
    df["margin"] = np.round(margin, 2)
    df["expected_incremental_net_value"] = np.round(true_cate * margin - cost, 2)

    return df


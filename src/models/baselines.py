from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class RandomTargetingBaseline:
    """Random targeting baseline: Assigns uniform random priority scores."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "RandomTargetingBaseline":
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        # Random score between 0 and 1
        return rng.uniform(0.0, 1.0, size=len(X))

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self.predict_uplift(X)
        return np.zeros(len(X)), p


class ZeroInterventionBaseline:
    """Zero intervention policy: Sends 0 offers."""

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "ZeroInterventionBaseline":
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(X)), np.zeros(len(X))


class AverageTreatmentEffectBaseline:
    """Constant ATE baseline: Assigns the average difference between treatment and control."""

    def __init__(self):
        self.ate_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "AverageTreatmentEffectBaseline":
        t_mask = treatment == 1
        c_mask = treatment == 0
        y_t = np.mean(y[t_mask]) if np.sum(t_mask) > 0 else 0.0
        y_c = np.mean(y[c_mask]) if np.sum(c_mask) > 0 else 0.0
        self.ate_ = float(y_t - y_c)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.ate_, dtype=float)

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(X)), np.full(len(X), self.ate_, dtype=float)


class PropensityBaseline:
    """Propensity model estimating treatment assignment probability e(X) = P(T=1 | X)."""

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(max_iter=500, random_state=random_state)
        self.is_fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray) -> "PropensityBaseline":
        self.model.fit(X, treatment)
        self.is_fitted = True
        return self

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        # Clip propensity away from 0 and 1 for numerical stability
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0.01, 0.99)


class ResponsePropensityBaseline:
    """Conventional customer response propensity model P(Y=1 | X).

    Crucial baseline: Demonstrates that customers with highest purchase probability
    are frequently 'Sure Things' who convert without an offer, wasting marketing spend.
    """

    def __init__(self, random_state: int = 42):
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: Optional[np.ndarray] = None) -> "ResponsePropensityBaseline":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """In traditional marketing, users are ranked directly by predicted conversion probability."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)[:, 1]

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        p = self.model.predict_proba(X)[:, 1]
        return p, p


class TwoModelNaiveBaseline:
    """Naive two-model approach using independent logistic regressors."""

    def __init__(self, random_state: int = 42):
        self.model_treated = LogisticRegression(max_iter=500, random_state=random_state)
        self.model_control = LogisticRegression(max_iter=500, random_state=random_state)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "TwoModelNaiveBaseline":
        t_mask = treatment == 1
        c_mask = treatment == 0

        self.model_treated.fit(X[t_mask], y[t_mask])
        self.model_control.fit(X[c_mask], y[c_mask])
        self.is_fitted = True
        return self

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        p1 = self.model_treated.predict_proba(X)[:, 1]
        p0 = self.model_control.predict_proba(X)[:, 1]
        return p0, p1

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        p0, p1 = self.predict_counterfactuals(X)
        return p1 - p0

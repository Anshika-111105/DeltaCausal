"""Uplift Meta-Learners for Counterfactual Treatment Effect Estimation.

Implements rigorous, production-grade causal meta-learners:
1. BaseUpliftLearner: Unified interface for all causal learners.
2. SLearner: Single-model estimator with treatment feature and counterfactual scoring.
3. TLearner: Separate outcome models for treated and control groups.
4. XLearner: Künzel et al. (2019) multi-stage crossover learner with propensity weighting.
5. DoublyRobustLearner: AIPW pseudo-outcome regression with double robustness.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor


class BaseUpliftLearner(ABC):
    """Abstract base class for all uplift learners."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "BaseUpliftLearner":
        """Fit causal model on training data."""
        pass

    @abstractmethod
    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect (CATE) tau(x) = E[Y(1) - Y(0) | X=x]."""
        pass

    @abstractmethod
    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate counterfactual outcome probabilities (P(Y(0)=1|X), P(Y(1)=1|X))."""
        pass


class SLearner(BaseUpliftLearner):
    """S-Learner (Single Model).

    Trains a single estimator mu(X, T) -> Y.
    CATE is estimated by: tau_hat(x) = mu_hat(x, 1) - mu_hat(x, 0).
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator or XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
        )
        self.model = clone(self.base_estimator)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "SLearner":
        # Augment feature matrix with treatment column: [X, T]
        X_aug = np.column_stack([X, treatment.astype(float)])
        self.model.fit(X_aug, y)
        self.is_fitted = True
        return self

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        # Counterfactual with T=0
        X_c = np.column_stack([X, np.zeros(len(X), dtype=float)])
        # Counterfactual with T=1
        X_t = np.column_stack([X, np.ones(len(X), dtype=float)])

        if hasattr(self.model, "predict_proba"):
            p0 = self.model.predict_proba(X_c)[:, 1]
            p1 = self.model.predict_proba(X_t)[:, 1]
        else:
            p0 = self.model.predict(X_c)
            p1 = self.model.predict(X_t)

        return p0, p1

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        p0, p1 = self.predict_counterfactuals(X)
        return p1 - p0


class TLearner(BaseUpliftLearner):
    """T-Learner (Two Models).

    Trains two independent estimators:
    mu_1(X) on treated group (T=1)
    mu_0(X) on control group (T=0)
    CATE is estimated by: tau_hat(x) = mu_hat_1(x) - mu_hat_0(x).
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        random_state: int = 42,
    ):
        base = base_estimator or XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
        )
        self.model_treated = clone(base)
        self.model_control = clone(base)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "TLearner":
        t_mask = treatment == 1
        c_mask = treatment == 0

        self.model_treated.fit(X[t_mask], y[t_mask])
        self.model_control.fit(X[c_mask], y[c_mask])
        self.is_fitted = True
        return self

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if hasattr(self.model_treated, "predict_proba"):
            p1 = self.model_treated.predict_proba(X)[:, 1]
            p0 = self.model_control.predict_proba(X)[:, 1]
        else:
            p1 = self.model_treated.predict(X)
            p0 = self.model_control.predict(X)

        return p0, p1

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        p0, p1 = self.predict_counterfactuals(X)
        return p1 - p0


class XLearner(BaseUpliftLearner):
    """X-Learner (Künzel et al. 2019).

    A multi-stage meta-learner specially tailored for unbalanced treatment/control ratios.
    Stage 1: Estimate outcome models mu_0 and mu_1.
    Stage 2: Impute counterfactual treatment effects:
             D_1 = Y_1 - mu_0(X_1)
             D_0 = mu_1(X_0) - Y_0
    Stage 3: Train effect regressors:
             tau_1(X) on (X_1, D_1)
             tau_0(X) on (X_0, D_0)
    Stage 4: Combine using propensity score e(X):
             tau_hat(X) = e(X)*tau_0(X) + (1 - e(X))*tau_1(X)
    """

    def __init__(
        self,
        base_classifier: Optional[Any] = None,
        base_regressor: Optional[Any] = None,
        random_state: int = 42,
    ):
        self.base_classifier = base_classifier or XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
        )
        self.base_regressor = base_regressor or XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_state,
        )
        self.propensity_model = LogisticRegression(max_iter=500, random_state=random_state)

        # Stage 1 models
        self.mu_1 = clone(self.base_classifier)
        self.mu_0 = clone(self.base_classifier)

        # Stage 3 models
        self.tau_1 = clone(self.base_regressor)
        self.tau_0 = clone(self.base_regressor)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "XLearner":
        t_mask = treatment == 1
        c_mask = treatment == 0

        X_1, y_1 = X[t_mask], y[t_mask]
        X_0, y_0 = X[c_mask], y[c_mask]

        # Stage 1: Fit outcome models
        self.mu_1.fit(X_1, y_1)
        self.mu_0.fit(X_0, y_0)

        # Fit propensity model
        self.propensity_model.fit(X, treatment)

        # Stage 2: Impute counterfactual treatment effects
        p0_for_treated = self.mu_0.predict_proba(X_1)[:, 1]
        p1_for_control = self.mu_1.predict_proba(X_0)[:, 1]

        D_1 = y_1 - p0_for_treated
        D_0 = p1_for_control - y_0

        # Stage 3: Train effect regressors
        self.tau_1.fit(X_1, D_1)
        self.tau_0.fit(X_0, D_0)

        self.is_fitted = True
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        # Estimate propensity score
        prop_score = self.propensity_model.predict_proba(X)[:, 1]
        prop_score = np.clip(prop_score, 0.05, 0.95)

        # Predict effects from both regressors
        pred_tau_1 = self.tau_1.predict(X)
        pred_tau_0 = self.tau_0.predict(X)

        # Stage 4: Weighted combination
        # When treated is dominant, prop_score is high, giving high weight to tau_0 (which used the accurate mu_1)
        return prop_score * pred_tau_0 + (1.0 - prop_score) * pred_tau_1

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        p1 = self.mu_1.predict_proba(X)[:, 1]
        p0 = self.mu_0.predict_proba(X)[:, 1]
        return p0, p1


class DoublyRobustLearner(BaseUpliftLearner):
    """Doubly Robust Learner (AIPW Pseudo-Outcome Regression).

    Combines propensity scoring and outcome modeling with double robustness:
    Gamma_i = mu_1(X_i) - mu_0(X_i) + T_i*(Y_i - mu_1(X_i))/e(X_i) - (1-T_i)*(Y_i - mu_0(X_i))/(1-e(X_i))
    tau_hat(X) is obtained by regressing Gamma on X.
    """

    def __init__(
        self,
        base_classifier: Optional[Any] = None,
        base_regressor: Optional[Any] = None,
        clip_propensity: Tuple[float, float] = (0.02, 0.98),
        random_state: int = 42,
    ):
        self.base_classifier = base_classifier or XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
        )
        self.base_regressor = base_regressor or XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_state,
        )
        self.clip_propensity = clip_propensity
        self.propensity_model = LogisticRegression(max_iter=500, random_state=random_state)
        self.mu_1 = clone(self.base_classifier)
        self.mu_0 = clone(self.base_classifier)
        self.effect_regressor = clone(self.base_regressor)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "DoublyRobustLearner":
        t_mask = treatment == 1
        c_mask = treatment == 0

        # 1. Fit outcome nuisance models
        self.mu_1.fit(X[t_mask], y[t_mask])
        self.mu_0.fit(X[c_mask], y[c_mask])

        # 2. Fit propensity model
        self.propensity_model.fit(X, treatment)
        e_hat = self.propensity_model.predict_proba(X)[:, 1]
        e_hat = np.clip(e_hat, self.clip_propensity[0], self.clip_propensity[1])

        # 3. Construct AIPW pseudo-outcome Gamma
        mu_1_hat = self.mu_1.predict_proba(X)[:, 1]
        mu_0_hat = self.mu_0.predict_proba(X)[:, 1]

        t = treatment.astype(float)
        y_flt = y.astype(float)

        gamma = (
            (mu_1_hat - mu_0_hat)
            + (t * (y_flt - mu_1_hat) / e_hat)
            - ((1.0 - t) * (y_flt - mu_0_hat) / (1.0 - e_hat))
        )

        # 4. Regress pseudo-outcome on X
        self.effect_regressor.fit(X, gamma)
        self.is_fitted = True
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.effect_regressor.predict(X)

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        p1 = self.mu_1.predict_proba(X)[:, 1]
        p0 = self.mu_0.predict_proba(X)[:, 1]
        return p0, p1


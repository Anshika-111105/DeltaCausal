"""Causal Feature Preprocessor and Leakage-Free Transformation Pipeline.

Ensures that preprocessing transformations (imputations, standardizations, encodings)
are strictly fit on training data only, with no information leakage to validation/test sets.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CausalPreprocessor:
    """Production preprocessor for pre-treatment covariates."""

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        imputation_strategy: str = "median",
        with_scaling: bool = True,
    ):
        self.feature_cols = feature_cols or [f"f{i}" for i in range(12)]
        self.imputation_strategy = imputation_strategy
        self.with_scaling = with_scaling
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False
        self.feature_names_out_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "CausalPreprocessor":
        """Fit preprocessor strictly on training data."""
        if isinstance(X, pd.DataFrame):
            # Select only requested feature columns
            available_cols = [c for c in self.feature_cols if c in X.columns]
            if not available_cols:
                raise ValueError(f"None of specified feature columns {self.feature_cols} found in input DataFrame.")
            self.feature_cols = available_cols
            X_mat = X[self.feature_cols].values
        else:
            X_mat = X

        steps = [
            ("imputer", SimpleImputer(strategy=self.imputation_strategy)),
        ]
        if self.with_scaling:
            steps.append(("scaler", StandardScaler()))

        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X_mat)
        self.is_fitted = True
        self.feature_names_out_ = list(self.feature_cols)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform features using fitted pipeline."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Preprocessor must be fit before calling transform.")

        if isinstance(X, pd.DataFrame):
            # Ensure columns match expected features, fill missing columns if any
            for c in self.feature_cols:
                if c not in X.columns:
                    X = X.copy()
                    X[c] = np.nan
            X_mat = X[self.feature_cols].values
        else:
            X_mat = X

        return self.pipeline.transform(X_mat)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit on X and transform X."""
        return self.fit(X).transform(X)

    def get_feature_names(self) -> List[str]:
        """Return feature column names."""
        return list(self.feature_names_out_)

    def save(self, path: Union[str, Path]) -> None:
        """Save fitted preprocessor to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CausalPreprocessor":
        """Load fitted preprocessor from disk."""
        return joblib.load(path)


from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import yaml

from ..features.preprocessor import CausalPreprocessor
from .baselines import (
    AverageTreatmentEffectBaseline,
    PropensityBaseline,
    RandomTargetingBaseline,
    ResponsePropensityBaseline,
    TwoModelNaiveBaseline,
    ZeroInterventionBaseline,
)
from .uplift_learners import (
    BaseUpliftLearner,
    DoublyRobustLearner,
    SLearner,
    TLearner,
    XLearner,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingExperimentRecord:
    experiment_id: str
    created_at_utc: str
    random_seed: int
    models_trained: List[str]
    n_train_samples: int
    n_features: int
    feature_names: List[str]
    treatment_column: str
    outcome_column: str
    model_artifacts_dir: str


class ModelTrainer:
    """Orchestrates end-to-end model training and artifact persistence."""

    def __init__(
        self,
        artifacts_root: str = "artifacts",
        random_seed: int = 42,
    ):
        self.artifacts_root = Path(artifacts_root)
        self.random_seed = random_seed

    def train_all(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
        experiment_id: Optional[str] = None,
        model_types: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], CausalPreprocessor, TrainingExperimentRecord]:
        """Fit preprocessor, baselines, and uplift learners on training set.

        Parameters
        ----------
        train_df : pd.DataFrame
            Pre-split training partition.
        feature_cols : List[str]
            Covariate names.
        treatment_col : str
            Treatment indicator.
        outcome_col : str
            Outcome indicator.
        experiment_id : Optional[str]
            Unique experiment identifier.
        model_types : Optional[List[str]]
            Subset of models to train. Defaults to all models.

        Returns
        -------
        Tuple[Dict[str, Any], CausalPreprocessor, TrainingExperimentRecord]
            Trained models dictionary, preprocessor, and experiment metadata.
        """
        if experiment_id is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"exp_{ts}"

        exp_dir = self.artifacts_root / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training run: {experiment_id} (N={len(train_df):,})")

        # 1. Fit Preprocessor strictly on training features
        preprocessor = CausalPreprocessor(
            feature_cols=feature_cols,
            imputation_strategy="median",
            with_scaling=True,
        )
        X_train = preprocessor.fit_transform(train_df)
        y_train = train_df[outcome_col].values
        t_train = train_df[treatment_col].values

        # Save preprocessor
        preprocessor.save(exp_dir / "preprocessor.joblib")

        # 2. Instantiate all candidate models
        available_models = {
            "random": RandomTargetingBaseline(random_state=self.random_seed),
            "zero_intervention": ZeroInterventionBaseline(),
            "overall_ate": AverageTreatmentEffectBaseline(),
            "response_propensity": ResponsePropensityBaseline(random_state=self.random_seed),
            "two_model_naive": TwoModelNaiveBaseline(random_state=self.random_seed),
            "s_learner": SLearner(random_state=self.random_seed),
            "t_learner": TLearner(random_state=self.random_seed),
            "x_learner": XLearner(random_state=self.random_seed),
            "doubly_robust": DoublyRobustLearner(random_state=self.random_seed),
        }

        selected_names = model_types if model_types else list(available_models.keys())
        fitted_models: Dict[str, Any] = {}

        for name in selected_names:
            if name not in available_models:
                logger.warning(f"Unknown model '{name}' requested. Skipping.")
                continue

            logger.info(f"Fitting model: {name}...")
            model = available_models[name]
            if name == "response_propensity":
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, t_train)

            fitted_models[name] = model

            # Persist model
            joblib.dump(model, exp_dir / f"{name}.joblib")

        # Also fit & save propensity model for causal adjustment
        prop_model = PropensityBaseline(random_state=self.random_seed)
        prop_model.fit(X_train, t_train)
        fitted_models["propensity"] = prop_model
        joblib.dump(prop_model, exp_dir / "propensity.joblib")

        # Save experiment record
        record = TrainingExperimentRecord(
            experiment_id=experiment_id,
            created_at_utc=datetime.utcnow().isoformat(),
            random_seed=self.random_seed,
            models_trained=list(fitted_models.keys()),
            n_train_samples=len(train_df),
            n_features=X_train.shape[1],
            feature_names=preprocessor.get_feature_names(),
            treatment_column=treatment_col,
            outcome_column=outcome_col,
            model_artifacts_dir=str(exp_dir),
        )

        with open(exp_dir / "experiment_record.json", "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)

        logger.info(f"Training run {experiment_id} successfully finalized and persisted to {exp_dir}.")
        return fitted_models, preprocessor, record


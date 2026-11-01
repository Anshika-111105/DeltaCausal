"""Causal Data Splitter with Stratified Train/Val/Test Separation.

Preserves the joint distribution of treatment assignment and outcomes across splits,
preventing partition imbalances and data leakage.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    """Container holding train, validation, and test datasets with metadata."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    split_ratios: Dict[str, float]
    stratify_col: Optional[str] = None
    random_seed: int = 42

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return diagnostic summary of sample size and outcome rates by split."""
        summary_dict = {}
        for name, d in [("train", self.train_df), ("val", self.val_df), ("test", self.test_df)]:
            n_tot = len(d)
            t_rate = float((d["treatment"] == 1).mean()) if "treatment" in d.columns else 0.0
            y_col = "conversion" if "conversion" in d.columns else ("visit" if "visit" in d.columns else None)
            y_rate = float(d[y_col].mean()) if y_col else 0.0
            summary_dict[name] = {
                "n_samples": n_tot,
                "treatment_ratio": round(t_rate, 4),
                "outcome_rate": round(y_rate, 4),
            }
        return summary_dict


class CausalDataSplitter:
    """Split causal datasets cleanly into train, validation, and test partitions."""

    def __init__(
        self,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        random_state: int = 42,
    ):
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0. Got {train_ratio + val_ratio + test_ratio}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
        stratify: bool = True,
    ) -> SplitData:
        """Perform stratified splitting preserving (treatment, outcome) joint strata.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset.
        treatment_col : str
            Column name indicating treatment (0 or 1).
        outcome_col : str
            Column name indicating outcome (0 or 1).
        stratify : bool
            Whether to stratify on composite (treatment, outcome).

        Returns
        -------
        SplitData
            Object containing train_df, val_df, and test_df.
        """
        if len(df) == 0:
            raise ValueError("Input dataframe is empty.")

        if stratify and treatment_col in df.columns and outcome_col in df.columns:
            # Composite stratum: 2 * treatment + outcome -> 4 distinct strata: (0,0), (0,1), (1,0), (1,1)
            strata = (df[treatment_col].astype(int) * 2 + df[outcome_col].astype(int)).values
        else:
            strata = None

        # First split off test set: test_size = test_ratio
        test_size = self.test_ratio
        train_val_df, test_df, strata_tv, _ = train_test_split(
            df,
            strata if strata is not None else np.zeros(len(df)),
            test_size=test_size,
            random_state=self.random_state,
            stratify=strata,
        )

        # Second split: separate train and validation from train_val_df
        # Val proportion relative to train+val = val_ratio / (train_ratio + val_ratio)
        val_prop_relative = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_prop_relative,
            random_state=self.random_state,
            stratify=strata_tv if strata is not None else None,
        )

        return SplitData(
            train_df=train_df.reset_index(drop=True),
            val_df=val_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True),
            split_ratios={
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            stratify_col=f"{treatment_col}_{outcome_col}" if stratify else None,
            random_seed=self.random_state,
        )


"""Data Loader for Real Criteo and Synthetic Uplift Datasets.

Provides:
- Chunked ingestion for large-scale files (e.g. Criteo v2.1 ~14M rows).
- Controlled random sampling for standard laptop workflows.
- Automatic synthetic fallback generation with ground truth CATE.
- Strict provenance labeling (Real Criteo vs. Synthetic) to prevent silent leakage.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .synthetic_generator import generate_synthetic_uplift_data

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata container for loaded causal datasets."""
    dataset_name: str
    is_synthetic: bool
    total_records: int
    n_features: int
    feature_columns: List[str]
    treatment_column: str
    outcome_column: str
    treatment_count: int
    control_count: int
    overall_conversion_rate: float
    treated_conversion_rate: float
    control_conversion_rate: float
    naive_ate: float
    file_path: Optional[str] = None


class UpliftDataLoader:
    """Robust data loader supporting both Criteo and Synthetic data."""

    def __init__(
        self,
        raw_path: Optional[str] = None,
        synthetic_path: Optional[str] = None,
        chunk_size: int = 100000,
        random_seed: int = 42,
    ):
        self.raw_path = Path(raw_path) if raw_path else None
        self.synthetic_path = Path(synthetic_path) if synthetic_path else Path("data/synthetic/synthetic_uplift.csv")
        self.chunk_size = chunk_size
        self.random_seed = random_seed

    def load_data(
        self,
        dataset_type: str = "synthetic",
        sample_size: Optional[int] = 50000,
        observational_bias: bool = False,
        force_regenerate: bool = False,
    ) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Load either Criteo real dataset or synthetic benchmark dataset.

        Parameters
        ----------
        dataset_type : str
            Either 'criteo' or 'synthetic'.
        sample_size : Optional[int]
            Subsample size for memory-bounded execution. If None, loads entire file.
        observational_bias : bool
            Used when generating synthetic data to simulate confounding.
        force_regenerate : bool
            If True, recreates synthetic benchmark file.

        Returns
        -------
        Tuple[pd.DataFrame, DatasetMetadata]
            Loaded DataFrame and associated metadata.
        """
        if dataset_type == "criteo":
            if self.raw_path and self.raw_path.exists():
                logger.info(f"Loading Criteo dataset from {self.raw_path}")
                df = self._load_criteo(self.raw_path, sample_size)
                is_synthetic = False
                dataset_name = "Criteo Uplift v2.1"
            else:
                logger.warning(
                    f"Criteo raw file not found at {self.raw_path}. "
                    "Falling back to clearly labeled synthetic benchmark data."
                )
                df = self._get_or_create_synthetic(
                    sample_size=sample_size or 50000,
                    observational_bias=observational_bias,
                    force_regenerate=force_regenerate,
                )
                is_synthetic = True
                dataset_name = "Synthetic Fallback (Criteo File Not Found)"
        else:
            logger.info("Loading Synthetic Uplift Benchmark Dataset...")
            df = self._get_or_create_synthetic(
                sample_size=sample_size or 50000,
                observational_bias=observational_bias,
                force_regenerate=force_regenerate,
            )
            is_synthetic = True
            dataset_name = "Synthetic E-Commerce Uplift (Known CATE)"

        metadata = self._compute_metadata(df, dataset_name=dataset_name, is_synthetic=is_synthetic)
        return df, metadata

    def _load_criteo(self, path: Path, sample_size: Optional[int]) -> pd.DataFrame:
        """Load Criteo CSV or CSV.GZ with reservoir/chunked sampling if sample_size is given."""
        if sample_size is None:
            return pd.read_csv(path)

        # Efficient reservoir/chunked sampling for large files
        chunks = []
        rows_so_far = 0
        rng = np.random.default_rng(self.random_seed)

        for chunk in pd.read_csv(path, chunksize=self.chunk_size):
            chunks.append(chunk)
            rows_so_far += len(chunk)
            if rows_so_far >= sample_size * 3:
                # Read sufficient candidate rows then sample
                break

        full_sample_df = pd.concat(chunks, ignore_index=True)
        if len(full_sample_df) > sample_size:
            sampled_indices = rng.choice(len(full_sample_df), size=sample_size, replace=False)
            return full_sample_df.iloc[sampled_indices].reset_index(drop=True)
        return full_sample_df

    def _get_or_create_synthetic(
        self,
        sample_size: int,
        observational_bias: bool,
        force_regenerate: bool,
    ) -> pd.DataFrame:
        """Load cached synthetic dataset or generate and persist."""
        self.synthetic_path.parent.mkdir(parents=True, exist_ok=True)

        if self.synthetic_path.exists() and not force_regenerate:
            df = pd.read_csv(self.synthetic_path)
            if len(df) >= sample_size:
                return df.iloc[:sample_size].copy()

        # Generate new synthetic dataset
        logger.info(f"Generating new synthetic dataset with N={sample_size}...")
        df = generate_synthetic_uplift_data(
            n_samples=sample_size,
            treatment_ratio=0.85,
            observational_bias=observational_bias,
            random_seed=self.random_seed,
        )
        df.to_csv(self.synthetic_path, index=False)
        return df

    def _compute_metadata(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        is_synthetic: bool,
    ) -> DatasetMetadata:
        """Compute basic dataset summary statistics."""
        feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
        t_col = "treatment" if "treatment" in df.columns else None
        y_col = "conversion" if "conversion" in df.columns else None

        if not t_col or not y_col:
            raise ValueError(f"Dataset missing required treatment ({t_col}) or outcome ({y_col}) columns.")

        treated_mask = df[t_col] == 1
        control_mask = df[t_col] == 0

        t_count = int(treated_mask.sum())
        c_count = int(control_mask.sum())

        p_overall = float(df[y_col].mean())
        p_t = float(df.loc[treated_mask, y_col].mean()) if t_count > 0 else 0.0
        p_c = float(df.loc[control_mask, y_col].mean()) if c_count > 0 else 0.0
        naive_ate = p_t - p_c

        return DatasetMetadata(
            dataset_name=dataset_name,
            is_synthetic=is_synthetic,
            total_records=len(df),
            n_features=len(feature_cols),
            feature_columns=feature_cols,
            treatment_column=t_col,
            outcome_column=y_col,
            treatment_count=t_count,
            control_count=c_count,
            overall_conversion_rate=round(p_overall, 5),
            treated_conversion_rate=round(p_t, 5),
            control_conversion_rate=round(p_c, 5),
            naive_ate=round(naive_ate, 6),
            file_path=str(self.synthetic_path if is_synthetic else self.raw_path),
        )


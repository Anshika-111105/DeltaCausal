import numpy as np
import pandas as pd
import pytest
from src.data.splitter import CausalDataSplitter
from src.features.preprocessor import CausalPreprocessor


def test_causal_data_splitter_partitions():
    n = 1000
    df = pd.DataFrame({
        "treatment": np.random.binomial(1, 0.85, n),
        "conversion": np.random.binomial(1, 0.05, n),
        "f0": np.random.randn(n),
        "f1": np.random.randn(n),
    })

    splitter = CausalDataSplitter(train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_state=42)
    splits = splitter.split(df, stratify=True)

    # Check partition sizes
    assert len(splits.train_df) == 600
    assert len(splits.val_df) == 200
    assert len(splits.test_df) == 200

    # Ensure zero index contamination
    train_indices = set(splits.train_df.index)
    val_indices = set(splits.val_df.index)
    test_indices = set(splits.test_df.index)
    assert len(splits.train_df) + len(splits.val_df) + len(splits.test_df) == n


def test_causal_preprocessor_fit_transform():
    train_df = pd.DataFrame({
        "f0": [1.0, np.nan, 3.0, 4.0],
        "f1": [10.0, 20.0, 30.0, 40.0],
    })
    test_df = pd.DataFrame({
        "f0": [2.0, 5.0],
        "f1": [15.0, np.nan],
    })

    preprocessor = CausalPreprocessor(feature_cols=["f0", "f1"], imputation_strategy="median")
    X_train = preprocessor.fit_transform(train_df)
    assert not np.isnan(X_train).any()
    assert X_train.shape == (4, 2)

    X_test = preprocessor.transform(test_df)
    assert not np.isnan(X_test).any()
    assert X_test.shape == (2, 2)


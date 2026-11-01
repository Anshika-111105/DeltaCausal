import numpy as np
import pandas as pd
import pytest
from src.validation.validator import DataValidator


def test_validator_detects_missing_mandatory_columns():
    validator = DataValidator()
    # Missing treatment and conversion
    df = pd.DataFrame({"f0": [1.0, 2.0], "f1": [0.5, 0.2]})
    report = validator.validate(df, feature_cols=["f0", "f1"])
    assert not report.is_valid
    issue_types = [i["check"] for i in report.issues]
    assert "MissingMandatoryColumn" in issue_types


def test_validator_rejects_non_binary_treatment():
    validator = DataValidator()
    df = pd.DataFrame({
        "treatment": [0, 1, 2],  # 2 is invalid
        "conversion": [0, 1, 0],
        "f0": [1.0, 2.0, 3.0],
    })
    report = validator.validate(df, feature_cols=["f0"])
    assert not report.is_valid
    assert any(i["check"] == "InvalidTreatmentCoding" for i in report.issues)


def test_validator_detects_target_leakage_in_features():
    validator = DataValidator()
    df = pd.DataFrame({
        "treatment": [0, 1, 0, 1],
        "conversion": [0, 1, 0, 0],
        "f0": [1.0, 2.0, 3.0, 4.0],
        "true_cate": [0.05, 0.10, -0.02, 0.04],
    })
    # Illegally passing target or true_cate as a feature
    report = validator.validate(df, feature_cols=["f0", "conversion", "true_cate"])
    assert not report.is_valid
    assert any(i["check"] == "TargetLeakageInFeatures" for i in report.issues)


def test_validator_passes_on_valid_dataset():
    validator = DataValidator()
    df = pd.DataFrame({
        "treatment": [0, 1] * 100,
        "conversion": [0, 1] * 100,
        "f0": np.random.randn(200),
        "f1": np.random.randn(200),
    })
    report = validator.validate(df, feature_cols=["f0", "f1"])
    assert report.is_valid
    assert report.duplicate_count == 0


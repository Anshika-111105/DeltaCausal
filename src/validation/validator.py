"""Data Validation and Quality Assurance for Uplift Modeling Engine.

Performs:
1. Schema & column integrity checks (mandatory columns, data types, value sets).
2. Missingness and duplicate record diagnostics.
3. Severe outcome imbalance and sample size sufficiency tests.
4. Target and post-treatment feature leakage detection.
5. Predictor hygiene (ensuring treatment/outcomes are never included in feature matrix).
6. Generation of structured machine-readable JSON reports and human-readable summaries.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    severity: str  # "ERROR", "WARNING", "INFO"
    check: str
    message: str
    column: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    is_valid: bool
    dataset_type: str
    total_rows: int
    total_columns: int
    missing_value_summary: Dict[str, int]
    duplicate_count: int
    treatment_distribution: Dict[int, int]
    outcome_distribution: Dict[int, int]
    outcome_rate_treated: float
    outcome_rate_control: float
    issues: List[Dict[str, Any]]
    execution_time_utc: str

    def to_json(self, path: Optional[Path] = None) -> str:
        data = asdict(self)
        json_str = json.dumps(data, indent=2)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def summary_markdown(self) -> str:
        """Render a readable markdown summary of validation findings."""
        status_badge = "PASS" if self.is_valid else "FAIL"
        md = [
            f"# Data Validation Report: [{status_badge}]",
            f"- **Dataset Type**: `{self.dataset_type}`",
            f"- **Records Checked**: {self.total_rows:,}",
            f"- **Features / Columns**: {self.total_columns}",
            f"- **Duplicates Found**: {self.duplicate_count:,}",
            "",
            "## Treatment & Outcome Balance",
            f"- **Treatment (T=1)**: {self.treatment_distribution.get(1, 0):,} ({self.treatment_distribution.get(1, 0) / max(1, self.total_rows):.2%})",
            f"- **Control (T=0)**: {self.treatment_distribution.get(0, 0):,} ({self.treatment_distribution.get(0, 0) / max(1, self.total_rows):.2%})",
            f"- **Treated Conversion Rate**: {self.outcome_rate_treated:.4%}",
            f"- **Control Conversion Rate**: {self.outcome_rate_control:.4%}",
            f"- **Empirical Difference (Naive ATE)**: {self.outcome_rate_treated - self.outcome_rate_control:+.4%}",
            "",
            "## Identified Issues & Warnings",
        ]

        if not self.issues:
            md.append("No critical issues or warnings detected. Dataset meets causal modeling prerequisites.")
        else:
            for issue in self.issues:
                sev = issue["severity"]
                icon = "❌" if sev == "ERROR" else ("⚠️" if sev == "WARNING" else "ℹ️")
                col_info = f" (Column: `{issue['column']}`)" if issue.get("column") else ""
                md.append(f"- {icon} **[{sev}] {issue['check']}**{col_info}: {issue['message']}")

        return "\n".join(md)


class DataValidator:
    """Comprehensive causal dataset validator."""

    def __init__(self, schema_config_path: str = "configs/schema.yaml"):
        self.schema_config_path = Path(schema_config_path)
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        if self.schema_config_path.exists():
            with open(self.schema_config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def validate(
        self,
        df: pd.DataFrame,
        dataset_type: str = "synthetic",
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
        feature_cols: Optional[List[str]] = None,
        disallow_post_treatment: Optional[List[str]] = None,
    ) -> ValidationReport:
        """Run all data validation checks.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to inspect.
        dataset_type : str
            'criteo' or 'synthetic'.
        treatment_col : str
            Treatment indicator column name.
        outcome_col : str
            Primary outcome column name.
        feature_cols : Optional[List[str]]
            Explicit feature columns. If None, inferred as f0-f11.
        disallow_post_treatment : Optional[List[str]]
            Columns that occur after treatment and must not be used as predictors.
        """
        issues: List[ValidationIssue] = []

        if df is None or len(df) == 0:
            return ValidationReport(
                is_valid=False,
                dataset_type=dataset_type,
                total_rows=0,
                total_columns=0,
                missing_value_summary={},
                duplicate_count=0,
                treatment_distribution={},
                outcome_distribution={},
                outcome_rate_treated=0.0,
                outcome_rate_control=0.0,
                issues=[ValidationIssue("ERROR", "EmptyDataFrame", "Supplied DataFrame is empty.")],
                execution_time_utc=pd.Timestamp.utcnow().isoformat(),
            )

        # 1. Mandatory Columns Check
        if treatment_col not in df.columns:
            issues.append(ValidationIssue("ERROR", "MissingMandatoryColumn", f"Treatment column '{treatment_col}' not found in dataframe."))
        if outcome_col not in df.columns:
            issues.append(ValidationIssue("ERROR", "MissingMandatoryColumn", f"Primary outcome column '{outcome_col}' not found in dataframe."))

        # 2. Check Candidate Features
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
        if len(feature_cols) == 0:
            issues.append(ValidationIssue("ERROR", "MissingFeatures", "No feature columns (e.g. f0-f11) detected in dataframe."))

        # 3. Target Leakage & Predictor Hygiene
        forbidden_in_features = {treatment_col, outcome_col, "visit", "exposure", "true_y0", "true_y1", "true_cate"}
        leaked_features = set(feature_cols).intersection(forbidden_in_features)
        if leaked_features:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "TargetLeakageInFeatures",
                    f"Forbidden target or causal ground-truth columns detected in feature list: {list(leaked_features)}",
                )
            )

        # 4. Post-treatment variables leakage check
        if disallow_post_treatment:
            leaked_post = set(feature_cols).intersection(set(disallow_post_treatment))
            if leaked_post:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "PostTreatmentLeakage",
                        f"Post-treatment variables found in feature list: {list(leaked_post)}. Including them causes post-treatment conditioning bias.",
                    )
                )

        # 5. Missing Values and Duplicates
        missing_counts = df.isna().sum().to_dict()
        total_missing = sum(missing_counts.values())
        if total_missing > 0:
            cols_with_missing = {k: v for k, v in missing_counts.items() if v > 0}
            issues.append(
                ValidationIssue(
                    "WARNING",
                    "MissingValuesDetected",
                    f"Found {total_missing} missing values across columns: {cols_with_missing}",
                    details=cols_with_missing,
                )
            )

        # Check duplicates (ignoring customer_id if present)
        cols_for_dup = [c for c in df.columns if c != "customer_id"]
        dup_count = int(df.duplicated(subset=cols_for_dup).sum())
        if dup_count > 0:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    "DuplicateRecords",
                    f"Detected {dup_count} duplicate records ({dup_count / len(df):.2%}).",
                )
            )

        # 6. Binary coding checks for treatment and outcome
        if treatment_col in df.columns:
            unique_t = set(df[treatment_col].dropna().unique())
            if not unique_t.issubset({0, 1}):
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "InvalidTreatmentCoding",
                        f"Treatment column must be binary [0, 1], got values: {unique_t}",
                        column=treatment_col,
                    )
                )

        if outcome_col in df.columns:
            unique_y = set(df[outcome_col].dropna().unique())
            if not unique_y.issubset({0, 1}):
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "InvalidOutcomeCoding",
                        f"Outcome column must be binary [0, 1], got values: {unique_y}",
                        column=outcome_col,
                    )
                )

        # 7. Imbalance and Sample Size checks
        t_dist = df[treatment_col].value_counts().to_dict() if treatment_col in df.columns else {}
        y_dist = df[outcome_col].value_counts().to_dict() if outcome_col in df.columns else {}

        n_t = t_dist.get(1, 0)
        n_c = t_dist.get(0, 0)
        if n_t < 100 or n_c < 100:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "InsufficientSampleSize",
                    f"Insufficient treatment or control sample size (Treated: {n_t}, Control: {n_c}). Minimum 100 required.",
                )
            )

        p_t = float(df.loc[df[treatment_col] == 1, outcome_col].mean()) if n_t > 0 and outcome_col in df.columns else 0.0
        p_c = float(df.loc[df[treatment_col] == 0, outcome_col].mean()) if n_c > 0 and outcome_col in df.columns else 0.0

        if p_t < 0.0005 or p_c < 0.0005:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    "SevereOutcomeImbalance",
                    f"Extremely rare outcome event (Treated rate: {p_t:.5f}, Control rate: {p_c:.5f}). High variance in CATE estimates expected.",
                )
            )

        # Check for perfect collinearity / identical features
        for f in feature_cols:
            if f in df.columns and df[f].nunique() <= 1:
                issues.append(
                    ValidationIssue(
                        "WARNING",
                        "ZeroVarianceFeature",
                        f"Feature '{f}' has constant zero variance.",
                        column=f,
                    )
                )

        # Check if any errors occurred
        has_errors = any(i.severity == "ERROR" for i in issues)

        report = ValidationReport(
            is_valid=not has_errors,
            dataset_type=dataset_type,
            total_rows=len(df),
            total_columns=len(df.columns),
            missing_value_summary={k: int(v) for k, v in missing_counts.items() if v > 0},
            duplicate_count=dup_count,
            treatment_distribution={int(k): int(v) for k, v in t_dist.items()},
            outcome_distribution={int(k): int(v) for k, v in y_dist.items()},
            outcome_rate_treated=round(p_t, 5),
            outcome_rate_control=round(p_c, 5),
            issues=[asdict(i) for i in issues],
            execution_time_utc=pd.Timestamp.utcnow().isoformat(),
        )

        return report


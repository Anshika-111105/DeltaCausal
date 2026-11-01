"""Causal Inference Diagnostics and Experimental Design Validation.

Evaluates foundational causal identification assumptions:
1. Covariate Balance: Standardized Mean Differences (SMD) and KS-tests between treatment and control.
2. Sample Ratio Mismatch (SRM): Chi-square goodness-of-fit test against planned allocation ratio.
3. Positivity & Common Support: Propensity score overlap diagnostics.
4. Unconfoundedness & Observational Sensitivity indicators.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class CovariateBalanceMetric:
    feature: str
    mean_treated: float
    mean_control: float
    std_treated: float
    std_control: float
    smd: float  # Standardized Mean Difference
    ks_statistic: float
    ks_p_value: float
    is_imbalanced: bool  # |SMD| > threshold (e.g. 0.10)


@dataclass
class SRMTestResult:
    observed_treated: int
    observed_control: int
    expected_ratio: float
    chi2_statistic: float
    p_value: float
    is_srm_detected: bool  # p_value < alpha (e.g. 0.01)
    interpretation: str


@dataclass
class PropensityOverlapResult:
    treated_min: float
    treated_max: float
    control_min: float
    control_max: float
    common_support_min: float
    common_support_max: float
    violations_count: int
    violations_pct: float
    is_overlap_adequate: bool
    summary: str


class CausalDiagnostics:
    """Causal diagnostics engine for experimental and observational datasets."""

    def __init__(self, smd_threshold: float = 0.10, srm_alpha: float = 0.01):
        self.smd_threshold = smd_threshold
        self.srm_alpha = srm_alpha

    def check_covariate_balance(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        treatment_col: str = "treatment",
    ) -> List[CovariateBalanceMetric]:
        """Compute Standardized Mean Differences (SMD) and KS tests for each covariate.

        SMD formula:
            SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)
        SMD < 0.10 indicates well-balanced covariates consistent with randomization.
        """
        results = []
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        for col in feature_cols:
            if col not in df.columns:
                continue

            x_t = treated[col].dropna().values
            x_c = control[col].dropna().values

            if len(x_t) == 0 or len(x_c) == 0:
                continue

            mean_t, std_t = float(np.mean(x_t)), float(np.std(x_t, ddof=1)) if len(x_t) > 1 else 0.0
            mean_c, std_c = float(np.mean(x_c)), float(np.std(x_c, ddof=1)) if len(x_c) > 1 else 0.0

            pooled_sd = np.sqrt((std_t ** 2 + std_c ** 2) / 2.0)
            if pooled_sd > 1e-8:
                smd = float((mean_t - mean_c) / pooled_sd)
            else:
                smd = 0.0

            # Kolmogorov-Smirnov test for distribution equivalence
            ks_res = stats.ks_2samp(x_t, x_c)

            results.append(
                CovariateBalanceMetric(
                    feature=col,
                    mean_treated=round(mean_t, 4),
                    mean_control=round(mean_c, 4),
                    std_treated=round(std_t, 4),
                    std_control=round(std_c, 4),
                    smd=round(smd, 4),
                    ks_statistic=round(float(ks_res.statistic), 4),
                    ks_p_value=round(float(ks_res.pvalue), 6),
                    is_imbalanced=bool(abs(smd) > self.smd_threshold),
                )
            )

        return results

    def check_sample_ratio_mismatch(
        self,
        df: pd.DataFrame,
        expected_treatment_ratio: float = 0.85,
        treatment_col: str = "treatment",
    ) -> SRMTestResult:
        """Run Pearson Chi-Square test for Sample Ratio Mismatch (SRM)."""
        counts = df[treatment_col].value_counts()
        n_treated = int(counts.get(1, 0))
        n_control = int(counts.get(0, 0))
        total = n_treated + n_control

        expected_t = total * expected_treatment_ratio
        expected_c = total * (1.0 - expected_treatment_ratio)

        observed = np.array([n_treated, n_control])
        expected = np.array([expected_t, expected_c])

        chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
        is_srm = bool(p_val < self.srm_alpha)

        if is_srm:
            msg = (
                f"Sample Ratio Mismatch detected (p={p_val:.4e} < {self.srm_alpha})! "
                f"Observed ratio {n_treated / total:.4f} differs significantly from expected {expected_treatment_ratio:.4f}. "
                "May indicate attrition, assignment corruption, or filtering bias."
            )
        else:
            msg = (
                f"No Sample Ratio Mismatch detected (p={p_val:.4f}). "
                f"Observed allocation ({n_treated:,} treated, {n_control:,} control) conforms to planned design."
            )

        return SRMTestResult(
            observed_treated=n_treated,
            observed_control=n_control,
            expected_ratio=expected_treatment_ratio,
            chi2_statistic=round(float(chi2_stat), 4),
            p_value=round(float(p_val), 6),
            is_srm_detected=is_srm,
            interpretation=msg,
        )

    def check_propensity_overlap(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        treatment_col: str = "treatment",
    ) -> Tuple[PropensityOverlapResult, np.ndarray]:
        """Fit baseline propensity model and check common support / overlap positivity.

        Returns
        -------
        Tuple[PropensityOverlapResult, np.ndarray]
            Diagnostic result and estimated propensity scores.
        """
        X = df[feature_cols].fillna(0.0).values
        y_t = df[treatment_col].values

        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X, y_t)
        prop_scores = clf.predict_proba(X)[:, 1]

        treated_scores = prop_scores[y_t == 1]
        control_scores = prop_scores[y_t == 0]

        t_min, t_max = float(np.min(treated_scores)), float(np.max(treated_scores))
        c_min, c_max = float(np.min(control_scores)), float(np.max(control_scores))

        common_min = max(t_min, c_min)
        common_max = min(t_max, c_max)

        # Extreme propensity scores (<0.02 or >0.98) violate practical positivity
        violations = (prop_scores < 0.02) | (prop_scores > 0.98)
        viol_count = int(np.sum(violations))
        viol_pct = float(viol_count / len(df))

        is_adequate = viol_pct < 0.05 and common_min < common_max

        summary = (
            f"Common support bounds: [{common_min:.3f}, {common_max:.3f}]. "
            f"Extreme positivity violations: {viol_count} ({viol_pct:.2%}). "
            + ("Overlap is adequate for causal adjustment." if is_adequate else "Overlap violations require trimming.")
        )

        res = PropensityOverlapResult(
            treated_min=round(t_min, 4),
            treated_max=round(t_max, 4),
            control_min=round(c_min, 4),
            control_max=round(c_max, 4),
            common_support_min=round(common_min, 4),
            common_support_max=round(common_max, 4),
            violations_count=viol_count,
            violations_pct=round(viol_pct, 4),
            is_overlap_adequate=is_adequate,
            summary=summary,
        )

        return res, prop_scores


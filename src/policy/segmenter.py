"""Customer Segmentation into Actionable Causal Archetypes.

Categorizes customer base into four canonical uplift quadrants:
1. Persuadables: Low-to-moderate baseline, high positive treatment effect (Primary Target).
2. Sure Things: High baseline conversion, negligible treatment effect (Wasted Budget).
3. Lost Causes: Low baseline conversion, negligible treatment effect (Unresponsive).
4. Sleeping Dogs / Do-Not-Target: Negative treatment effect (Harmful / Reactance).

Caveat: These segment assignments are analytical model approximations based on
estimated conditional expectations, not guaranteed individual causal truths.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CustomerSegment:
    segment_name: str
    count: int
    percentage: float
    avg_predicted_baseline: float
    avg_predicted_treated: float
    avg_estimated_uplift: float
    targeting_recommendation: str
    action_description: str


class CustomerSegmenter:
    """Classifies customers into actionable causal targeting segments."""

    def __init__(
        self,
        uplift_positive_threshold: float = 0.02,
        uplift_negative_threshold: float = -0.01,
        baseline_sure_thing_threshold: float = 0.20,
    ):
        self.uplift_positive_threshold = uplift_positive_threshold
        self.uplift_negative_threshold = uplift_negative_threshold
        self.baseline_sure_thing_threshold = baseline_sure_thing_threshold

    def segment_customers(
        self,
        pred_p0: np.ndarray,
        pred_p1: np.ndarray,
        customer_ids: Optional[List[str]] = None,
    ) -> Tuple[pd.Series, List[CustomerSegment]]:
        """Assign each customer to a causal segment based on predicted counterfactuals.

        Parameters
        ----------
        pred_p0 : np.ndarray
            Predicted baseline outcome probability P(Y(0)=1|X).
        pred_p1 : np.ndarray
            Predicted treated outcome probability P(Y(1)=1|X).
        customer_ids : Optional[List[str]]
            Optional identifiers for output DataFrame.

        Returns
        -------
        Tuple[pd.Series, List[CustomerSegment]]
            Series of segment assignments per row, and segment summary list.
        """
        uplift = pred_p1 - pred_p0
        n = len(pred_p0)
        labels = []

        for p0, p1, tau in zip(pred_p0, pred_p1, uplift):
            if tau > self.uplift_positive_threshold and p0 < 0.60:
                labels.append("Persuadables")
            elif tau < self.uplift_negative_threshold:
                labels.append("Sleeping Dogs")
            elif p0 >= self.baseline_sure_thing_threshold and abs(tau) <= self.uplift_positive_threshold:
                labels.append("Sure Things")
            else:
                labels.append("Lost Causes")

        series_labels = pd.Series(labels, name="causal_segment")

        # Compile summaries
        summaries = []
        rec_map = {
            "Persuadables": (
                "TARGET PRIORITY",
                "Allocate marketing budget here. These customers show strong positive incremental response.",
            ),
            "Sure Things": (
                "DO NOT TARGET (Organic)",
                "Suppression list. High organic purchase probability; sending discounts erodes profit margins.",
            ),
            "Lost Causes": (
                "DO NOT TARGET (Unresponsive)",
                "Low response in both conditions. Save promotional budget; consider alternative channels.",
            ),
            "Sleeping Dogs": (
                "STRICTLY EXCLUDE",
                "High churn/unsubscription risk when contacted. Exclude from all campaign touchpoints.",
            ),
        }

        for seg_name in ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]:
            mask = series_labels == seg_name
            cnt = int(mask.sum())
            pct = round(cnt / max(1, n) * 100.0, 2)

            if cnt > 0:
                avg_p0 = round(float(np.mean(pred_p0[mask])), 4)
                avg_p1 = round(float(np.mean(pred_p1[mask])), 4)
                avg_tau = round(float(np.mean(uplift[mask])), 5)
            else:
                avg_p0, avg_p1, avg_tau = 0.0, 0.0, 0.0

            rec, desc = rec_map[seg_name]
            summaries.append(
                CustomerSegment(
                    segment_name=seg_name,
                    count=cnt,
                    percentage=pct,
                    avg_predicted_baseline=avg_p0,
                    avg_predicted_treated=avg_p1,
                    avg_estimated_uplift=avg_tau,
                    targeting_recommendation=rec,
                    action_description=desc,
                )
            )

        return series_labels, summaries


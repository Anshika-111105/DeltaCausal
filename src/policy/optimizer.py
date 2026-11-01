"""Budget-Constrained Policy Optimization and Campaign Financial Simulator.

Simulates and solves for optimal customer targeting policies under budget constraints,
comparing counterfactual uplift targeting against conventional response propensity and random allocation.

Formulation:
    Expected Incremental Net Value per customer i:
        V_i = tau_hat(x_i) * margin_i - cost_i
    Maximize:
        Sum_{i in S} V_i
    Subject to:
        Sum_{i in S} cost_i <= Budget
        |S| <= Max_Interventions
        V_i > 0 (Rationality constraint: never target expected loss-making candidates)
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PolicySimulationResult:
    policy_name: str
    budget_allocated: float
    budget_spent: float
    customers_targeted: int
    targeting_rate: float
    expected_incremental_conversions: float
    expected_incremental_revenue: float
    expected_incremental_profit: float
    roi_percentage: float
    avg_net_value_per_targeted: float
    all_candidates_negative_value: bool = False


@dataclass
class SensitivityScenario:
    budget: float
    cost_per_intervention: float
    margin_per_conversion: float
    uncertainty_discount: float
    customers_targeted: int
    budget_spent: float
    expected_incremental_profit: float
    roi_percentage: float


class PolicyOptimizer:
    """Optimizes marketing targeting decisions under financial and capacity constraints."""

    def __init__(
        self,
        default_cost: float = 2.50,
        default_margin: float = 50.0,
        uncertainty_discount: float = 1.0,  # 1.0 = neutral, 0.8 = conservative
    ):
        self.default_cost = default_cost
        self.default_margin = default_margin
        self.uncertainty_discount = uncertainty_discount

    def optimize_policy(
        self,
        uplift_scores: np.ndarray,
        budget: float = 10000.0,
        cost_per_intervention: Optional[Union[float, np.ndarray]] = None,
        margin_per_conversion: Optional[Union[float, np.ndarray]] = None,
        max_interventions: Optional[int] = None,
        only_positive_value: bool = True,
        policy_name: str = "Uplift Targeting",
    ) -> Tuple[np.ndarray, PolicySimulationResult]:
        """Solve knapsack / greedy ranking targeting optimization.

        Parameters
        ----------
        uplift_scores : np.ndarray
            Predicted treatment effects tau(x).
        budget : float
            Total allowable campaign expenditure.
        cost_per_intervention : float or np.ndarray
            Cost of intervention per customer.
        margin_per_conversion : float or np.ndarray
            Gross margin per incremental conversion.
        max_interventions : Optional[int]
            Cap on total number of outreach actions.
        only_positive_value : bool
            If True, only target customers where Expected Net Value > 0.
        policy_name : str
            Name of policy for reporting.

        Returns
        -------
        Tuple[np.ndarray, PolicySimulationResult]
            Binary selection array (1=targeted, 0=suppressed), and financial simulation result.
        """
        n = len(uplift_scores)
        costs = np.full(n, cost_per_intervention if cost_per_intervention is not None else self.default_cost, dtype=float)
        margins = np.full(n, margin_per_conversion if margin_per_conversion is not None else self.default_margin, dtype=float)

        # Apply conservatism discount if requested
        discounted_uplift = uplift_scores * self.uncertainty_discount

        # Expected Incremental Net Value: V_i = tau_i * margin_i - cost_i
        net_values = discounted_uplift * margins - costs

        # ROI Ratio for fractional knapsack sorting: Net Value / Cost
        value_densities = net_values / np.maximum(costs, 0.01)

        # Sort indices in descending order of value density
        sorted_indices = np.argsort(-value_densities)

        selected_mask = np.zeros(n, dtype=bool)
        spent = 0.0
        targeted_count = 0
        cap = max_interventions if max_interventions is not None else n

        for idx in sorted_indices:
            # Rationality check: don't target expected negative value customers
            if only_positive_value and net_values[idx] <= 0.0:
                break

            c_i = costs[idx]
            if spent + c_i <= budget and targeted_count < cap:
                selected_mask[idx] = True
                spent += c_i
                targeted_count += 1
            elif spent + c_i > budget:
                # Budget exhausted
                break

        # Compute aggregate financial metrics for the policy
        if targeted_count > 0:
            exp_conversions = float(np.sum(discounted_uplift[selected_mask]))
            exp_revenue = float(np.sum(discounted_uplift[selected_mask] * margins[selected_mask]))
            exp_profit = float(exp_revenue - spent)
            roi = float((exp_profit / max(1.0, spent)) * 100.0)
            avg_net_val = float(exp_profit / targeted_count)
        else:
            exp_conversions = 0.0
            exp_revenue = 0.0
            exp_profit = 0.0
            roi = 0.0
            avg_net_val = 0.0

        all_neg = bool(np.all(net_values <= 0.0))

        sim_result = PolicySimulationResult(
            policy_name=policy_name,
            budget_allocated=float(budget),
            budget_spent=round(float(spent), 2),
            customers_targeted=targeted_count,
            targeting_rate=round(float(targeted_count / n), 4),
            expected_incremental_conversions=round(exp_conversions, 2),
            expected_incremental_revenue=round(exp_revenue, 2),
            expected_incremental_profit=round(exp_profit, 2),
            roi_percentage=round(roi, 2),
            avg_net_value_per_targeted=round(avg_net_val, 4),
            all_candidates_negative_value=all_neg,
        )

        return selected_mask.astype(int), sim_result

    def compare_targeting_policies(
        self,
        uplift_scores: np.ndarray,
        response_propensity_scores: np.ndarray,
        true_cate: Optional[np.ndarray] = None,
        budget: float = 10000.0,
        cost_per_intervention: float = 2.50,
        margin_per_conversion: float = 50.0,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Compare all canonical policies under the exact same marketing budget."""
        results = []

        # 1. No Campaign (Status Quo)
        _, res_none = self.optimize_policy(
            uplift_scores=np.zeros(len(uplift_scores)),
            budget=0.0,
            cost_per_intervention=cost_per_intervention,
            margin_per_conversion=margin_per_conversion,
            policy_name="Status Quo (No Campaign)",
        )
        results.append(asdict(res_none))

        # 2. Random Targeting
        rng = np.random.default_rng(random_seed)
        random_scores = rng.uniform(0.0, 1.0, size=len(uplift_scores))
        # Estimate expected conversions under random allocation = sample_rate * average ATE
        _, res_rand = self.optimize_policy(
            uplift_scores=random_scores,
            budget=budget,
            cost_per_intervention=cost_per_intervention,
            margin_per_conversion=margin_per_conversion,
            only_positive_value=False,
            policy_name="Random Targeting",
        )
        # Recalculate true expected conversion under random using overall mean uplift
        mean_uplift = float(np.mean(uplift_scores))
        rand_targeted = res_rand.customers_targeted
        rand_spent = rand_targeted * cost_per_intervention
        rand_conv = rand_targeted * mean_uplift
        rand_rev = rand_conv * margin_per_conversion
        rand_prof = rand_rev - rand_spent
        rand_roi = (rand_prof / max(1.0, rand_spent)) * 100.0

        res_rand.expected_incremental_conversions = round(rand_conv, 2)
        res_rand.expected_incremental_revenue = round(rand_rev, 2)
        res_rand.expected_incremental_profit = round(rand_prof, 2)
        res_rand.roi_percentage = round(rand_roi, 2)
        res_rand.avg_net_value_per_targeted = round(rand_prof / max(1, rand_targeted), 4)
        results.append(asdict(res_rand))

        # 3. Response Propensity Targeting (Conventional Marketing)
        # Propensity models rank by high conversion, not incremental lift
        _, res_prop = self.optimize_policy(
            uplift_scores=response_propensity_scores,
            budget=budget,
            cost_per_intervention=cost_per_intervention,
            margin_per_conversion=margin_per_conversion,
            only_positive_value=False,
            policy_name="Response Propensity Targeting",
        )
        # Evaluate incremental impact of the propensity-targeted cohort using their actual uplift
        order_prop = np.argsort(-response_propensity_scores)[:res_prop.customers_targeted]
        prop_conv = float(np.sum(uplift_scores[order_prop]))
        prop_rev = prop_conv * margin_per_conversion
        prop_spent = len(order_prop) * cost_per_intervention
        prop_prof = prop_rev - prop_spent
        prop_roi = (prop_prof / max(1.0, prop_spent)) * 100.0

        res_prop.expected_incremental_conversions = round(prop_conv, 2)
        res_prop.expected_incremental_revenue = round(prop_rev, 2)
        res_prop.expected_incremental_profit = round(prop_prof, 2)
        res_prop.roi_percentage = round(prop_roi, 2)
        res_prop.avg_net_value_per_targeted = round(prop_prof / max(1, len(order_prop)), 4)
        results.append(asdict(res_prop))

        # 4. Uplift Targeting (Our Engine)
        _, res_uplift = self.optimize_policy(
            uplift_scores=uplift_scores,
            budget=budget,
            cost_per_intervention=cost_per_intervention,
            margin_per_conversion=margin_per_conversion,
            policy_name="Counterfactual Uplift Targeting",
        )
        results.append(asdict(res_uplift))

        # 5. Oracle Targeting (if synthetic ground truth is provided)
        if true_cate is not None:
            _, res_oracle = self.optimize_policy(
                uplift_scores=true_cate,
                budget=budget,
                cost_per_intervention=cost_per_intervention,
                margin_per_conversion=margin_per_conversion,
                policy_name="Oracle Ground-Truth Policy",
            )
            results.append(asdict(res_oracle))

        df_comparison = pd.DataFrame(results)
        return df_comparison

    def run_sensitivity_grid(
        self,
        uplift_scores: np.ndarray,
        base_budget: float = 10000.0,
        base_cost: float = 2.50,
        base_margin: float = 50.0,
    ) -> pd.DataFrame:
        """Run multidimensional sensitivity analysis across budget, cost, margin, and uncertainty discount."""
        budget_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
        cost_multipliers = [0.8, 1.0, 1.2]
        margin_multipliers = [0.8, 1.0, 1.2]
        discounts = [0.8, 1.0]

        records = []
        for b_m in budget_multipliers:
            for c_m in cost_multipliers:
                for m_m in margin_multipliers:
                    for disc in discounts:
                        b = base_budget * b_m
                        c = base_cost * c_m
                        m = base_margin * m_m

                        opt = PolicyOptimizer(default_cost=c, default_margin=m, uncertainty_discount=disc)
                        _, res = opt.optimize_policy(uplift_scores, budget=b, cost_per_intervention=c, margin_per_conversion=m)

                        records.append({
                            "budget": b,
                            "cost": round(c, 2),
                            "margin": round(m, 2),
                            "uncertainty_discount": disc,
                            "customers_targeted": res.customers_targeted,
                            "budget_spent": res.budget_spent,
                            "expected_incremental_conversions": res.expected_incremental_conversions,
                            "expected_incremental_profit": res.expected_incremental_profit,
                            "roi_percentage": res.roi_percentage,
                        })

        return pd.DataFrame(records)


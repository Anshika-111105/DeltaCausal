import numpy as np
import pytest
from src.policy.optimizer import PolicyOptimizer
from src.policy.segmenter import CustomerSegmenter


def test_budget_constraint_is_strictly_respected():
    n = 1000
    uplifts = np.random.uniform(0.01, 0.15, size=n)
    cost = 2.50
    budget = 500.0  # Allows at most 200 customers (500 / 2.5)

    optimizer = PolicyOptimizer(default_cost=cost, default_margin=50.0)
    selected_mask, sim_res = optimizer.optimize_policy(
        uplift_scores=uplifts,
        budget=budget,
        cost_per_intervention=cost,
    )

    assert sim_res.budget_spent <= budget
    assert sim_res.customers_targeted <= 200
    assert np.sum(selected_mask) == sim_res.customers_targeted


def test_optimizer_handles_all_negative_net_values():
    """When all estimated treatment effects are negative or below cost threshold, do not target anyone."""
    n = 100
    # Uplifts negative or extremely tiny: V_i = 0.001 * 50 - 2.50 = 0.05 - 2.50 = -2.45
    negative_uplifts = np.full(n, -0.05)

    optimizer = PolicyOptimizer(default_cost=2.50, default_margin=50.0)
    selected_mask, sim_res = optimizer.optimize_policy(
        uplift_scores=negative_uplifts,
        budget=1000.0,
        only_positive_value=True,
    )

    assert sim_res.customers_targeted == 0
    assert sim_res.budget_spent == 0.0
    assert sim_res.expected_incremental_profit == 0.0
    assert sim_res.all_candidates_negative_value is True
    assert np.sum(selected_mask) == 0


def test_customer_segmenter_categorization():
    segmenter = CustomerSegmenter(
        uplift_positive_threshold=0.02,
        uplift_negative_threshold=-0.01,
        baseline_sure_thing_threshold=0.20,
    )
    # 4 distinct archetypes:
    # 1. Persuadable: p0=0.05, p1=0.15 (tau=0.10)
    # 2. Sleeping Dog: p0=0.10, p1=0.02 (tau=-0.08)
    # 3. Sure Thing: p0=0.50, p1=0.51 (tau=0.01)
    # 4. Lost Cause: p0=0.02, p1=0.02 (tau=0.00)
    p0 = np.array([0.05, 0.10, 0.50, 0.02])
    p1 = np.array([0.15, 0.02, 0.51, 0.02])

    labels, summaries = segmenter.segment_customers(p0, p1)
    assert list(labels) == ["Persuadables", "Sleeping Dogs", "Sure Things", "Lost Causes"]
    assert len(summaries) == 4


def test_optimizer_handles_series_lists_and_empty_inputs():
    import pandas as pd
    opt = PolicyOptimizer(default_cost=2.50, default_margin=50.0)

    # 1. Test with pandas Series having non-default custom index
    series_input = pd.Series([0.15, 0.08, -0.05], index=[100, 200, 300])
    mask, res = opt.optimize_policy(series_input, budget=100.0)
    assert len(mask) == 3
    assert res.customers_targeted == 2  # Only the 2 positive ones

    # 2. Test with python list
    list_input = [0.20, 0.10, -0.02]
    mask_list, res_list = opt.optimize_policy(list_input, budget=50.0)
    assert len(mask_list) == 3
    assert res_list.customers_targeted == 2

    # 3. Test with empty array (edge case N=0)
    empty_mask, empty_res = opt.optimize_policy(np.array([]), budget=100.0)
    assert len(empty_mask) == 0
    assert empty_res.customers_targeted == 0
    assert empty_res.budget_spent == 0.0


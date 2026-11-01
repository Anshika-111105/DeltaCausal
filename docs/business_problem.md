# Business Scenario & Problem Formulation

## The Marketing Dilemma: Propensity vs. Uplift

In conventional e-commerce marketing, businesses train supervised classification models to predict purchase probability:
$$P(Y = 1 \mid X = x)$$
Targeting customers with the highest predicted propensity creates a costly inefficiency:
- **Sure Things**: High-intent customers who would purchase organically without promotional coupons. Giving them a 20% discount or spending ad dollars reduces contribution margins without generating incremental revenue.
- **Lost Causes**: Disengaged customers who will not convert even with aggressive discounting.
- **Sleeping Dogs**: Customers who find frequent promotional emails intrusive, prompting unsubscribes or churn.

## Counterfactual Uplift Modeling

Uplift modeling reformulates marketing targeting from outcome prediction to **incremental effect optimization**:
$$\tau(x) = E[Y(1) - Y(0) \mid X = x]$$

Where:
- $X$: Pre-treatment customer characteristics (behavioral signals, engagement, spend history).
- $Y(1)$: Potential conversion outcome if the customer receives the promotion.
- $Y(0)$: Potential conversion outcome if the customer does NOT receive the promotion.
- $\tau(x)$: Conditional Average Treatment Effect (CATE).

## The Fundamental Problem of Causal Inference

For any observed customer $i$, we observe only the realized outcome under the assigned treatment condition:
$$Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)$$
We never simultaneously observe $Y_i(1)$ and $Y_i(0)$ for the same customer at the same instant. Therefore, individual treatment effects cannot be directly measured; the engine instead estimates conditional averages over observable subgroups $X=x$.

## Business Objective & Unit Economics

Under a finite campaign budget $B$, each intervention costs $c$ and each incremental conversion generates gross contribution margin $m$.

The Expected Incremental Net Value for customer $i$ is:
$$V_i = \hat{\tau}(x_i) \cdot m - c$$

The optimal policy selects the subset $S \subseteq \{1, \dots, N\}$ that solves:
$$\max_S \sum_{i \in S} V_i \quad \text{subject to} \quad \sum_{i \in S} c_i \le B \quad \text{and} \quad V_i > 0$$


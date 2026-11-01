# Causal Identification Assumptions & Experimental Diagnostics

To interpret estimated conditional average treatment effects causally rather than purely associational, several mathematical conditions must be satisfied.

## 1. Stable Unit Treatment Value Assumption (SUTVA)

SUTVA consists of two core tenets:
1. **No Interference**: The treatment assigned to customer $i$ does not affect the potential outcomes of customer $j$ ($Y_j(t_1, \dots, t_N) = Y_j(t_j)$).
   - *Risk in E-Commerce*: Referral programs, coupon code leakage on Reddit/Honey, viral social media ads.
   - *Mitigation*: Single-use personalized promo codes tied to customer accounts.
2. **No Hidden Variations of Treatment**: Treatment $T=1$ is homogeneous across all recipients (the same offer value, terms, and delivery channel).

## 2. Positivity (Overlap)

$$0 < P(T = 1 \mid X = x) < 1 \quad \forall x \text{ with } P(X = x) > 0$$
Every customer covariate profile must have a non-zero probability of being assigned to both treatment and control.
- If $P(T=1 \mid X=x) = 1$, we have no counterfactual control data to estimate $E[Y(0) \mid X=x]$.
- *Diagnostic*: The engine calculates propensity scores $e(X)$ using logistic regression and inspects the common support region $[\min_{T=1} e(X), \max_{T=0} e(X)]$. Records outside $[0.02, 0.98]$ trigger positivity warnings.

## 3. Unconfoundedness (Conditional Independence)

$$(Y(0), Y(1)) \perp T \mid X$$
Given pre-treatment covariates $X$, treatment assignment is conditionally independent of potential outcomes.
- In **Randomized Controlled Trials (RCTs)** (e.g. Criteo experiment), unconfoundedness holds unconditionally: $(Y(0), Y(1)) \perp T$.
- In **Observational Data**, unconfoundedness is an uncheckable assumption. All confounding factors influencing both marketing targeting and purchase probability must be measured in $X$.

## 4. Sample Ratio Mismatch (SRM) Diagnostic

When an experiment is planned with an allocation ratio $p_{\text{expected}}$ (e.g., 85% treated / 15% control in Criteo v2.1), deviation from the planned ratio suggests:
- Asymmetric attrition or technical drops.
- Faulty assignment randomization or hash modulo skew.
- Inconsistent post-assignment filtering.

The engine runs a Pearson Chi-Square goodness-of-fit test:
$$\chi^2 = \sum_{k \in \{0, 1\}} \frac{(O_k - E_k)^2}{E_k}$$
A $p$-value $< 0.01$ flags an SRM warning, indicating that internal validity may be compromised.

## 5. Covariate Balance & Standardized Mean Difference (SMD)

In a valid randomized experiment, pre-treatment covariates $X$ should have identical distributions across treatment and control:
$$\text{SMD}_j = \frac{\bar{X}_{1, j} - \bar{X}_{0, j}}{\sqrt{\frac{s_{1, j}^2 + s_{0, j}^2}{2}}}$$
Standard convention: $|\text{SMD}_j| < 0.10$ indicates satisfactory balance.


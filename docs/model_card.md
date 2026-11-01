# Model Card: Causal Meta-Learners Suite

## Model Details
- **Architecture**: Causal Meta-Learners (`SLearner`, `TLearner`, `XLearner`, `DoublyRobustLearner`).
- **Base Estimators**: Scikit-learn and XGBoost (`XGBClassifier`, `XGBRegressor`, `LogisticRegression`).
- **Version**: 1.0.0.
- **Frameworks**: Python 3.11, Scikit-Learn 1.5.2, XGBoost 3.2.0.

---

## Intended Use
- **Primary Use**: Estimating Conditional Average Treatment Effects (CATE) $\tau(x) = E[Y(1) - Y(0) \mid X=x]$ to allocate promotional marketing budgets and optimize customer targeting.
- **Outcomes**: Binary conversion outcome (purchase) and secondary outcome (website visit).
- **Users**: Growth teams, marketing operations, causal inference researchers, pricing strategists.

---

## Model Descriptions & Formulations

### 1. S-Learner (Single Model)
- Fits a single model $\mu(X, T)$ predicting $Y$ from augmented feature vector $[X, T]$.
- Counterfactual scoring: $\hat{\tau}(X) = \hat{\mu}(X, 1) - \hat{\mu}(X, 0)$.
- **Limitation**: Prone to treatment effect shrinkage when $X$ contains dominant predictive signals.

### 2. T-Learner (Two Models)
- Fits separate models $\mu_1(X)$ on treated cohort and $\mu_0(X)$ on control cohort.
- Uplift: $\hat{\tau}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$.
- **Advantage**: Prevents treatment shrinkage.
- **Limitation**: Can overfit if control group size is small.

### 3. X-Learner (Künzel et al. 2019)
- Specifically designed for unbalanced sample sizes (e.g. Criteo 85% treated / 15% control).
- Stage 1: Fit $\hat{\mu}_1(X)$ on treated and $\hat{\mu}_0(X)$ on control.
- Stage 2: Impute counterfactuals:
  $$D_1 = Y_1 - \hat{\mu}_0(X_1), \quad D_0 = \hat{\mu}_1(X_0) - Y_0$$
- Stage 3: Train effect regressors $\hat{\tau}_1(X)$ on $(X_1, D_1)$ and $\hat{\tau}_0(X)$ on $(X_0, D_0)$.
- Stage 4: Combine using propensity score $e(X)$:
  $$\hat{\tau}(X) = e(X) \hat{\tau}_0(X) + (1 - e(X)) \hat{\tau}_1(X)$$

### 4. Doubly Robust Learner (AIPW)
- Regresses AIPW pseudo-outcome:
  $$\Gamma_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{e(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1 - e(X_i)}$$
- **Property**: Consistent if either the outcome models or the propensity model is correctly specified.

---

## Measured Benchmark Performance (Held-Out Test Set)
From validated run `exp_20261030_133039` ($N=25,000$, Test $N=5,000$):

| Model | Qini Score | AUUC | Top 20% Lift | Test Profit ($) | ROI (%) | PEHE |
|:---|---:|---:|---:|---:|---:|---:|
| **T-Learner (Selected Best)** | **72.78** | **78.12** | **12.01%** | **$7,092.48** | **148.7%** | 0.0666 |
| Response Propensity Baseline | 70.44 | 65.45 | 13.10% | $2,338.71 | 23.4% | 0.0957 |
| Random Targeting Baseline | -3.07 | -8.34 | 2.78% | $43.25 | 0.4% | 0.5406 |
| Status Quo (No Campaign) | 0.00 | 0.00 | 0.00% | $0.00 | 0.0% | — |

*Note: Counterfactual uplift targeting generated **3.0x higher incremental profit** ($7,092 vs $2,338) while spending less than half the budget by suppressing non-persuadable cohorts.*


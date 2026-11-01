# Uplift Evaluation Methodology & Metrics

## Why Standard ML Metrics Fail for Causal Models

Standard classification metrics (LogLoss, ROC-AUC, F1-Score) evaluate:
$$P(Y = 1 \mid X = x)$$
They quantify how well a model predicts *who will buy*. In uplift modeling, our target is the unobservable counterfactual contrast:
$$\tau(x) = E[Y(1) - Y(0) \mid X = x]$$
Because no customer is simultaneously treated and untreated, we cannot compute $(y_i - \hat{\tau}_i)^2$ on individual empirical records.

Instead, uplift models are evaluated by measuring the **empirical concentration of incremental conversions** when customers are sorted by predicted CATE.

---

## 1. Qini Curve & Qini Coefficient

Developed by Nicholas Radcliffe (2007) specifically for marketing campaigns with unequal treatment and control sample sizes.

### Formula
For a cohort of top $k$ customers ranked in descending order of predicted uplift:
$$Q(k) = Y_{T, k} - Y_{C, k} \cdot \left( \frac{N_{T, \text{total}}}{N_{C, \text{total}}} \right)$$

Where:
- $Y_{T, k}$: Cumulative conversions among treated units in top $k$.
- $Y_{C, k}$: Cumulative conversions among control units in top $k$.
- $\frac{N_{T, \text{total}}}{N_{C, \text{total}}}$: Global treatment-to-control ratio (e.g. $85 / 15 \approx 5.67$ in Criteo).

### Qini Score
$$\text{Qini Score} = \int_0^1 Q(\phi) \, d\phi - \int_0^1 Q_{\text{random}}(\phi) \, d\phi$$
Measures the area between the model's Qini curve and the diagonal random baseline. Higher positive values indicate superior prioritization of persuadable customers.

---

## 2. Area Under the Uplift Curve (AUUC)

Cumulative empirical uplift at fraction $\phi = k / N$:
$$U(k) = \left( \frac{Y_{T, k}}{N_{T, k}} - \frac{Y_{C, k}}{N_{C, k}} \right) \cdot k$$
The AUUC is the trapezoidal integral of $U(k)$ over $\phi \in [0, 1]$ minus the random baseline.

---

## 3. Top-Decile Uplift & Incremental Lift per 1,000

Measures the treatment effect localized to the top targeted deciles (e.g. top 10% or top 20%):
$$\Delta p_{20\%} = \frac{Y_{T, 20\%}}{N_{T, 20\%}} - \frac{Y_{C, 20\%}}{N_{C, 20\%}}$$
$$\text{Incremental Conversions per 1,000} = 1,000 \cdot \Delta p_{20\%}$$

---

## 4. Ground Truth CATE Error (Synthetic Data)

When evaluating synthetic benchmarks with mathematically known potential outcomes:

### PEHE (Precision in Estimation of Heterogeneous Effects)
$$\text{PEHE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( \hat{\tau}_i - \tau_i^* \right)^2}$$

### CATE Rank Correlation
- Pearson correlation: $r(\hat{\tau}, \tau^*)$
- Spearman rank correlation: $\rho(\hat{\tau}, \tau^*)$
Indicates whether the model ranks customers in the exact order of true responsiveness.

---

## 5. Bootstrap Confidence Intervals

To prevent spurious conclusions from sampling noise, held-out evaluation computes $B=50$ to $100$ bootstrap resamples of the test set, reporting the 2.5% and 97.5% quantiles for Qini and AUUC.


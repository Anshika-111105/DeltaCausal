# Limitations, Risk Analysis & Recommended Safeguards

## 1. Methodological Limitations

1. **Individual Counterfactual Unobservability**:
   The engine estimates Conditional Average Treatment Effects over feature slices $X=x$, not deterministic individual causal truths. Individual predictions carry irreducible variance.
2. **Unmeasured Confounding in Observational Deployments**:
   If deployed on observational CRM data where marketing touches were not randomized, unobserved attributes (e.g. offline retail store visits, competitor promotions, brand sentiment) violate the unconfoundedness assumption.
3. **SUTVA Violation Risks**:
   Promotional coupons shared via browser extensions (e.g., Honey, Coupert) or social media introduce interference between customers, biasing control group conversion baselines upward.

---

## 2. Operational & Business Safeguards

1. **Randomized Holdout Groups**:
   Never deploy an uplift targeting policy to 100% of candidate customers without maintaining a dedicated 10% pure control holdout group. This enables continuous empirical measurement of true realized lift.
2. **Rationality Constraint**:
   The policy optimizer enforces $V_i = \hat{\tau}(x_i) \cdot m - c > 0$. If expected incremental margin does not exceed coupon cost, the engine suppresses targeting even if budget remains.
3. **Conservatism Discount Factor**:
   Decision-makers can apply an uncertainty discount $\alpha \in [0.7, 0.9]$ to shrink noisy CATE estimates before budget commitment:
   $$\hat{\tau}_{\text{conservative}}(x) = \hat{\tau}(x) \cdot \alpha$$

---

## 3. Recommended Follow-Up Experiments

If validation Qini score is low or bootstrap confidence intervals span zero:
1. **Targeted Micro-A/B Trial**:
   Run a two-week randomized trial with equal 50/50 split restricted to the highest-uncertainty deciles (deciles 4 to 6).
2. **Dose-Response Trial**:
   Test multiple promotional incentive levels ($5 discount, $10 discount, free shipping) to model multi-action treatment effects rather than a single binary treatment.


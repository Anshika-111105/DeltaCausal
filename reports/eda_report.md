# Exploratory Data Analysis & Causal Diagnostics Report

## 1. Dataset Dimensions & Summary

- **Total Records Analyzed**: 25,000 customers
- **Feature Dimensionality**: 12 pre-treatment continuous covariates (`f0` – `f11`)
- **Treatment Assignment Ratio**: 85.0% Treated ($N_T = 21,250$), 15.0% Control ($N_C = 3,750$)
- **Missing Values**: 0 (Complete dataset)
- **Duplicate Records**: 0

---

## 2. Empirical Outcome Rates & Naive ATE

| Group | Population ($N$) | Conversions ($Y=1$) | Conversion Rate ($p$) |
|:---|---:|---:|---:|
| **Control ($T=0$)** | 3,750 | 32 | 0.85% |
| **Treated ($T=1$)** | 21,250 | 1,107 | 5.21% |
| **Overall Cohort** | 25,000 | 1,139 | 4.56% |

- **Naive Observed ATE**: $5.21\% - 0.85\% = +4.36\%$ (Risk Difference).
- **Causal Interpretation Guard**: This observed empirical difference reflects a valid causal effect only under randomized assignment where $(Y(0), Y(1)) \perp T$.

---

## 3. Covariate Balance & Pre-Treatment Diagnostics

Standardized Mean Differences (SMD) and two-sample Kolmogorov-Smirnov tests across all 12 pre-treatment features:

| Feature | Mean Treated | Mean Control | Std Treated | Std Control | SMD | KS Stat | KS P-Value | Imbalance Flag |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| `f0` | -0.0051 | 0.0124 | 0.9982 | 1.0121 | -0.0174 | 0.0142 | 0.4721 | Balanced |
| `f1` | 0.0082 | -0.0041 | 1.0021 | 0.9945 | 0.0123 | 0.0118 | 0.6912 | Balanced |
| `f2` | -0.0019 | 0.0095 | 0.9954 | 1.0084 | -0.0114 | 0.0135 | 0.5410 | Balanced |
| `f3` | 0.0044 | -0.0121 | 1.0012 | 0.9978 | 0.0165 | 0.0129 | 0.6012 | Balanced |
| `f4` | -0.0022 | 0.0081 | 0.9976 | 1.0034 | -0.0103 | 0.0112 | 0.7412 | Balanced |
| `f5` | 0.0061 | -0.0052 | 1.0045 | 0.9912 | 0.0113 | 0.0120 | 0.6721 | Balanced |
| `f6` | -0.0035 | 0.0048 | 0.9989 | 1.0012 | -0.0083 | 0.0098 | 0.8512 | Balanced |
| `f7` | 0.0019 | -0.0032 | 1.0004 | 0.9967 | 0.0051 | 0.0105 | 0.8014 | Balanced |
| `f8` | -0.0048 | 0.0062 | 0.9971 | 1.0045 | -0.0110 | 0.0124 | 0.6421 | Balanced |
| `f9` | 0.0072 | -0.0084 | 1.0032 | 0.9987 | 0.0156 | 0.0131 | 0.5812 | Balanced |
| `f10` | -0.0012 | 0.0031 | 0.9991 | 1.0023 | -0.0043 | 0.0092 | 0.8912 | Balanced |
| `f11` | 0.0038 | -0.0049 | 1.0018 | 0.9974 | 0.0087 | 0.0115 | 0.7214 | Balanced |

**Diagnostic Conclusion**: All features exhibit $|SMD| < 0.02 \ll 0.10$ and KS test $p$-values $> 0.45$. No evidence of pre-treatment covariate imbalance or selection bias.

---

## 4. Sample Ratio Mismatch (SRM) Test

- **Planned Allocation Ratio**: 85.0% Treatment / 15.0% Control
- **Observed Counts**: $N_T = 21,250$, $N_C = 3,750$
- **Chi-Square Statistic**: 0.0000 ($p = 1.0000$)
- **SRM Status**: PASSED. No evidence of attrition, assignment leakage, or sample distortion.

---

## 5. Propensity Overlap & Positivity Check

- **Common Support Bounds**: $[0.842, 0.858]$ (Randomized assignment)
- **Positivity Violations**: 0%
- **Status**: PASSED. Positivity assumption holds across the entire covariate distribution.


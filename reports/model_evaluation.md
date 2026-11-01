# Model Evaluation & Benchmark Report: `exp_20261101_114818`
- **Primary Selection Metric**: `qini_score`
- **Winning Model**: `two_model_naive`

## Validation Set Comparison
| model               |   qini_score |    auuc |   uplift_top_10pct |   uplift_top_20pct |   incremental_per_1000 |   policy_net_value_top_20pct |
|:--------------------|-------------:|--------:|-------------------:|-------------------:|-----------------------:|-----------------------------:|
| random              |      -1.49   | 10.0392 |            0.01472 |            0.04074 |                  40.74 |                      -370.53 |
| zero_intervention   |      -7.657  | -4.5641 |            0.04202 |            0.0563  |                  56.3  |                       251.85 |
| overall_ate         |      -7.657  | -4.5641 |            0.04202 |            0.0563  |                  56.3  |                       251.85 |
| response_propensity |      41.5021 | 58.0634 |            0.20715 |            0.16295 |                 162.95 |                      4517.86 |
| two_model_naive     |      40.6378 | 58.1577 |            0.22339 |            0.17955 |                 179.55 |                      5181.95 |
| s_learner           |      39.6205 | 58.5192 |            0.20224 |            0.16172 |                 161.72 |                      4468.76 |
| t_learner           |      31.542  | 51.5528 |            0.14509 |            0.16017 |                 160.17 |                      4406.8  |
| x_learner           |      32.8253 | 46.8088 |            0.16257 |            0.13554 |                 135.54 |                      3421.69 |
| doubly_robust       |      37.5557 | 52.046  |            0.24406 |            0.15097 |                 150.97 |                      4038.83 |

## Held-Out Test Set Performance
| model               |   qini_score |    auuc |   uplift_top_10pct |   uplift_top_20pct |   incremental_per_1000 |   policy_net_value_top_20pct |
|:--------------------|-------------:|--------:|-------------------:|-------------------:|-----------------------:|-----------------------------:|
| two_model_naive     |      59.768  | 73.1053 |            0.23844 |            0.17134 |                 171.34 |                      4853.59 |
| response_propensity |      44.9649 | 59.1944 |            0.21044 |            0.17048 |                 170.48 |                      4819.18 |
| random              |      12.5378 | 12.144  |            0.05742 |            0.05906 |                  59.06 |                       362.32 |

## Bootstrap Confidence Intervals (95% CI)
- **two_model_naive**: Qini Score CI: `(10.5064, 69.979)`, AUUC CI: `(32.5923, 89.4306)`
- **response_propensity**: Qini Score CI: `(17.9077, 73.5815)`, AUUC CI: `(32.4897, 93.3074)`
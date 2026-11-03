# Campaign Policy Simulation & Optimization Report: `exp_20261103_021444`
- **Budget**: $10,000.00
- **Intervention Cost per Customer**: $2.50
- **Gross Margin per Conversion**: $50.00

## Targeting Policy Comparison
| policy_name                     |   budget_allocated |   budget_spent |   customers_targeted |   targeting_rate |   expected_incremental_conversions |   expected_incremental_revenue |   expected_incremental_profit |   roi_percentage |   avg_net_value_per_targeted | all_candidates_negative_value   |
|:--------------------------------|-------------------:|---------------:|---------------------:|-----------------:|-----------------------------------:|-------------------------------:|------------------------------:|-----------------:|-----------------------------:|:--------------------------------|
| Status Quo (No Campaign)        |                  0 |            0   |                    0 |           0      |                               0    |                           0    |                          0    |             0    |                       0      | True                            |
| Random Targeting                |              10000 |        10000   |                 4000 |           0.4    |                             140.62 |                        7031.06 |                      -2968.94 |           -29.69 |                      -0.7422 | False                           |
| Response Propensity Targeting   |              10000 |        10000   |                 4000 |           0.4    |                             284.87 |                       14243.5  |                       4243.45 |            42.43 |                       1.0609 | False                           |
| Counterfactual Uplift Targeting |              10000 |         5407.5 |                 2163 |           0.2163 |                             236.54 |                       11826.9  |                       6419.36 |           118.71 |                       2.9678 | False                           |
| Oracle Ground-Truth Policy      |              10000 |         6942.5 |                 2777 |           0.2777 |                             347.91 |                       17395.3  |                      10452.8  |           150.56 |                       3.7641 | False                           |

## Customer Segmentation Breakdown
### Persuadables (46.2% of cohort, 4,620 customers)
- **Recommendation**: `TARGET PRIORITY`
- **Avg Predicted Uplift**: +0.0682
- **Avg Baseline Rate P(Y(0))**: 0.1327
- **Action**: Allocate marketing budget here. These customers show strong positive incremental response.

### Sure Things (0.4% of cohort, 40 customers)
- **Recommendation**: `DO NOT TARGET (Organic)`
- **Avg Predicted Uplift**: +0.0063
- **Avg Baseline Rate P(Y(0))**: 0.2367
- **Action**: Suppression list. High organic purchase probability; sending discounts erodes profit margins.

### Lost Causes (51.45% of cohort, 5,145 customers)
- **Recommendation**: `DO NOT TARGET (Unresponsive)`
- **Avg Predicted Uplift**: +0.0078
- **Avg Baseline Rate P(Y(0))**: 0.0554
- **Action**: Low response in both conditions. Save promotional budget; consider alternative channels.

### Sleeping Dogs (1.95% of cohort, 195 customers)
- **Recommendation**: `STRICTLY EXCLUDE`
- **Avg Predicted Uplift**: -0.0203
- **Avg Baseline Rate P(Y(0))**: 0.1247
- **Action**: High churn/unsubscription risk when contacted. Exclude from all campaign touchpoints.

# Campaign Policy Simulation & Optimization Report: `exp_20261101_114818`
- **Budget**: $10,000.00
- **Intervention Cost per Customer**: $2.50
- **Gross Margin per Conversion**: $50.00

## Targeting Policy Comparison
| policy_name                     |   budget_allocated |   budget_spent |   customers_targeted |   targeting_rate |   expected_incremental_conversions |   expected_incremental_revenue |   expected_incremental_profit |   roi_percentage |   avg_net_value_per_targeted | all_candidates_negative_value   |
|:--------------------------------|-------------------:|---------------:|---------------------:|-----------------:|-----------------------------------:|-------------------------------:|------------------------------:|-----------------:|-----------------------------:|:--------------------------------|
| Status Quo (No Campaign)        |                  0 |              0 |                    0 |           0      |                               0    |                           0    |                          0    |             0    |                       0      | True                            |
| Random Targeting                |              10000 |          10000 |                 4000 |           1      |                             201.45 |                       10072.3  |                         72.3  |             0.72 |                       0.0181 | False                           |
| Response Propensity Targeting   |              10000 |          10000 |                 4000 |           1      |                             201.45 |                       10072.3  |                         72.3  |             0.72 |                       0.0181 | False                           |
| Counterfactual Uplift Targeting |              10000 |           3395 |                 1358 |           0.3395 |                             169.35 |                        8467.55 |                       5072.55 |           149.41 |                       3.7353 | False                           |

## Customer Segmentation Breakdown
### Persuadables (55.2% of cohort, 2,208 customers)
- **Recommendation**: `TARGET PRIORITY`
- **Avg Predicted Uplift**: +0.0894
- **Avg Baseline Rate P(Y(0))**: 0.1150
- **Action**: Allocate marketing budget here. These customers show strong positive incremental response.

### Sure Things (0.07% of cohort, 3 customers)
- **Recommendation**: `DO NOT TARGET (Organic)`
- **Avg Predicted Uplift**: +0.0058
- **Avg Baseline Rate P(Y(0))**: 0.2286
- **Action**: Suppression list. High organic purchase probability; sending discounts erodes profit margins.

### Lost Causes (40.35% of cohort, 1,614 customers)
- **Recommendation**: `DO NOT TARGET (Unresponsive)`
- **Avg Predicted Uplift**: +0.0043
- **Avg Baseline Rate P(Y(0))**: 0.0435
- **Action**: Low response in both conditions. Save promotional budget; consider alternative channels.

### Sleeping Dogs (4.38% of cohort, 175 customers)
- **Recommendation**: `STRICTLY EXCLUDE`
- **Avg Predicted Uplift**: -0.0168
- **Avg Baseline Rate P(Y(0))**: 0.0679
- **Action**: High churn/unsubscription risk when contacted. Exclude from all campaign touchpoints.

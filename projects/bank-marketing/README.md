# Bank Marketing Under a Call-Time Budget

A bank must decide which clients to contact when calls consume staff time. This case connects subscription prediction to targeting policies, and asks whether better ranking translates into a feasible contact plan.

## Data and workflow

The supplied UCI Bank Marketing data contains **41,188 records**. The classifier uses 19 predictors available in the project before the contact decision; observed call duration and the subscription outcome are excluded from its inputs.

1. Split rows into stratified training, validation, and test sets: **24,712 / 8,238 / 8,238**.
2. Compare logistic regression, a 300-tree random forest, and a sigmoid-calibrated forest. Select the classifier with the lowest validation Brier score.
3. Compare log-duration linear regression, log-duration forest, and a training-mean baseline. Select the duration model by validation RMSE.
4. Evaluate no-contact, all-contact, random, probability-ranked, and duration-aware policies.
5. Choose safety buffers from `[0.6, 0.7, 0.8, 0.9, 1.0]` using validation simulated profit among candidates whose actual validation time meets the budget; then apply the fixed choices to test rows.

Each estimator has its own preprocessing pipeline. For probability calibration, the entire pipeline is fitted separately within each calibration fold. All raw preprocessing is learned from training rows.

## Reproduced classification results

| Model | Split | ROC-AUC | Average precision | Brier score ↓ |
| --- | --- | ---: | ---: | ---: |
| Logistic regression | Validation | 0.7951 | 0.4463 | 0.07886 |
| Random forest | Validation | 0.7994 | 0.4660 | 0.11510 |
| Calibrated random forest | Validation | 0.8012 | 0.4659 | 0.07747 |
| Selected calibrated forest | Test | **0.8092** | **0.4886** | **0.07470** |

Calibration substantially improves probability quality relative to the class-weighted forest. It does not win every ranking metric: the uncalibrated forest has slightly higher validation average precision.

## Capacity is a separate prediction problem

| Duration model | Validation MAE, seconds ↓ | Validation RMSE, seconds ↓ |
| --- | ---: | ---: |
| Training mean | 170.65 | **254.48** |
| Log-linear regression | 154.16 | 264.58 |
| Log-random forest | **153.35** | 262.51 |

The log forest improves MAE but has negative validation R² and worse RMSE than the mean. Exponentiating a log prediction does not in general estimate the conditional mean call length. The comparison retains that model as an experimental policy while separately reporting the RMSE-selected mean baseline.

Baseline economic assumptions: 100 monetary units per subscription, 1 per attempted contact, 4 per call minute, and a 3,000-minute budget. Under the log-duration forest:

| Test policy | Buffer | Contacts | Predicted minutes | Actual minutes | Actual overrun |
| --- | ---: | ---: | ---: | ---: | ---: |
| Probability ranking | 1.0 | 776 | 2,998.46 | 3,893.45 | 29.78% |
| Duration-aware ranking | 1.0 | 875 | 2,994.57 | 4,051.45 | 35.05% |
| Probability ranking, validation buffer | 0.8 | 602 | 2,395.84 | 3,110.12 | 3.67% |
| Duration-aware ranking, validation buffer | 0.7 | 624 | 2,097.43 | 2,828.40 | 0.00% |

The buffer improves feasibility in this example, with a smaller contact plan. It is not a general guarantee: the validation-selected buffer for the mean-duration model still exceeds actual test capacity by **6.42%**. Both duration-model variants and both economic scenarios are retained in the exported results.

![Calibration and capacity](figures/calibration_and_capacity.svg)

## Explore and reproduce

- [Notebook walkthrough](analysis.ipynb)
- [Classifier results](results/classification.csv) and [duration results](results/duration.csv)
- [Full test policy table](results/policies_test.csv)
- [Validation buffer search](results/buffer_search_validation.csv)
- [Recorded environment and parameters](results/run.json)
- [Runnable experiment](../../src/decision_analytics/bank.py) and [selection/evaluation functions](../../src/decision_analytics/policies.py)
- [Original source notebook](../../archive/bank-marketing/original_source.ipynb) and [original report](../../archive/bank-marketing/original_report.pdf)

From the repository root, install with `python -m pip install -e .`, then run `python -m decision_analytics.bank`. The default reproduces the 300-tree configuration. `--trees` supports smaller exploratory runs; such runs overwrite exports and should not be compared with the recorded configuration as if identical.

## Interpretation limits

The source is chronologically ordered, but this experiment uses a random split. An out-of-time campaign test remains necessary. Outcomes for contacted historical clients do not identify what would happen without contact. Costs and revenue are assumptions; profits are historical simulations. Selection is a sorted-prefix heuristic, not an exact knapsack optimizer. Safety buffers are finite validation searches, not statistical chance constraints. The original test set had already been reviewed before this refactor.

The original report is retained for provenance. It predates the curated calibration pipeline, mean-duration baseline, and validation-selected buffer experiment; use the current CSVs for the reproduced results above.

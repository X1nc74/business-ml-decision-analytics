# Additional case reviews and next implementation steps

The supplied batch contains four distinct new cases, plus the previously supplied bank case. Duplicate causal notebooks and the empty midterm template are not separate projects. Two data-complete cases have been refactored and rerun. The following three retain source snapshots and review findings; their historical output metrics are not presented as validated portfolio results.

## Multi-SKU inventory optimization

**Business question:** how should a retailer turn demand forecasts into SKU-specific order quantities under asymmetric stockout/overstock costs and a procurement budget?

The notebook contains substantive work: AR/LightGBM comparisons, quantile forecasting, isotonic correction of quantile crossing, interpolation to cost-specific critical quantiles, global versus volume-tier models, budget allocation, and risk/return portfolio heuristics. Saved outputs cover 823 SKUs and a final 28-day evaluation window.

Required changes before promotion:

1. Recover the exact `sales_all_sku.csv`, or provide a documented reconstruction pipeline. That file is not present in this supplied batch; saved outputs alone do not make the experiment reproducible.
2. Compute `avg_sales`, `cv`, `zero_frac`, and volume-tier cutoffs from training dates only. The supplied setup computes these over the full dataset before splitting, then uses them as predictors.
3. State the forecasting horizon. Shifted lag features allow observed earlier holdout days to enter later predictions, which is consistent with a rolling one-step experiment, not a single untouched 28-day-ahead forecast.
4. Revise the Normal-approximation baseline: the auxiliary model is trained on absolute residuals and its output is called `sigma_hat`. Expected absolute error is not automatically a conditional standard deviation. Use a justified scale conversion/distribution, residual variance model, or direct quantile baseline.
5. Separate portfolio design from evaluation. The final SKU risk statistics and selected portfolio use the same evaluation-period returns. Fit selection on an earlier period and report subsequent performance; also state the independence assumption behind adding variances without covariance.
6. Describe the margin/cost greedy allocator as a heuristic. It does not optimize diminishing marginal expected newsvendor profit, and the saved comparison does not show that greedy allocation beats proportional scaling.

Suggested case directory after repair: `projects/inventory-optimization/`.

## Hotel cancellation and overbooking

**Business question:** how many reservations should a hotel accept when cancellations create empty rooms and overselling incurs relocation costs?

The notebook connects logistic cancellation models, feature engineering, tuning, calibration, Monte Carlo arrival probabilities, a constrained stopping rule, and paired policy comparisons. It is a hotel case, not an airline case.

Required changes before promotion:

1. Preserve source row IDs through the train/test split. The data-loading cell resets `X_train_base` and `X_test_base` indexes. The feature-engineering cell then reads `df_raw.loc[X_train_base.index]` and the corresponding test expression, which select unrelated raw rows. Store the split IDs before resetting and assert that raw cancellation labels match the aligned labels.
2. Rerun the feature-engineering, tuning, calibration, and policy comparisons after alignment is fixed. The existing model-improvement conclusions are affected by that bug.
3. Estimate the population cancellation baseline from training or validation data rather than `y_test.mean()` for a forward-use experiment. The supplied exercise explicitly uses the test rate, so the original scenario and the revised deployment protocol should be distinguished.
4. Keep simulation policy comparisons paired. The Normal-approximation comparison uses a reset seed; the Monte Carlo version consumes extra random numbers during each policy, so later sampled queues can differ across configurations. Pre-generate booking queues and use a separate Monte Carlo random stream.
5. Retain risk measurements, uncertainty, and approximation differences. Small simulated profit differences are not enough to establish that feature engineering or calibration improves business value.

The supplied source downloads `jessemostipak/hotel-booking-demand` through KaggleHub. Raw data is not bundled in this upload batch, and no full rerun of this case was performed here.

Suggested case directory after repair: `projects/hotel-overbooking/`.

## Causal coupon targeting

**Business question:** can estimated heterogeneous treatment effects improve who receives an offer under cost and fairness constraints?

The notebook implements OLS, S/T learners, causal forests, DML variants, cross-fitting, AUUC/Qini comparisons, budget curves, and a fairness-constrained linear program. This provides technical breadth, but credible causal identification must precede presenting business uplift.

Required changes before promotion:

1. Split unique customers before any bootstrap/resampling. The current pipeline samples each treatment arm with replacement to 50,000 rows and only then randomly splits rows. Copies of the same customer can enter both training and test sets. The 100,000 rows are not 100,000 distinct customers.
2. Define comparable observation windows and verify the source time unit. The code uses `window_end = median_time + 14`; treated transactions are after the individual's first offer, while control transactions are before their own first offer. That code alone does not establish a common 14-day exposure period.
3. Justify treatment assignment and confounding control. Treatment is early versus late first-offer timing, not an assignment shown to be randomized in this notebook. A within-selected-group mean difference is not automatically an unbiased treatment effect.
4. Reserve separate model-selection and final evaluation populations. The named winner follows test AUUC inspection; final policy evaluation on that same sample is exploratory.
5. Compare fairness and unconstrained policies under the same objective and spending rule. The budget curve forces exactly `k` targets, while the LP permits spending below budget and can avoid negative-net-value targets. Their reported profit gap therefore does not isolate a cost of fairness.
6. Recheck all fairness and budget constraints after binarizing the continuous LP solution. Reporting the continuous selection-rate constraint does not establish that the final thresholded policy satisfies it.

The two uploaded variants have exactly matching cell sources. The `new_` prefix does not indicate a substantive model revision. This case requires the KaggleHub Starbucks data and optional EconML/LightGBM dependencies, which are not part of the featured pipelines' environment.

Suggested case directory after repair: `projects/causal-targeting/`.

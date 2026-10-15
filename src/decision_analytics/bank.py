"""Train, validate, and evaluate budget-aware bank marketing policies."""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .common import ROOT, SEED, environment, model_pipeline, read_dataset, write_json
from .policies import bank_policies, evaluate_bank


def classification_metrics(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier_score": float(brier_score_loss(y, p))}


def run(trees=300):
    started = time.monotonic()
    folder = ROOT / "projects" / "bank-marketing"
    results, figures = folder / "results", folder / "figures"
    results.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    df = read_dataset("bank-marketing", "bank-additional-full.csv", sep=";")
    features = [c for c in df if c not in ["y", "duration"]]
    x, y = df[features], df.y.eq("yes").astype(int)
    tv, te = train_test_split(np.arange(len(df)), test_size=.2, stratify=y, random_state=SEED)
    tr, va = train_test_split(tv, test_size=.25, stratify=y.iloc[tv], random_state=SEED)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    pd.DataFrame({"row_id": np.concatenate([tr, va, te]), "split": ["train"]*len(tr)+["validation"]*len(va)+["test"]*len(te)}).to_csv(results / "row_splits.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    train = x.iloc[tr]

    def rf():
        return RandomForestClassifier(n_estimators=trees, min_samples_leaf=5,
                class_weight="balanced_subsample", random_state=SEED, n_jobs=2)

    classifiers = {
        "Logistic regression": model_pipeline(train, LogisticRegression(max_iter=2000, random_state=SEED)),
        "Random forest": model_pipeline(train, rf()),
        # Include the entire preprocessing pipeline inside calibration CV.
        "Calibrated random forest": CalibratedClassifierCV(
            model_pipeline(train, rf()), method="sigmoid", cv=3, n_jobs=1),
    }
    cls_rows, valid_probs = [], {}
    for name, model in classifiers.items():
        print(f"[bank] Fitting {name}", flush=True)
        model.fit(train, y.iloc[tr])
        valid_probs[name] = model.predict_proba(x.iloc[va])[:, 1]
        cls_rows.append({"model": name, "split": "validation", **classification_metrics(y.iloc[va], valid_probs[name])})
    cls_df = pd.DataFrame(cls_rows)
    winner = cls_df.sort_values(["brier_score", "model"]).iloc[0].model
    model = classifiers[winner]
    p_test = model.predict_proba(x.iloc[te])[:, 1]
    pd.concat([cls_df, pd.DataFrame([{"model": winner, "split": "test", **classification_metrics(y.iloc[te], p_test)}])], ignore_index=True).to_csv(results / "classification.csv", index=False)

    durations = df.duration.to_numpy(dtype=float)
    regressors = {
        "Training mean": (model_pipeline(train, DummyRegressor(strategy="mean")), False),
        "Log-linear regression": (model_pipeline(train, LinearRegression()), True),
        "Log-random forest": (model_pipeline(train, RandomForestRegressor(n_estimators=trees, min_samples_leaf=10, random_state=SEED, n_jobs=2)), True),
    }
    duration_rows, duration_valid = [], {}
    for name, (reg, log_target) in regressors.items():
        print(f"[bank] Fitting {name}", flush=True)
        reg.fit(train, np.log1p(durations[tr]) if log_target else durations[tr])
        raw = reg.predict(x.iloc[va])
        pred = np.maximum(np.expm1(raw) if log_target else raw, 1.0)
        duration_valid[name] = pred
        duration_rows.append({"model": name, "split": "validation", "mae_seconds": float(mean_absolute_error(durations[va], pred)), "rmse_seconds": float(np.sqrt(mean_squared_error(durations[va], pred))), "r2": float(r2_score(durations[va], pred))})
    dur_df = pd.DataFrame(duration_rows)
    duration_winner = dur_df.sort_values(["rmse_seconds", "model"]).iloc[0].model
    reg, log_target = regressors[duration_winner]
    raw = reg.predict(x.iloc[te])
    d_test = np.maximum(np.expm1(raw) if log_target else raw, 1.0)
    pd.concat([dur_df, pd.DataFrame([{"model": duration_winner, "split": "test", "mae_seconds": float(mean_absolute_error(durations[te], d_test)), "rmse_seconds": float(np.sqrt(mean_squared_error(durations[te], d_test))), "r2": float(r2_score(durations[te], d_test))}])], ignore_index=True).to_csv(results / "duration.csv", index=False)

    # Preserve the original duration model as an experimental policy. Compare
    # it with the independently selected RMSE baseline without hiding failure.
    log_reg, _ = regressors["Log-random forest"]
    log_rf_test = np.maximum(np.expm1(log_reg.predict(x.iloc[te])), 1.0)
    policy_rows, buffer_rows = [], []
    scenarios = {"baseline": {"revenue":100., "fixed_cost":1., "cost_per_minute":4., "budget_minutes":3000.},
                 "stress": {"revenue":100., "fixed_cost":1., "cost_per_minute":1., "budget_minutes":1500.}}
    for duration_name in dict.fromkeys([duration_winner, "Log-random forest"]):
        pv = valid_probs[winner]
        dv = duration_valid[duration_name] / 60
        dt = (d_test if duration_name == duration_winner else log_rf_test) / 60
        for scenario, params in scenarios.items():
            candidates = []
            for buffer in [.6, .7, .8, .9, 1.0]:
                for policy, ix in bank_policies(pv, dv, **params, buffer=buffer).items():
                    if policy not in ["Probability ranking", "Duration-aware ranking"]:
                        continue
                    row = {"duration_model":duration_name, "scenario":scenario, "policy":policy, "buffer":buffer,
                           **evaluate_bank(ix, pv, dv, y.iloc[va], durations[va]/60, **params)}
                    candidates.append(row)
                    buffer_rows.append(row)
            for policy, ix in bank_policies(p_test, dt, **params).items():
                policy_rows.append({"duration_model":duration_name, "scenario":scenario, "variant":"unbuffered", "policy":policy, "buffer":1., **params,
                    **evaluate_bank(ix, p_test, dt, y.iloc[te], durations[te]/60, **params)})
            for policy in ["Probability ranking", "Duration-aware ranking"]:
                options = [r for r in candidates if r["policy"] == policy and r["actual_within_budget"]]
                # A zero-contact fallback is explicit when no validation candidate is feasible.
                if options:
                    chosen = max(options, key=lambda r:(r["historical_simulated_profit"], -r["buffer"]))
                    buffer = chosen["buffer"]
                    ix = bank_policies(p_test, dt, **params, buffer=buffer)[policy]
                else:
                    buffer, ix = 0.0, np.array([], dtype=int)
                policy_rows.append({"duration_model":duration_name, "scenario":scenario, "variant":"validation-selected buffer", "policy":policy, "buffer":buffer, **params,
                    **evaluate_bank(ix, p_test, dt, y.iloc[te], durations[te]/60, **params)})
    policies = pd.DataFrame(policy_rows)
    policies.to_csv(results / "policies_test.csv", index=False)
    pd.DataFrame(buffer_rows).to_csv(results / "buffer_search_validation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for name, p in valid_probs.items():
        obs, pred = calibration_curve(y.iloc[va], p, n_bins=10, strategy="quantile")
        axes[0].plot(pred, obs, marker="o", ms=3, label=name)
    axes[0].plot([0,1],[0,1],"--", color="#8893a3", lw=1)
    axes[0].set(xlabel="Predicted subscription probability", ylabel="Observed subscription rate", title="Validation probability calibration")
    axes[0].legend(fontsize=7)
    focus = policies[(policies.scenario=="baseline") & (policies.duration_model=="Log-random forest") & (policies.policy=="Duration-aware ranking")]
    axes[1].bar(["Unbuffered", "Validation buffer"], focus.actual_minutes, color=["#d49537", "#245b78"])
    axes[1].axhline(3000, ls="--", color="#ae4b45", label="Actual time budget")
    axes[1].set(ylabel="Actual call minutes on test rows", title="Forecast feasibility vs actual capacity")
    axes[1].legend(fontsize=8)
    fig.savefig(figures / "calibration_and_capacity.svg", metadata={"Date":None})
    plt.close(fig)
    summary = {"environment":environment(), "trees":trees, "split_sizes":{"train":len(tr), "validation":len(va), "test":len(te)}, "feature_columns":features,
               "classification_selection":"minimum validation Brier score", "classifier":winner,
               "duration_selection":"minimum validation RMSE", "duration_model":duration_winner,
               "validation_buffer_grid":[.6,.7,.8,.9,1.], "scenarios":scenarios,
               "elapsed_seconds":round(time.monotonic()-started, 2),
               "evaluation_scope":"Retrospective stratified random holdout; original test rows were previously examined in the course notebook. Not an untouched future campaign or causal evaluation."}
    write_json(results / "run.json", summary)
    print(f"[bank] Complete: classifier={winner}; duration={duration_winner}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    if args.trees < 1:
        parser.error("--trees must be positive")
    run(args.trees)

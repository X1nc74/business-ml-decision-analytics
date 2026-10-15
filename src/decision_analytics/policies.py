"""Selection rules receive predictions only; outcomes are used for scoring."""

import numpy as np
import pandas as pd


def select_prefix(scores, predicted_minutes, budget_minutes, positive=None):
    scores = np.asarray(scores, dtype=float)
    minutes = np.asarray(predicted_minutes, dtype=float)
    if scores.shape != minutes.shape or scores.ndim != 1:
        raise ValueError("Scores and durations must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(minutes)):
        raise ValueError("Scores and durations must be finite")
    if np.any(minutes <= 0) or not np.isfinite(budget_minutes) or budget_minutes < 0:
        raise ValueError("Durations must be positive and budget finite and nonnegative")
    order = np.argsort(-scores, kind="stable")
    if positive is not None:
        positive = np.asarray(positive, dtype=bool)
        if positive.shape != scores.shape:
            raise ValueError("Eligibility mask is not aligned")
        order = order[positive[order]]
    return order[np.cumsum(minutes[order]) <= budget_minutes + 1e-10]


def bank_policies(probabilities, predicted_minutes, revenue, fixed_cost,
                  cost_per_minute, budget_minutes, buffer=1.0, seed=42):
    """Sorted-prefix heuristics under a forecast budget, not exact optimizers."""
    if not 0 < buffer <= 1:
        raise ValueError("Buffer must be in (0, 1]")
    p = np.asarray(probabilities, dtype=float)
    minutes = np.asarray(predicted_minutes, dtype=float)
    expected = revenue * p - fixed_cost - cost_per_minute * minutes
    cap = budget_minutes * buffer
    return {
        "Contact nobody": np.array([], dtype=int),
        "Contact everyone": np.arange(len(p)),
        "Random ranking": select_prefix(np.random.default_rng(seed).random(len(p)), minutes, cap),
        "Probability ranking": select_prefix(p, minutes, cap),
        "Duration-aware ranking": select_prefix(expected / minutes, minutes, cap, expected > 0),
    }


def evaluate_bank(selected, probabilities, predicted_minutes, outcomes,
                  actual_minutes, revenue, fixed_cost, cost_per_minute, budget_minutes):
    ix = np.asarray(selected, dtype=int)
    predicted = float(np.asarray(predicted_minutes)[ix].sum())
    actual = float(np.asarray(actual_minutes)[ix].sum())
    conversions = int(np.asarray(outcomes)[ix].sum())
    return {
        "contacts": len(ix), "subscriptions": conversions,
        "predicted_minutes": predicted, "actual_minutes": actual,
        "projected_profit": float(revenue * np.asarray(probabilities)[ix].sum()
                                  - fixed_cost * len(ix) - cost_per_minute * predicted),
        "historical_simulated_profit": revenue * conversions - fixed_cost * len(ix) - cost_per_minute * actual,
        "predicted_within_budget": predicted <= budget_minutes + 1e-8,
        "actual_within_budget": actual <= budget_minutes + 1e-8,
        "actual_overrun_pct": 100 * max(0.0, actual - budget_minutes) / budget_minutes,
    }

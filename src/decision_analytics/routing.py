"""Deterministic round-trip heuristics using approximate great-circle distance."""

import numpy as np


def distance_matrix(points, depot=(31.1508, 121.4770)):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must be finite latitude/longitude pairs")
    all_points = np.vstack([depot, points])
    if np.any(np.abs(all_points[:, 0]) > 90) or np.any(np.abs(all_points[:, 1]) > 180):
        raise ValueError("Coordinates are out of range")
    r = np.radians(all_points)
    diff = r[:, None, :] - r[None, :, :]
    a = np.sin(diff[:, :, 0] / 2)**2 + np.cos(r[:, None, 0]) * np.cos(r[None, :, 0]) * np.sin(diff[:, :, 1] / 2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def route_distance(route, distances):
    return float(sum(distances[a, b] for a, b in zip(route[:-1], route[1:])))


def build_route(distances, budget_km, candidates=None, ordered=False):
    if not np.isfinite(budget_km) or budget_km < 0:
        raise ValueError("Distance budget must be finite and nonnegative")
    remaining = list(range(1, len(distances))) if candidates is None else list(dict.fromkeys(candidates))
    if any(i < 1 or i >= len(distances) for i in remaining):
        raise ValueError("Candidate indices must exclude the depot and be in range")
    route, travelled = [0], 0.0
    while remaining:
        current = route[-1]
        feasible = [i for i in remaining if travelled + distances[current, i] + distances[i, 0] <= budget_km + 1e-10]
        if not feasible:
            break
        nxt = feasible[0] if ordered else min(feasible, key=lambda i: (distances[current, i], i))
        travelled += distances[current, nxt]
        route.append(nxt)
        remaining.remove(nxt)
    return route + [0]


def route_projection(route, distances, scores, donation_value=200.0,
                     visit_cost=5.0, cost_per_km=10.0):
    ix = np.asarray(route[1:-1], dtype=int) - 1
    km = route_distance(route, distances)
    return {
        "visited": len(ix), "distance_km": km,
        "score_based_projected_net": float(donation_value * np.asarray(scores)[ix].sum() - visit_cost * len(ix) - cost_per_km * km),
    }


def select_threshold(scores, distances, budget_km, thresholds):
    """Select on scores only, before observing the candidate outcomes."""
    rows, routes = [], []
    for threshold in thresholds:
        candidates = np.flatnonzero(np.asarray(scores) >= threshold) + 1
        route = build_route(distances, budget_km, candidates)
        rows.append({"threshold": float(threshold), **route_projection(route, distances, scores)})
        routes.append(route)
    if not rows:
        raise ValueError("At least one threshold is required")
    best = max(range(len(rows)), key=lambda i: rows[i]["score_based_projected_net"])
    return rows[best]["threshold"], routes[best], rows

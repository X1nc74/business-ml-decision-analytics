# Validation record

Recorded during portfolio curation on 2026-10-15 UTC (2026-10-14 in New York).

## Executed locally

| Check | Outcome |
| --- | --- |
| Full bank experiment, default 300-tree configuration | Completed; new model, duration, policy, and buffer-search exports generated |
| Full donor experiment, 200-tree candidate models | Completed; new selection, subgroup, route, and sensitivity exports generated |
| Editable package installation | Passed using the already-installed exact dependencies, with `--no-deps --no-build-isolation` |
| `python -m unittest discover -s tests -v` | 11 tests passed |
| `python scripts/verify_repository.py` | 6 verification groups passed |
| Walkthrough table cells | Executed against the generated CSV files; saved tables embedded in both notebooks |
| Python source compilation | Passed for `src`, `scripts`, and `tests` |
| Three SVG figures | Rendered to raster previews and visually inspected |

Per-case `results/run.json` records actual package versions, parameters, and experiment scope. Tests cover forecast versus actual capacity, budget boundaries, score-only policy selection, nonpositive expected-profit filtering, training-only preprocessing, unseen categories, and feasible depot returns. The integrity script verifies data/source checksums, split disjointness, monetary arithmetic, budget flags, and validation-derived buffer choices.

## Scope of the automation

The included GitHub Actions workflow runs unit tests and repository checks. A local pass does not establish that a remote workflow has passed. CI does not retrain the full models. A clean internet-based dependency installation and a notebook-server UI session were not separately tested.

The hotel, inventory, and causal-targeting notebooks were source-reviewed rather than reproduced. Their required repairs and input gaps are documented in [CASE_REVIEW.md](CASE_REVIEW.md).

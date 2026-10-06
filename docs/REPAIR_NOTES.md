# Repository audit and repair — 2026-10-05

Baseline: `1ddda52`. Original source and notebook experiments remain available in Git history.

## Findings and changes

| Area | Original issue | Change |
|---|---|---|
| Installation | Missing `m5_wrmsse`; heavy unrelated dependencies; inconsistent Python metadata | Local evaluator; small core and separate extras; Python >=3.10 |
| Training/inference | Shifted training features but unshifted test features derived from observed holdout sales | Same horizon transformation, mask all future targets before features |
| Features | Rolling operation crossed groups; incorrect M5 weekend mapping | Group-local rolling; weekend from actual date |
| Evaluation | Hidden evaluator inputs, silently zero-filled missing IDs | Explicit full-history inputs, 12 hierarchy levels, pre-cutoff revenue weights, strict alignment |
| Data | Extra history day; relative paths tied to working directory | Exact history window, daily calendar validation, YAML-relative paths |
| Artifacts | Standard and MLflow summaries had different keys; models not saved | One shared pipeline; CSV/JSON and LightGBM text artifacts |
| Serving | Unknown IDs became zeros; fixed training date; missing artifacts crashed startup | 404/422/503 responses, actual metadata, explicit stored-backtest contract |
| Docker | Copied absent ignored outputs; health check required missing curl | Read-only artifact mount, stdlib HTTP check, libgomp runtime |
| Download | Ignored Kaggle exit status | Checked subprocess and safe extraction |
| Notebooks 01–03 | Large hidden intermediate pickles and brittle sequencing | Independent raw-data EDA, baselines, demand clustering |
| Notebook 04 | Inconsistent scaling, context included evaluation observations, missing optional package | Replaced with explicitly different univariate zero-shot TTM benchmark |
| Notebook 05 | Wrong cluster path; mislabeled subgroup MAE as WRMSSE | Standalone TSB and common correctly aligned evaluation |
| Notebooks 06–08 | Inconsistent test dates, holdout model selection and/or future target leakage | Shared tested pipeline; pooled, direct-bottom and direct-hierarchical variants |
| Notebook 09 | Off-by-one forecast origin and silent incomplete-forecast padding | Explicit future timestamps, known calendar covariates, checked alignment |
| Notebook 10 | Calibration errors from naive model applied to LightGBM; wrong absolute-error quantile | Earlier same-model fold, finite-sample quantile, separate final holdout |
| Portfolio claims | Unverified leaderboard rank and production-ready label | Removed; scope and remaining validation are explicit |

Notebook restructuring is substantial: reusable computation moved into modules and notebooks were rewritten as smaller independent experiments. Old plotting/analysis cells are retained through Git history, not asserted to be repaired. The duplicate `09_conformal_prediction.ipynb` was removed in favor of notebook 10. TTM's former multivariate fine-tuning procedure has **not** been reproduced by the new zero-shot benchmark.

## Verification

- Editable install of core, API and development dependencies succeeded on Python 3.12.
- `python -m pytest -q`: **11 passed**, including full-pipeline holdout-target invariance for both direct and shared models, group isolation, manually checked WRMSSE examples, an independent multi-level reference calculation, API errors, and notebook validity/syntax.
- Synthetic demo: 16 series, 160 days, 28 forecast horizons; all horizon model artifacts written. Five-round demo LightGBM WRMSSE approximately 0.7326; weekly seasonal-naive approximately 0.7042. These are synthetic integration checks, not competition scores.
- All code cells in notebooks **01, 02, 03, 05, 06, 07, 08 and 10** executed in order against synthetic data with headless plotting. Execution used Python cell execution with a display stub; it did not validate interactive Jupyter rendering. The default 200-round notebook configurations were exercised, including both conformal folds.
- MLflow demo entry point completed successfully with MLflow 3.16.1 and a local SQLite tracking store; metrics, parameters and CSV/JSON artifacts were recorded.
- Python compilation and `git diff --check` passed.
- CI added for Python 3.10, 3.11 and 3.12; remote CI status should be inspected on the pull request.

## Remaining validation

The full licensed Kaggle dataset and TTM/Chronos model weights were not available locally. Full M5 training, leaderboard comparisons, foundation-model weight loading/execution and performance are therefore **unverified**. Docker is not installed in this environment, so the image was reviewed but not built or run. The API serves artifacts from a historical holdout, not live future inference. Interval coverage is empirical under temporal dependence.

## Memory follow-up — 2026-10-06

A real full-data attempt failed while pandas copied six object identifier columns in a hierarchical merge: 23,111,420 rows, requesting another 1.03 GiB allocation. The tiny initial demo had not exposed this scaling problem.

The follow-up changes:

- Read only the requested wide sales columns; construct the long panel directly in id/date order, with categorical identifiers and float32 sales.
- Replace price and hierarchy-wide merges with bounded 250,000-row key lookups. Use observed category combinations only, and float32 lag/rolling features.
- Avoid duplicate train/test panels, repeated full-table sorts, and full shifted feature matrices. Gather the needed target/source rows into one float32 training matrix; prediction gathers only the forecast-day rows.
- Free the feature table before evaluation, and compute WRMSSE scales in 1,024-series blocks. Evaluation still uses the complete pre-cutoff history and unchanged hierarchy/weight definitions.
- Add `config/low_memory.yaml`: 180 training days, at most 500,000 deterministically sampled candidate training rows per horizon, 63 histogram bins, and 128 MiB histogram cache. All series and all 28 holdout days remain in evaluation. The sampling/history/bin changes affect model fit and may affect accuracy; the histogram setting is not a process RAM limit.
- Add progress messages and report the sample cap and actual training rows in JSON summaries.

Verification: **14 tests passed**, including independent merge/group-shift equivalence with unused categories and missing keys, and 28-horizon sampled-training repeatability and holdout-target invariance.

A separate Linux synthetic stress test used **30,490 series x 758 days = 23,111,420 rows**, all four hierarchy feature levels, and all 34 model features. Feature construction completed with a 3,176.7 MiB stored panel. The test then trained **one shared horizon-28 model, one boosting round, from at most 500,000 sampled candidates** and produced a finite 30,490 x 28 prediction table. Peak process RSS was **3,798.9 MiB**, elapsed approximately **34 seconds** in this environment. These measurements cover a synthetic feature/training/prediction stress test, not CSV loading, the complete evaluator, a 28-model/200-round M5 run, or Windows. They are not a RAM guarantee or an accuracy result for the real dataset.

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

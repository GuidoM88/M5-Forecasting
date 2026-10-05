# M5 retail demand forecasting

A reproducible forecasting portfolio project: fixed-origin backtesting, direct LightGBM models, hierarchical sales features, twelve-level WRMSSE evaluation, and a FastAPI service for saved forecasts.

**Status:** the core pipeline and API run on deterministic synthetic data and have regression tests. Full M5 training and the optional foundation-model experiments still require the dataset/model downloads and a local run. The historical **0.6140 / top 2.7%** claim has been withdrawn: the earlier implementation used holdout sales in prediction features and inconsistent horizon alignment. No corrected competition result is claimed.

## Quick start: no Kaggle credentials needed

Use Python 3.10–3.12. From the repository root:

```bash
git clone https://github.com/GuidoM88/M5-Forecasting.git
cd M5-Forecasting
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[api,dev]"
python -m scripts.make_demo_data
python -m scripts.train_hierarchical_lgbm --config config/demo.yaml
python -m pytest -q
```

The demo creates 16 synthetic product/store series, trains 28 horizon models, evaluates against a weekly seasonal-naive baseline, and writes `outputs/demo/forecasts.csv`, `summary.json`, `config.yaml` and models under `models/demo/`. It proves that the plumbing executes; its scores are **not M5 performance evidence**. The tiny five-round model need not outperform the baseline. The generator refuses to overwrite existing CSVs.

Serve the demo:

```bash
# Linux/macOS
export M5_OUTPUT_DIR=outputs/demo
# Windows PowerShell: $env:M5_OUTPUT_DIR="outputs/demo"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000/docs. Query `/items` to obtain real IDs. Example request to `POST /predict`:

```json
{"item_ids":["FOODS_1_001_CA_1_evaluation"]}
```

The response includes the cutoff and forecast dates. This API **serves stored backtest forecasts**; it does not retrain a model or generate live future forecasts. Unknown IDs return 404, invalid requests 422, and missing artifacts 503. Restart the API after replacing artifacts.

## Run on M5

Accept the [Kaggle competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), configure your Kaggle credentials, then:

```bash
python -m pip install -e ".[download]"
python -m src.data.download
python -m scripts.train_hierarchical_lgbm --config config/hierarchical_lgbm.yaml
```

Alternatively place `calendar.csv`, `sell_prices.csv`, and `sales_train_evaluation.csv` in `data/raw/`. This last file contains 1,941 observed days, not 1,969. The final 28 observed days (`d_1914`–`d_1941` for the official file) are held out. The hidden competition horizon is a different task; this pipeline does not produce a Kaggle submission.

Configuration paths are resolved relative to the YAML file, independently of the working directory. With the default 730-day feature history, full M5 creates tens of millions of rows. Expect substantial RAM and training time; begin with the demo. No full-data memory or timing claim has been measured here.

## Forecasting design

- One model per horizon `h=1..28`; forecasts do not feed back recursively.
- Calendar and price features correspond to the target date. **Future selling prices are assumed known**; results are conditional on that assumption.
- For target date `t`, lag `l` uses sales at `t-h+1-l`, and a rolling window ends at `t-h`. Training and prediction use this identical transformation. Future sales are masked before any feature construction.
- Features include bottom-series lags/rolling means and group-average histories across items, department/store and state/store. These are hierarchical covariates, not a reconciliation procedure. All rolling operations stay within their group.
- LightGBM retains missing prices as NaN. No backward filling from future prices, no silent zero forecasts for missing series, and no holdout early stopping.
- The shared-model notebook uses one pooled model and features shifted by the full horizon, making every prediction safe at the fixed origin.

WRMSSE sums series at all twelve M5 hierarchy levels. Scale uses the complete pre-holdout history after the first nonzero sale; dollar weights use the last 28 **training** days. Levels receive equal weight. Invalid IDs, missing positive-sale prices and undefined positive-weight zero scales raise errors. The default model can train on a shorter window while the evaluator uses the complete available history.

Metric reference: [M5 accuracy competition: results, findings, and conclusions](https://doi.org/10.1016/j.ijforecast.2021.11.013).

## Notebooks

Install `python -m pip install -e ".[notebooks]"`, use that environment's kernel, then open `notebooks/`. Each notebook loads raw inputs directly; there are no required hidden pickle artifacts. Set `M5_RAW_DIR` to an absolute data directory to change the dataset. The original lengthy prototypes remain in Git history; the maintained notebooks delegate reusable logic to tested modules.

| Notebook | Purpose |
|---|---|
| 01 | Training-only EDA |
| 02 | Naive, weekly seasonal-naive, historical-mean benchmarks |
| 03 | Training-only demand descriptors and clustering |
| 04 | Optional univariate zero-shot TTM benchmark |
| 05 | TSB intermittent-demand benchmark |
| 06 | One pooled LightGBM with full-horizon-safe features |
| 07 | Direct horizon models, bottom-level features |
| 08 | Direct horizon models with hierarchical features |
| 09 | Optional Chronos-2 with calendar covariates |
| 10 | Same-model earlier calibration fold and final holdout intervals |

The duplicate conformal notebook was removed. Notebook 10 fits the same modeling procedure on an earlier fold to obtain calibration residuals, then refits for the final holdout. Intervals use the finite-sample absolute-error quantile and report empirical coverage. Pooled time-series residuals do not provide an unconditional exchangeability guarantee.

Foundation models need separate optional environments and internet access to download weights. Follow [IBM granite-tsfm](https://github.com/ibm-granite/granite-tsfm) for `granite-tsfm`, and [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting) for `chronos-forecasting>=2.0`. These dependencies are deliberately excluded from the core install. The TTM notebook now demonstrates univariate zero-shot inference; the former multivariate fine-tuning experiment is not claimed reproduced. Both foundation notebooks default to a labeled 64-series subset and **have not been executed with real weights during this repair**.

## MLflow, API and Docker

```bash
python -m pip install -e ".[tracking]"
python -m scripts.train_hierarchical_lgbm_mlflow --config config/demo.yaml
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Both CLI entry points call the same pipeline. Models are saved as LightGBM text files, forecasts as CSV, and metadata as JSON. No serving-time pickle loading is required.

For full-data API serving, the default output directory is `outputs/forecasts`. See [README_DOCKER.md](README_DOCKER.md) for Docker, artifact mounts and readiness checks.

## Validation and limitations

Regression tests cover origin/horizon alignment, isolation of rolling groups, holdout-target invariance, WRMSSE arithmetic and ID alignment, configuration consistency, API errors and saved artifacts. GitHub Actions runs tests and the synthetic demo on Python 3.10–3.12.

Full-data accuracy, ranking, real-weight TTM/Chronos execution, and Docker runtime behavior need separate validation. Model text files are saved for inspection/reuse, but the API's contract is artifact retrieval. Multi-origin model selection, production monitoring, authentication and live future inference are outside the current implementation.

See [docs/REPAIR_NOTES.md](docs/REPAIR_NOTES.md) for the audit and exact verification performed.

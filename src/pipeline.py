"""Shared holdout pipeline for CLI, notebooks and optional MLflow tracking."""
import gc
import json
import time
from datetime import datetime, timezone
import pandas as pd
from src.config import Config
from src.data_loader import M5DataLoader
from src.evaluation import M5Evaluator
from src.feature_engineering import FeatureEngineer
from src.model import HierarchicalLGBM


def run_pipeline(config_path='config/hierarchical_lgbm.yaml'):
    config = Config(config_path)
    start = time.perf_counter()
    loader = M5DataLoader(config.raw_data_path, config.history_days, config.test_horizon)
    data = loader.load_data()
    cutoff = data.date.max() - pd.Timedelta(days=config.test_horizon)
    forecast_dates = sorted(data.loc[data.date > cutoff, 'date'].unique())
    actual_training_days = int(data.loc[data.date <= cutoff, 'date'].nunique())
    # The baseline needs only the final week, not a second full training panel.
    recent = data.loc[(data.date <= cutoff) & (data.date > cutoff-pd.Timedelta(days=7)),
                      ['id','date','sales']]
    history = recent.pivot(index='id', columns='date', values='sales').sort_index(axis=1)
    baseline = pd.DataFrame({f'F{h}':history.iloc[:,-7+(h-1)%7]
                             for h in range(1,config.test_horizon+1)})
    del recent, history
    # Never expose holdout targets to feature construction.
    data.loc[data.date > cutoff, 'sales'] = float('nan')
    fe = FeatureEngineer(config.lags, config.rolling_windows)
    featured = fe.create_all_features(data, config.hierarchical_levels)
    del data
    gc.collect()
    print(f"Feature panel: {featured.memory_usage(deep=True).sum()/2**20:.0f} MiB", flush=True)
    names = config.base_features + [c for c in fe.get_feature_names(config.hierarchical_levels)
                                    if '_lag_' in c or '_roll_' in c]
    model = HierarchicalLGBM(config.model_params, config.num_boost_round, config.num_models,
                             shared_model=config.get('model.shared_model', False),
                             max_train_rows=config.get('model.training.max_train_rows'))
    train_start = time.perf_counter()
    model.train(featured, names, cutoff=cutoff)
    training_time = time.perf_counter() - train_start
    forecast = model.predict(featured, names, cutoff)
    del featured
    gc.collect()
    print('Evaluating on complete pre-cutoff history...', flush=True)
    raw = config.raw_data_path
    header = pd.read_csv(raw / 'sales_train_evaluation.csv', nrows=0)
    evaluator = M5Evaluator(pd.read_csv(raw / 'sales_train_evaluation.csv',
                            dtype={c:'float32' for c in header if c.startswith('d_')}),
                            pd.read_csv(raw / 'calendar.csv'),
                            pd.read_csv(raw / 'sell_prices.csv'), config.test_horizon)
    score = evaluator.evaluate(forecast)
    baseline_score = evaluator.evaluate(baseline)
    summary = dict(wrmsse=score, seasonal_naive_wrmsse=baseline_score,
                   training_date=datetime.now(timezone.utc).isoformat(),
                   cutoff=str(cutoff.date()), forecast_dates=[str(d.date()) for d in map(pd.Timestamp, forecast_dates)],
                   history_days=config.history_days, lags=config.lags,
                   rolling_windows=config.rolling_windows, num_boost_round=config.num_boost_round,
                   training_time=training_time, pipeline_time=time.perf_counter()-start, num_features=len(names),
                   actual_training_days=actual_training_days,
                   max_train_rows=model.max_train_rows, training_rows_by_horizon=model.training_rows,
                   horizon=config.test_horizon, n_series=len(forecast),
                   mode='holdout_backtest', price_assumption='target-day prices known in advance')
    config.output_path.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(config.output_path / 'forecasts.csv')
    (config.output_path / 'summary.json').write_text(json.dumps(summary, indent=2))
    (config.output_path / 'config.yaml').write_text(config.config_path.read_text())
    if config.get('output.save_models', True):
        model.save(config.models_path)
    print(json.dumps(summary, indent=2))
    return forecast, summary

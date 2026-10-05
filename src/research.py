"""Common aligned inputs and evaluation for the research notebooks."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation import M5Evaluator


def load_panel(raw_dir, horizon=28):
    raw_dir = Path(raw_dir)
    sales = pd.read_csv(raw_dir / 'sales_train_evaluation.csv')
    calendar = pd.read_csv(raw_dir / 'calendar.csv')
    prices = pd.read_csv(raw_dir / 'sell_prices.csv')
    days = sorted([c for c in sales if c.startswith('d_')], key=lambda c: int(c[2:]))
    history = sales.set_index('id')[days[:-horizon]].astype(float)
    actual = sales.set_index('id')[days[-horizon:]].astype(float)
    actual.columns = [f'F{h}' for h in range(1, horizon+1)]
    return sales, calendar, prices, history, actual


def baseline_forecasts(history, horizon=28):
    if history.shape[1] < 7:
        raise ValueError('At least seven historical days required')
    return {
        'naive': pd.DataFrame({f'F{h}':history.iloc[:,-1] for h in range(1,horizon+1)}),
        'seasonal_naive': pd.DataFrame({f'F{h}':history.iloc[:,-7+(h-1)%7] for h in range(1,horizon+1)}),
        'historical_mean': pd.DataFrame({f'F{h}':history.mean(axis=1) for h in range(1,horizon+1)}),
    }


def tsb_forecast(history, horizon=28, alpha_d=0.2, alpha_p=0.2):
    """TSB: smooth demand occurrence each day, size on positive-demand days."""
    values = history.to_numpy(dtype=float)
    probability = np.zeros(len(values))
    size = np.zeros(len(values))
    seen = np.zeros(len(values), dtype=bool)
    for day in values.T:
        positive = day > 0
        first = positive & ~seen
        size[first] = day[first]
        continuing = positive & seen
        size[continuing] += alpha_d * (day[continuing] - size[continuing])
        probability += alpha_p * (positive - probability)
        seen |= positive
    return pd.DataFrame({f'F{h}':size*probability for h in range(1,horizon+1)}, index=history.index)


def score_forecasts(sales, calendar, prices, forecasts):
    evaluator = M5Evaluator(sales, calendar, prices, len(next(iter(forecasts.values())).columns))
    return pd.Series({name:evaluator.evaluate(pred) for name,pred in forecasts.items()}, name='WRMSSE')


def conformal_radius(actual, prediction, coverage=0.9):
    """Finite-sample absolute-error quantile; no independence guarantee for time series."""
    if not 0 < coverage < 1:
        raise ValueError('coverage must be between zero and one')
    if set(actual.index) != set(prediction.index) or list(actual.columns) != list(prediction.columns):
        raise ValueError('Calibration IDs and horizons must match')
    residuals = np.abs(actual.reindex(prediction.index).to_numpy() - prediction.to_numpy()).ravel()
    if not np.isfinite(residuals).all():
        raise ValueError('Calibration residuals must be finite')
    rank = int(np.ceil((len(residuals)+1)*coverage))
    if rank > len(residuals):
        raise ValueError('Insufficient calibration observations for requested coverage')
    return float(np.partition(residuals, rank-1)[rank-1])

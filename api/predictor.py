"""Serve validated, precomputed backtest forecasts (no online inference)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


class M5Predictor:
    def __init__(self, forecasts_path, summary_path):
        self.forecasts_path = Path(forecasts_path)
        self.summary_path = Path(summary_path)
        self.model_loaded = False

    def load_model(self):
        self.model_loaded = False
        forecasts = pd.read_csv(self.forecasts_path, index_col='id')
        summary = json.loads(self.summary_path.read_text())
        expected = [f'F{i}' for i in range(1, summary['horizon'] + 1)]
        if (forecasts.empty or forecasts.index.has_duplicates or forecasts.index.isna().any()
                or list(forecasts.columns) != expected or len(forecasts) != summary['n_series']
                or not np.isfinite(forecasts.to_numpy()).all() or (forecasts < 0).any().any()):
            raise ValueError('Invalid forecast artifact')
        self.forecasts_df, self.summary = forecasts, summary
        self.model_loaded = True

    def predict(self, item_ids):
        if not self.model_loaded:
            raise RuntimeError('Forecast artifacts are not loaded')
        missing = sorted(set(item_ids) - set(self.forecasts_df.index))
        if missing:
            raise KeyError(f'Unknown item IDs: {missing}')
        return {item: self.forecasts_df.loc[item].tolist() for item in item_ids}

    def get_available_items(self):
        if not self.model_loaded:
            raise RuntimeError('Forecast artifacts are not loaded')
        return self.forecasts_df.index.tolist()

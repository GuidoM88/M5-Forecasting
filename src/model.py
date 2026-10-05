"""Direct horizon models with the same information cutoff at fit and predict time."""
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd


class HierarchicalLGBM:
    def __init__(self, params, num_boost_round, num_models=28, shared_model=False):
        self.params = dict(params)
        self.num_boost_round = num_boost_round
        self.num_models = num_models
        self.models = {}
        self.shared_model = shared_model
        self.feature_names = []

    @staticmethod
    def horizon_features(frame, feature_names, horizon):
        """At target t, sales features refer to origin t-h (never later)."""
        frame = frame.sort_values(['id', 'date'])
        X = frame[feature_names].copy()
        dynamic = [c for c in feature_names if '_lag_' in c or '_roll_' in c]
        X[dynamic] = frame.groupby('id', sort=False)[dynamic].shift(horizon - 1)
        return X

    def train(self, train_df, feature_names):
        self.models = {}
        self.feature_names = list(feature_names)
        train_df = train_df.sort_values(['id', 'date'])
        for h in ([self.num_models] if self.shared_model else range(1, self.num_models + 1)):
            X = self.horizon_features(train_df, feature_names, h)
            # Keep missing prices: LightGBM handles NaN. Require usable sales history.
            dynamic = [c for c in feature_names if '_lag_' in c or '_roll_' in c]
            valid = X[dynamic].notna().all(axis=1) & train_df.sales.notna()
            if not valid.any():
                raise ValueError(f'Insufficient training history for horizon {h}')
            self.models[h] = lgb.train(
                self.params, lgb.Dataset(X.loc[valid], label=train_df.loc[valid, 'sales']),
                num_boost_round=self.num_boost_round,
                callbacks=[lgb.log_evaluation(0)],
            )

        if self.shared_model:
            self.models = {h: self.models[self.num_models] for h in range(1, self.num_models + 1)}

    def predict(self, data, feature_names, cutoff):
        """data includes historical features and masked future rows, not just test."""
        if list(feature_names) != self.feature_names or len(self.models) != self.num_models:
            raise ValueError('Model is not trained or feature schema differs')
        data = data[data.date >= pd.Timestamp(cutoff) - pd.Timedelta(days=self.num_models)].sort_values(['id', 'date'])
        ids = pd.Index(sorted(data.id.unique()), name='id')
        result = pd.DataFrame(index=ids)
        for h in range(1, self.num_models + 1):
            day = pd.Timestamp(cutoff) + pd.Timedelta(days=h)
            mask = data.date.eq(day)
            rows = data.loc[mask]
            if len(rows) != len(ids) or rows.id.duplicated().any():
                raise ValueError(f'Missing or duplicate series on {day.date()}')
            X = self.horizon_features(data, feature_names, self.num_models if self.shared_model else h).loc[mask]
            result[f'F{h}'] = pd.Series(
                np.maximum(self.models[h].predict(X), 0), index=rows.id
            ).reindex(ids)
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for h, model in self.models.items():
            model.save_model(str(directory / f'horizon_{h}.txt'))
        (directory / 'features.json').write_text(json.dumps(self.feature_names))

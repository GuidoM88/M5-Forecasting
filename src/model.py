"""Direct-horizon models, building only the selected float32 training matrix."""
import gc
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.memory import order_panel


class HierarchicalLGBM:
    def __init__(self, params, num_boost_round, num_models=28, shared_model=False,
                 max_train_rows=None):
        self.params = dict(params)
        self.num_boost_round = num_boost_round
        self.num_models = num_models
        self.models = {}
        self.shared_model = shared_model
        self.feature_names = []
        self.max_train_rows = max_train_rows
        self.training_rows = {}
        if max_train_rows is not None and (not isinstance(max_train_rows, int) or max_train_rows < 1):
            raise ValueError('max_train_rows must be a positive integer or null')

    @staticmethod
    def _matrix(frame, feature_names, horizon, positions):
        """Gather rows by position, checking that shifted sources stay in the same series."""
        positions = np.asarray(positions, dtype=np.int64)
        sources = positions - (horizon - 1)
        safe_sources = np.maximum(sources, 0)
        ids = (frame.id.cat.codes.to_numpy() if isinstance(frame.id.dtype, pd.CategoricalDtype)
               else frame.id.to_numpy())
        same = (sources >= 0) & (ids[safe_sources] == ids[positions])
        X = np.empty((len(positions), len(feature_names)), dtype=np.float32)
        for j, col in enumerate(feature_names):
            dynamic = '_lag_' in col or '_roll_' in col
            X[:,j] = frame[col].to_numpy()[safe_sources if dynamic else positions]
            if dynamic:
                X[~same,j] = np.nan
        return X

    @staticmethod
    def horizon_features(frame, feature_names, horizon):
        frame = order_panel(frame)
        return pd.DataFrame(HierarchicalLGBM._matrix(frame, feature_names, horizon, np.arange(len(frame))),
                            index=frame.index, columns=feature_names)

    def train(self, train_df, feature_names, cutoff=None):
        self.models = {}
        self.training_rows = {}
        self.feature_names = list(feature_names)
        train_df = order_panel(train_df)
        eligible = train_df.sales.notna().to_numpy()
        if cutoff is not None:
            eligible &= train_df.date.le(cutoff).to_numpy()
        positions = np.flatnonzero(eligible)
        del eligible
        if self.max_train_rows is not None and len(positions) > self.max_train_rows:
            rng = np.random.default_rng(self.params.get('seed', 42))
            positions = np.sort(rng.choice(positions, self.max_train_rows, replace=False))
        dynamic = [i for i,c in enumerate(feature_names) if '_lag_' in c or '_roll_' in c]
        for h in ([self.num_models] if self.shared_model else range(1,self.num_models+1)):
            print(f'Training horizon {h}/{self.num_models} (at most {len(positions):,} rows)', flush=True)
            X = self._matrix(train_df, feature_names, h, positions)
            valid = np.ones(len(positions), dtype=bool)
            for j in dynamic:
                valid &= np.isfinite(X[:,j])
            if not valid.any():
                raise ValueError(f'Insufficient training history for horizon {h}')
            y = train_df.sales.to_numpy()[positions[valid]]
            X = X[valid]
            self.training_rows[h] = len(y)
            dataset = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=True)
            self.models[h] = lgb.train(self.params, dataset, num_boost_round=self.num_boost_round,
                                       callbacks=[lgb.log_evaluation(0)])
            del X, y, dataset, valid
            gc.collect()
        if self.shared_model:
            self.models = {h:self.models[self.num_models] for h in range(1,self.num_models+1)}

    def predict(self, data, feature_names, cutoff):
        if list(feature_names) != self.feature_names or len(self.models) != self.num_models:
            raise ValueError('Model is not trained or feature schema differs')
        data = order_panel(data)
        ids = pd.Index(sorted(data.id.unique()), name='id')
        result = pd.DataFrame(index=ids)
        for h in range(1,self.num_models+1):
            day = pd.Timestamp(cutoff) + pd.Timedelta(days=h)
            positions = np.flatnonzero(data.date.eq(day).to_numpy())
            row_ids = data.id.iloc[positions]
            if len(positions) != len(ids) or row_ids.duplicated().any():
                raise ValueError(f'Missing or duplicate series on {day.date()}')
            X = self._matrix(data, feature_names, self.num_models if self.shared_model else h, positions)
            result[f'F{h}'] = pd.Series(np.maximum(self.models[h].predict(X),0), index=row_ids).reindex(ids)
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for h,model in self.models.items():
            model.save_model(str(directory/f'horizon_{h}.txt'))
        (directory/'features.json').write_text(json.dumps(self.feature_names))

"""Group-local, float32 hierarchical features without wide-table merges."""
import gc
import numpy as np
import pandas as pd
from src.memory import compact_panel, order_panel, add_lookup_columns


class FeatureEngineer:
    def __init__(self, lags, rolling_windows):
        self.lags = lags
        self.rolling_windows = rolling_windows

    def create_all_features(self, df, hierarchical_levels):
        if 'sales' not in df:
            raise ValueError("Column 'sales' required")
        df = order_panel(compact_panel(df))
        for col in ['state_id', 'store_id', 'dept_id', 'item_id']:
            if col in df:
                df[f'{col}_enc'] = pd.factorize(df[col], sort=True)[0].astype('int32')
        df['is_weekend'] = df.date.dt.dayofweek.isin([5, 6]).astype('int8')
        for level in hierarchical_levels:
            keys, prefix = level['groupby'], level['prefix']
            print(f"Features: {prefix} ({len(df):,} rows)", flush=True)
            if keys == ['id']:
                df = self._add_lag_rolling_features(df, keys, 'sales', prefix, presorted=True)
            else:
                agg = self._create_aggregated_sales(df, keys, prefix)
                agg = self._add_lag_rolling_features(agg, keys, f'sales_{prefix}', prefix)
                columns = [c for c in agg if c.startswith(f'{prefix}_')]
                add_lookup_columns(df, agg, keys + ['date'], columns)
                del agg
                gc.collect()
        return df

    def _create_aggregated_sales(self, df, groupby_cols, prefix):
        # observed=True avoids a Cartesian product of unused categorical combinations.
        return (df.groupby(groupby_cols + ['date'], as_index=False, observed=True, sort=False)['sales']
                .mean().rename(columns={'sales': f'sales_{prefix}'}))

    def _add_lag_rolling_features(self, df, groupby_cols, value_col, prefix, presorted=False):
        if not presorted:
            df = df.sort_values(groupby_cols + ['date'])
        grp = df.groupby(groupby_cols, sort=False, observed=True)[value_col]
        for lag in self.lags:
            df[f'{prefix}_lag_{lag}'] = grp.shift(lag).astype('float32')
        for window in self.rolling_windows:
            df[f'{prefix}_roll_{window}'] = grp.transform(
                lambda s: s.shift(1).rolling(window).mean().astype('float32'))
        return df

    def get_feature_names(self, hierarchical_levels):
        base = ['wday', 'month', 'year', 'is_weekend', 'snap', 'sell_price',
                'state_id_enc', 'store_id_enc', 'dept_id_enc', 'item_id_enc']
        return base + [f"{v['prefix']}_lag_{lag}" for v in hierarchical_levels for lag in self.lags] + [
            f"{v['prefix']}_roll_{window}" for v in hierarchical_levels for window in self.rolling_windows]

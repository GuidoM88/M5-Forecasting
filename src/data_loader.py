"""Load only the requested sales window and expand compact categorical identifiers."""
from pathlib import Path
import gc
import numpy as np
import pandas as pd
from src.memory import ID_COLUMNS, compact_panel, add_lookup_columns


class M5DataLoader:
    def __init__(self, raw_dir, history_days, test_horizon=28):
        self.raw_dir = Path(raw_dir)
        self.history_days = history_days
        self.test_horizon = test_horizon

    def load_data(self):
        print('Loading raw files...', flush=True)
        path = self.raw_dir / 'sales_train_evaluation.csv'
        all_days = sorted([c for c in pd.read_csv(path, nrows=0) if c.startswith('d_')],
                          key=lambda c: int(c[2:]))
        days = all_days[-(self.history_days + self.test_horizon):]
        sales = pd.read_csv(path, usecols=ID_COLUMNS + days,
                            dtype={**{c:'category' for c in ID_COLUMNS}, **{c:'float32' for c in days}})
        sales = sales.sort_values('id').reset_index(drop=True)
        if sales.id.duplicated().any():
            raise ValueError('Duplicate sales IDs')
        calendar = pd.read_csv(self.raw_dir / 'calendar.csv').set_index('d').loc[days]
        dates = pd.to_datetime(calendar.date)
        if dates.isna().any() or not dates.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError('Sales calendar must be daily and contiguous')
        n_series, n_days = len(sales), len(days)
        # Build directly in id/date order; melt and a later full-table sort are unnecessary.
        long = pd.DataFrame({c:pd.Categorical.from_codes(
            np.repeat(sales[c].cat.codes.to_numpy(), n_days), categories=sales[c].cat.categories)
            for c in ID_COLUMNS})
        long['date'] = np.tile(dates.to_numpy(), n_series)
        long['sales'] = sales[days].to_numpy(dtype=np.float32).ravel()
        if not np.isfinite(long.sales).all() or (long.sales < 0).any():
            raise ValueError('Sales must be finite and nonnegative')
        del sales
        for col in ['wm_yr_wk', 'wday', 'month', 'year']:
            long[col] = np.tile(pd.to_numeric(calendar[col], downcast='integer').to_numpy(), n_series)
        long['snap'] = np.zeros(len(long), dtype=np.int8)
        for state in ['CA','TX','WI']:
            mask = long.state_id.eq(state).to_numpy()
            values = np.tile(calendar[f'snap_{state}'].to_numpy(dtype=np.int8), n_series)
            long.loc[mask,'snap'] = values[mask]
        prices = pd.read_csv(self.raw_dir / 'sell_prices.csv',
                             dtype={'store_id':'category', 'item_id':'category', 'sell_price':'float32'})
        add_lookup_columns(long, prices, ['store_id','item_id','wm_yr_wk'], ['sell_price'])
        del prices, calendar
        long.drop(columns='wm_yr_wk', inplace=True)
        gc.collect()
        long = compact_panel(long)
        print(f'Loaded {len(long):,} rows; panel {long.memory_usage(deep=True).sum()/2**20:.0f} MiB', flush=True)
        return long

    def split_train_test(self, df):
        cutoff = df.date.max() - pd.Timedelta(days=self.test_horizon)
        return df[df.date <= cutoff].copy(), df[df.date > cutoff].copy()

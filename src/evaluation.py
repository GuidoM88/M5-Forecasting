"""M5's twelve-level dollar-weighted RMSSE, using pre-cutoff data only."""
import numpy as np
import pandas as pd

LEVELS = [[], ['state_id'], ['store_id'], ['cat_id'], ['dept_id'],
          ['state_id', 'cat_id'], ['state_id', 'dept_id'],
          ['store_id', 'cat_id'], ['store_id', 'dept_id'], ['item_id'],
          ['item_id', 'state_id'], ['id']]


class M5Evaluator:
    def __init__(self, sales_df, calendar, prices, horizon=28):
        self.sales = sales_df.reset_index(drop=True)
        self.ids = pd.Index(self.sales.id)
        if self.ids.has_duplicates:
            raise ValueError('Duplicate sales IDs')
        days = sorted([c for c in sales_df if c.startswith('d_')], key=lambda c: int(c[2:]))
        if len(days) <= horizon + 28:
            raise ValueError('Need training history, 28 weight days, and holdout')
        self.horizon = horizon
        self.train_cols, self.test_cols = days[:-horizon], days[-horizon:]
        self.train = self.sales[self.train_cols].to_numpy(dtype=float)
        self.actual = self.sales[self.test_cols].to_numpy(dtype=float)
        recent = self.sales[['id', 'item_id', 'store_id'] + self.train_cols[-28:]].melt(
            id_vars=['id', 'item_id', 'store_id'], var_name='d', value_name='units')
        recent = recent.merge(calendar[['d', 'wm_yr_wk']], on='d', how='left', validate='many_to_one')
        if recent.wm_yr_wk.isna().any():
            raise ValueError('Calendar missing weight-window days')
        recent = recent.merge(prices, on=['item_id', 'store_id', 'wm_yr_wk'],
                              how='left', validate='many_to_one')
        if (recent.sell_price.isna() & recent.units.gt(0)).any():
            raise ValueError('Missing price for positive sales in weight window')
        recent['revenue'] = recent.units * recent.sell_price.fillna(0)
        self.revenue = recent.groupby('id').revenue.sum().reindex(self.ids).to_numpy()
        if self.revenue.sum() <= 0:
            raise ValueError('Weight window has no positive revenue')

    def evaluate(self, forecasts):
        columns = [f'F{i}' for i in range(1, self.horizon + 1)]
        if (forecasts.index.has_duplicates or set(forecasts.index) != set(self.ids)
                or list(forecasts.columns) != columns):
            raise ValueError('Forecast IDs and horizon must exactly match evaluation data')
        pred = forecasts.reindex(self.ids).to_numpy(dtype=float)
        if not np.isfinite(pred).all() or (pred < 0).any():
            raise ValueError('Forecasts must be finite and nonnegative')
        scores = []
        for keys in LEVELS:
            codes = (pd.factorize(pd.MultiIndex.from_frame(self.sales[keys]))[0]
                     if keys else np.zeros(len(self.ids), dtype=int))
            n = codes.max() + 1
            hist = np.zeros((n, self.train.shape[1]))
            errors = np.zeros((n, self.horizon))
            weights = np.zeros(n)
            np.add.at(hist, codes, self.train)
            np.add.at(errors, codes, pred - self.actual)
            np.add.at(weights, codes, self.revenue)
            # Exclude differences before (and into) the first nonzero observation.
            first = (hist != 0).argmax(axis=1)
            valid = np.arange(hist.shape[1] - 1)[None, :] >= first[:, None]
            scale = np.divide((np.diff(hist, axis=1)**2 * valid).sum(axis=1),
                              valid.sum(axis=1), out=np.zeros(n), where=valid.sum(axis=1) > 0)
            if ((scale == 0) & (weights > 0)).any():
                raise ValueError('RMSSE is undefined for a positive-weight constant series')
            rmsse = np.sqrt(np.divide((errors**2).mean(axis=1), scale,
                                     out=np.zeros(n), where=scale > 0))
            scores.append(float(np.dot(weights / weights.sum(), rmsse)))
        self.level_scores = scores
        return float(np.mean(scores))

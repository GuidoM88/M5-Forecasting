"""Create deterministic synthetic M5-shaped data, never a competition benchmark."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_demo_data(directory, n_days=160):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob('*.csv')):
        raise FileExistsError(f'Refusing to overwrite existing CSV files in {directory}')
    dates = pd.date_range('2015-01-01', periods=n_days)
    calendar = pd.DataFrame({'date': dates, 'd': [f'd_{i+1}' for i in range(n_days)],
                             'wm_yr_wk': np.arange(n_days)//7, 'wday': (dates.dayofweek+2)%7+1,
                             'month': dates.month, 'year': dates.year,
                             'snap_CA': 0, 'snap_TX': 0, 'snap_WI': 0})
    rng = np.random.default_rng(42)
    rows, prices = [], []
    for state in ['CA', 'TX']:
        for store in [1, 2]:
            for item in range(1, 5):
                item_id, store_id = f'FOODS_1_{item:03d}', f'{state}_{store}'
                row = dict(id=f'{item_id}_{store_id}_evaluation', item_id=item_id,
                           dept_id='FOODS_1', cat_id='FOODS', store_id=store_id, state_id=state)
                values = rng.poisson(item + 2 + 2*(dates.dayofweek >= 5))
                row.update(zip(calendar.d, values))
                rows.append(row)
                for week in calendar.wm_yr_wk.unique():
                    prices.append(dict(store_id=store_id, item_id=item_id, wm_yr_wk=week, sell_price=1.+item))
    calendar.to_csv(directory/'calendar.csv', index=False)
    pd.DataFrame(rows).to_csv(directory/'sales_train_evaluation.csv', index=False)
    pd.DataFrame(prices).to_csv(directory/'sell_prices.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/demo')
    args = parser.parse_args()
    make_demo_data(args.output)

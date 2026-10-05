import json
import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from scripts.make_demo_data import make_demo_data
from src.config import Config
from src.data_loader import M5DataLoader
from src.evaluation import M5Evaluator
from src.feature_engineering import FeatureEngineer
from src.model import HierarchicalLGBM
from src.pipeline import run_pipeline
from api.main import create_app


@pytest.fixture
def demo(tmp_path):
    raw = tmp_path / 'raw'
    make_demo_data(raw, 100)
    config = yaml.safe_load(open('config/demo.yaml'))
    config['paths'] = {'raw_data': str(raw), 'output': str(tmp_path / 'out'), 'models': str(tmp_path / 'models')}
    config['data'] = {'history_days': 72, 'test_horizon': 7}
    config['model']['training'].update(num_models=7, num_boost_round=2)
    path = tmp_path / 'test.yaml'
    path.write_text(yaml.safe_dump(config))
    return path, raw


def test_group_rolling_and_weekend():
    dates = pd.date_range('2020-01-01', periods=5)
    df = pd.DataFrame({'id': ['a']*5 + ['b']*5, 'date': list(dates)*2,
                       'sales': [100]*5 + [1,2,3,4,5]})
    fe = FeatureEngineer([1], [3])
    result = fe.create_all_features(df, [{'groupby':['id'], 'prefix':'b'}])
    b = result[result.id.eq('b')]
    assert b.b_roll_3.iloc[:3].isna().all()
    assert b.b_roll_3.iloc[3] == 2
    assert result.is_weekend.iloc[:5].tolist() == [0,0,0,1,1]


def test_horizon_origin_alignment():
    dates = pd.date_range('2020-01-01', periods=10)
    df = pd.DataFrame({'id':'a', 'date':dates, 'sales':np.arange(10)})
    fe = FeatureEngineer([1], [2])
    featured = fe.create_all_features(df, [{'groupby':['id'], 'prefix':'b'}])
    for h in range(1,5):
        X = HierarchicalLGBM.horizon_features(featured, ['b_lag_1','b_roll_2'], h)
        assert X.loc[5+h, 'b_lag_1'] == 5
        assert X.loc[5+h, 'b_roll_2'] == 4.5


def test_pipeline_and_holdout_invariance(demo):
    config, raw = demo
    forecast, summary = run_pipeline(config)
    assert forecast.shape == (16,7)
    assert np.isfinite(forecast.values).all()
    assert summary['mode'] == 'holdout_backtest'
    assert summary['wrmsse'] >= 0
    assert len(list((config.parent/'models').glob('horizon_*.txt'))) == 7
    sales = pd.read_csv(raw/'sales_train_evaluation.csv')
    days = [f'd_{i}' for i in range(94,101)]
    sales[days] += 1000
    sales.to_csv(raw/'sales_train_evaluation.csv', index=False)
    again, changed = run_pipeline(config)
    pd.testing.assert_frame_equal(forecast, again)
    assert changed['wrmsse'] != summary['wrmsse']
    loader = M5DataLoader(raw, 72, 7)
    train, test = loader.split_train_test(loader.load_data())
    assert train.date.nunique() == 72
    assert test.date.nunique() == 7


def test_wrmsse_alignment_and_perfect_forecast(demo):
    _, raw = demo
    sales = pd.read_csv(raw/'sales_train_evaluation.csv')
    evaluator = M5Evaluator(sales, pd.read_csv(raw/'calendar.csv'), pd.read_csv(raw/'sell_prices.csv'), 7)
    perfect = sales.set_index('id')[evaluator.test_cols]
    perfect.columns = [f'F{i}' for i in range(1,8)]
    assert evaluator.evaluate(perfect.iloc[::-1]) == 0
    with pytest.raises(ValueError):
        evaluator.evaluate(perfect.iloc[:-1])
    with pytest.raises(ValueError):
        evaluator.evaluate(perfect.assign(F1=np.nan))


def test_wrmsse_hand_calculation():
    # One series repeats 1,2, so every hierarchy has scale=1; +1 error => score=1.
    sales = pd.DataFrame([dict(id='a', item_id='a', store_id='s', state_id='CA', cat_id='c', dept_id='d',
                               **{f'd_{i}': 1+i%2 for i in range(1,41)})])
    cal = pd.DataFrame({'d':[f'd_{i}' for i in range(1,41)], 'wm_yr_wk':0})
    prices = pd.DataFrame({'item_id':['a'], 'store_id':['s'], 'wm_yr_wk':[0], 'sell_price':[2]})
    evaluator = M5Evaluator(sales, cal, prices, 2)
    assert evaluator.evaluate(pd.DataFrame([[3,2]], index=['a'], columns=['F1','F2'])) == pytest.approx(1)
    # Leading zero period must not dilute the denominator.
    sales.loc[0, ['d_1','d_2','d_3']] = 0
    evaluator = M5Evaluator(sales, cal, prices, 2)
    assert evaluator.evaluate(pd.DataFrame([[3,2]], index=['a'], columns=['F1','F2'])) == pytest.approx(1)


def test_api(demo):
    config, _ = demo
    forecast, summary = run_pipeline(config)
    app = create_app(config.parent/'out')
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/model/info').json()['training_date'] == summary['training_date']
        response = client.post('/predict', json={'item_ids':[forecast.index[0]]})
        assert response.status_code == 200
        assert response.json()['data'][0]['forecasts'] == pytest.approx(forecast.iloc[0].tolist())
        assert client.post('/predict', json={'item_ids':['missing']}).status_code == 404
        for body in [[], [''], ['a','a']]:
            assert client.post('/predict', json={'item_ids':body}).status_code == 422
        assert client.get('/items?offset=-1').status_code == 422
        assert client.get('/items?limit=1001').status_code == 422
    with TestClient(create_app(config.parent/'missing')) as client:
        assert client.get('/health').status_code == 503
        assert client.post('/predict', json={'item_ids':['a']}).status_code == 503


def test_configuration_rejects_inconsistent_horizon(demo):
    config, _ = demo
    settings = yaml.safe_load(config.read_text())
    settings['model']['training']['num_models'] = 28
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match='num_models'):
        Config(config)


def test_shared_model_and_research_baselines(demo):
    from src.research import load_panel, baseline_forecasts, tsb_forecast, conformal_radius
    config, raw = demo
    settings = yaml.safe_load(config.read_text())
    settings['model']['shared_model'] = True
    config.write_text(yaml.safe_dump(settings))
    forecast, _ = run_pipeline(config)
    sales = pd.read_csv(raw/'sales_train_evaluation.csv')
    sales[[f'd_{i}' for i in range(94,101)]] += 999
    sales.to_csv(raw/'sales_train_evaluation.csv', index=False)
    again, _ = run_pipeline(config)
    pd.testing.assert_frame_equal(forecast, again)
    history = pd.DataFrame([[1,2,3,4,5,6,7], [0]*7], index=['a','b'])
    base = baseline_forecasts(history, 9)
    assert base['seasonal_naive'].loc['a'].tolist() == [1,2,3,4,5,6,7,1,2]
    tsb = tsb_forecast(history, 9)
    assert (tsb.loc['b'] == 0).all()
    assert (tsb.loc['a'] > 0).all()
    actual = pd.DataFrame([np.arange(9)], index=['a'])
    pred = actual*0
    assert conformal_radius(actual, pred, 0.8) == 7


def test_evaluator_multilevel_weighting(demo):
    from src.evaluation import LEVELS
    _, raw = demo
    sales = pd.read_csv(raw/'sales_train_evaluation.csv')
    cal = pd.read_csv(raw/'calendar.csv')
    prices = pd.read_csv(raw/'sell_prices.csv')
    evaluator = M5Evaluator(sales, cal, prices, 7)
    pred = sales.set_index('id')[evaluator.test_cols].copy()
    pred.columns = [f'F{i}' for i in range(1,8)]
    pred.iloc[0] += 10
    # Slow reference groups independently; aggregate errors before taking RMSSE.
    expected = []
    for keys in LEVELS:
        groups = sales.groupby(keys).indices.values() if keys else [np.arange(len(sales))]
        subtotal = 0
        for indices in groups:
            indices = list(indices)
            history = evaluator.train[indices].sum(axis=0)
            history = history[np.flatnonzero(history)[0]:]
            scale = np.mean(np.diff(history)**2)
            error = (pred.to_numpy()[indices]-evaluator.actual[indices]).sum(axis=0)
            weight = evaluator.revenue[indices].sum()/evaluator.revenue.sum()
            subtotal += weight*np.sqrt(np.mean(error**2)/scale)
        expected.append(subtotal)
    assert evaluator.evaluate(pred) == pytest.approx(np.mean(expected))


def test_notebooks_valid_and_compile():
    import ast
    from pathlib import Path
    import nbformat
    for path in Path('notebooks').glob('*.ipynb'):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                assert not cell.outputs


def test_api_rejects_corrupt_forecasts(demo):
    config, _ = demo
    run_pipeline(config)
    path = config.parent/'out/forecasts.csv'
    frame = pd.read_csv(path)
    frame.loc[0,'F1'] = -1
    frame.to_csv(path,index=False)
    with TestClient(create_app(path.parent)) as client:
        assert client.get('/health').status_code == 503

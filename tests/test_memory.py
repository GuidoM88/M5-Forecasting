"""Regression tests for memory-bounded feature joins and sampled training."""
import numpy as np
import pandas as pd
import pytest
import yaml
from src.feature_engineering import FeatureEngineer
from src.memory import add_lookup_columns
from src.model import HierarchicalLGBM
from src.pipeline import run_pipeline
from scripts.make_demo_data import make_demo_data


def test_chunked_lookup_matches_merge_and_missing_values():
    frame = pd.DataFrame({'a':pd.Categorical(['b','a','a','c']), 'b':[1,2,1,3]})
    lookup = pd.DataFrame({'a':['a','a','b'], 'b':[1,2,1], 'value':[2.5,4.5,8.5]})
    expected = frame.merge(lookup, how='left', on=['a','b']).value
    result = add_lookup_columns(frame.copy(), lookup, ['a','b'], ['value'], chunk_size=2)
    np.testing.assert_allclose(result.value,expected,equal_nan=True)
    assert result.value.dtype == np.float32
    with pytest.raises(ValueError,match='Duplicate'):
        add_lookup_columns(frame, pd.concat([lookup,lookup]), ['a','b'], ['value'])


def test_hierarchical_features_match_independent_group_reference():
    dates = pd.date_range('2020-01-01',periods=10)
    df = pd.DataFrame({'id':np.repeat(['b','a','c'],10), 'date':np.tile(dates,3),
                       'item_id':np.repeat(['i1','i1','i2'],10),
                       'store_id':np.repeat(['s2','s1','s1'],10),
                       'sales':np.arange(30,dtype=float)})
    df.loc[df.date > dates[7],'sales'] = np.nan
    df.item_id = pd.Categorical(df.item_id,categories=['i1','i2','unused'])
    levels=[{'groupby':['id'],'prefix':'b'}, {'groupby':['item_id'],'prefix':'it'},
            {'groupby':['item_id','store_id'],'prefix':'is'}]
    fe=FeatureEngineer([1,3],[2,3])
    result=fe.create_all_features(df.sample(frac=1,random_state=42),levels)
    assert len(result)==len(df)
    assert not len(result.select_dtypes('object').columns)
    for level in levels:
        keys,prefix=level['groupby'],level['prefix']
        agg=(df.groupby(keys+['date'],observed=True,as_index=False).sales.mean()
             .sort_values(keys+['date']))
        g=agg.groupby(keys,observed=True).sales
        agg['expected_lag']=g.shift(3)
        agg['expected_roll']=g.transform(lambda s:s.shift(1).rolling(3).mean())
        merged=result.merge(agg[keys+['date','expected_lag','expected_roll']],on=keys+['date'])
        np.testing.assert_allclose(merged[f'{prefix}_lag_3'],merged.expected_lag,equal_nan=True)
        np.testing.assert_allclose(merged[f'{prefix}_roll_3'],merged.expected_roll,equal_nan=True,rtol=1e-6)
        assert result[f'{prefix}_roll_3'].dtype==np.float32
    # Per-horizon gather must exactly reproduce pandas group shift, including boundaries.
    names=['it_lag_3','it_roll_3','is_lag_3','is_roll_3','is_weekend']
    ordered=result.sort_values(['id','date'])
    for h in [1,2,7]:
        expected=ordered[names].copy()
        expected[names[:-1]]=ordered.groupby('id',observed=True)[names[:-1]].shift(h-1)
        actual=HierarchicalLGBM.horizon_features(ordered,names,h)
        np.testing.assert_allclose(actual,expected,equal_nan=True)


def test_sampled_training_is_repeatable_and_holdout_independent(tmp_path):
    raw=tmp_path/'raw'
    make_demo_data(raw,120)
    settings=yaml.safe_load(open('config/demo.yaml'))
    settings['paths']={'raw_data':str(raw),'output':str(tmp_path/'out'),'models':str(tmp_path/'models')}
    settings['data']={'history_days':92,'test_horizon':28}
    settings['model']['training'].update(num_boost_round=1,max_train_rows=400)
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(settings))
    first,summary=run_pipeline(path)
    assert max(summary['training_rows_by_horizon'].values())<=400
    sales=pd.read_csv(raw/'sales_train_evaluation.csv')
    sales[[f'd_{i}' for i in range(93,121)]]+=1000
    sales.to_csv(raw/'sales_train_evaluation.csv',index=False)
    second,changed=run_pipeline(path)
    pd.testing.assert_frame_equal(first,second)
    assert changed['wrmsse']!=summary['wrmsse']

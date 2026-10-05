"""Optional zero-shot benchmarks; model weights are downloaded only when called."""
import numpy as np
import pandas as pd


def ttm_forecast(history, horizon=28, batch_size=32):
    """Univariate zero-shot TTM, context-only scaling and a fixed holdout origin."""
    import torch
    from tsfm_public import TinyTimeMixerForPrediction
    model = TinyTimeMixerForPrediction.from_pretrained(
        'ibm-granite/granite-timeseries-ttm-r2', revision='main',
        prediction_filter_length=horizon,
    ).eval()
    context_length = model.config.context_length
    if history.shape[1] < context_length:
        raise ValueError(f'TTM checkpoint needs {context_length} context days')
    if horizon > model.config.prediction_length:
        raise ValueError('Requested horizon exceeds checkpoint prediction length')
    values = history.iloc[:, -context_length:].to_numpy(dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True).clip(min=1e-5)
    normalized = (values-mean)/scale
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(normalized[start:start+batch_size, :, None])
            output = model(past_values=batch).prediction_outputs[:, :horizon, 0].cpu().numpy()
            predictions.append(output*scale[start:start+batch_size]+mean[start:start+batch_size])
    return pd.DataFrame(np.maximum(np.concatenate(predictions), 0), index=history.index,
                        columns=[f'F{h}' for h in range(1,horizon+1)])


def chronos_forecast(history, calendar, horizon=28):
    """Chronos-2 with known future calendar covariates and checked date alignment."""
    from chronos import Chronos2Pipeline
    calendar = calendar.copy()
    calendar['date'] = pd.to_datetime(calendar.date)
    history_long = history.rename_axis('id').reset_index().melt(id_vars='id', var_name='d', value_name='target')
    columns = ['d','date','wday','month']
    context = history_long.merge(calendar[columns], on='d', validate='many_to_one').drop(columns='d')
    cutoff = context.date.max()
    future_days = pd.date_range(cutoff+pd.Timedelta(days=1), periods=horizon)
    future = pd.DataFrame({'id':history.index}).merge(
        calendar.loc[calendar.date.isin(future_days), ['date','wday','month']], how='cross')
    pipeline = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu')
    result = pipeline.predict_df(df=context, future_df=future, prediction_length=horizon,
                                 id_column='id', timestamp_column='date', target='target',
                                 quantile_levels=[0.5], batch_size=32)
    if '0.5' not in result:
        raise ValueError(f'Expected median column 0.5; got {list(result.columns)}')
    pivot = result.pivot(index='id', columns='date', values='0.5')
    pivot.columns = pd.to_datetime(pivot.columns)
    if set(pivot.columns) != set(future_days) or set(pivot.index) != set(history.index):
        raise ValueError('Foundation model returned a different forecast period or ID set')
    pivot = pivot.reindex(index=history.index, columns=future_days).clip(lower=0)
    if not np.isfinite(pivot.to_numpy()).all():
        raise ValueError('Foundation model returned incomplete forecasts')
    pivot.columns = [f'F{h}' for h in range(1,horizon+1)]
    return pivot

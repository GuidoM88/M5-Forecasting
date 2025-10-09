import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive,
    SeasonalNaive,
    CrostonClassic as Croston,
    ARIMA,
    AutoARIMA
)

def prepare_data_for_forecast(sales_long, n_series=100):
    """Prepare data in StatsForecast format"""
    
    # Select subset of series for faster testing
    series_ids = sales_long['id'].unique()[:n_series]
    data = sales_long[sales_long['id'].isin(series_ids)].copy()
    
    # Rename columns to StatsForecast format
    data = data.rename(columns={'id': 'unique_id', 'date': 'ds', 'sales': 'y'})
    data = data[['unique_id', 'ds', 'y']].sort_values(['unique_id', 'ds'])
    
    return data

def train_baseline_models(data, horizon=28):
    """Train multiple baseline models"""
    
    models = [
        Naive(),
        SeasonalNaive(season_length=7),  # Weekly seasonality
        Croston(),
        AutoARIMA(season_length=7)
    ]
    
    sf = StatsForecast(
        models=models,
        freq='D',
        n_jobs=-1
    )
    
    print(f"Training {len(models)} models on {data['unique_id'].nunique()} series...")
    forecasts = sf.forecast(df=data, h=horizon)
    
    return forecasts, sf

if __name__ == "__main__":
    # Example usage
    print("Baseline models module loaded successfully!")

"""Evaluation metrics for M5 forecasting."""
import pandas as pd
import numpy as np
from m5_wrmsse import wrmsse


class M5Evaluator:
    """Evaluate forecasts using M5 metrics."""
    
    def __init__(self, sales_df: pd.DataFrame):
        """
        Initialize evaluator.
        
        Parameters
        ----------
        sales_df : pd.DataFrame
            Original sales dataframe with 'id' column
        """
        self.sales_ids = sales_df["id"].tolist()
    
    def evaluate(self, forecast_df: pd.DataFrame) -> float:
        """
        Calculate WRMSSE score.
        
        Parameters
        ----------
        forecast_df : pd.DataFrame
            Forecast dataframe (30490 x 28)
            
        Returns
        -------
        float
            WRMSSE score
        """
        # Reindex to match expected order
        forecast_df = forecast_df.reindex(self.sales_ids).fillna(0)
        
        # Convert to numpy array
        forecast_array = forecast_df.values
        
        # Calculate WRMSSE
        score = wrmsse(forecast_array)
        
        return score

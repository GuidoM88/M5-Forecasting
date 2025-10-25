"""Predictor class for loading model and generating predictions."""
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class M5Predictor:
    """Load model and generate predictions."""
    
    def __init__(self, forecasts_path: str, summary_path: str):
        """
        Initialize predictor.
        
        Parameters
        ----------
        forecasts_path : str
            Path to forecasts pickle file
        summary_path : str
            Path to summary pickle file
        """
        self.forecasts_path = Path(forecasts_path)
        self.summary_path = Path(summary_path)
        self.forecasts_df = None
        self.summary = None
        self.model_loaded = False
        
    def load_model(self):
        """Load forecasts and summary from disk."""
        try:
            logger.info(f"Loading forecasts from {self.forecasts_path}")
            self.forecasts_df = pd.read_pickle(self.forecasts_path)
            
            logger.info(f"Loading summary from {self.summary_path}")
            with open(self.summary_path, "rb") as f:
                self.summary = pickle.load(f)
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(self, item_ids: List[str]) -> Dict[str, List[float]]:
        """
        Get predictions for specified item IDs.
        
        Parameters
        ----------
        item_ids : List[str]
            List of item IDs to predict
            
        Returns
        -------
        Dict[str, List[float]]
            Dictionary mapping item_id to list of 28 forecast values
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        predictions = {}
        
        for item_id in item_ids:
            if item_id in self.forecasts_df.index:
                # Get forecasts as list
                forecast = self.forecasts_df.loc[item_id].values.tolist()
                predictions[item_id] = forecast
            else:
                logger.warning(f"Item ID {item_id} not found in forecasts")
                # Return zeros for missing items
                predictions[item_id] = [0.0] * 28
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get model information and metrics."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return {
            "model_name": "Hierarchical LightGBM",
            "model_version": "v1",
            "wrmsse": self.summary.get("wrmsse"),
            "training_time": self.summary.get("training_time"),
            "history_days": self.summary.get("history_days"),
            "num_boost_round": self.summary.get("num_boost_round"),
            "lags": self.summary.get("lags"),
            "rolling_windows": self.summary.get("rolling_windows")
        }
    
    def get_available_items(self) -> List[str]:
        """Get list of all available item IDs."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.forecasts_df.index.tolist()

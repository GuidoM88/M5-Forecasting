"""Configuration loader for hierarchical LightGBM forecasting."""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration handler for the forecasting pipeline."""
    
    def __init__(self, config_path: str = "config/hierarchical_lgbm.yaml"):
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation).
        
        Parameters
        ----------
        key : str
            Configuration key (e.g., 'model.params.learning_rate')
        default : Any
            Default value if key not found
            
        Returns
        -------
        Any
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    @property
    def raw_data_path(self) -> Path:
        """Get raw data directory path."""
        return Path(self.get('paths.raw_data'))
    
    @property
    def output_path(self) -> Path:
        """Get output directory path."""
        return Path(self.get('paths.output'))
    
    @property
    def history_days(self) -> int:
        """Get number of historical days to use."""
        return self.get('data.history_days')
    
    @property
    def test_horizon(self) -> int:
        """Get test horizon (days to forecast)."""
        return self.get('data.test_horizon')
    
    @property
    def lags(self) -> list:
        """Get lag features."""
        return self.get('features.lags')
    
    @property
    def rolling_windows(self) -> list:
        """Get rolling window sizes."""
        return self.get('features.rolling_windows')
    
    @property
    def base_features(self) -> list:
        """Get list of base features."""
        return self.get('features.base_features')
    
    @property
    def categorical_features(self) -> list:
        """Get list of categorical features."""
        return self.get('features.categorical_features')
    
    @property
    def hierarchical_levels(self) -> list:
        """Get hierarchical aggregation levels."""
        return self.get('features.hierarchical_levels')
    
    @property
    def model_params(self) -> Dict[str, Any]:
        """Get LightGBM model parameters."""
        return self.get('model.params')
    
    @property
    def num_boost_round(self) -> int:
        """Get number of boosting rounds."""
        return self.get('model.training.num_boost_round')
    
    @property
    def num_models(self) -> int:
        """Get number of models to train (one per horizon)."""
        return self.get('model.training.num_models')

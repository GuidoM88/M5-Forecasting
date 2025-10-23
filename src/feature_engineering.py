"""Feature engineering for hierarchical forecasting - DEBUG VERSION."""
import pandas as pd
import numpy as np
from typing import List, Dict
import gc


class FeatureEngineer:
    """Create features for hierarchical time series forecasting."""
    
    def __init__(self, lags: List[int], rolling_windows: List[int]):
        self.lags = lags
        self.rolling_windows = rolling_windows
        
    def create_all_features(
        self, 
        df: pd.DataFrame, 
        hierarchical_levels: List[Dict]
    ) -> pd.DataFrame:
        """Create all features including hierarchical aggregations."""
        assert "sales" in df.columns, "Column 'sales' required"
                
        # Encode categorical variables
        df = self._encode_categorical(df)
        
        # Create date features
        df = self._create_date_features(df)
        
        # Create hierarchical features
        df = self._create_hierarchical_features(df, hierarchical_levels)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        for col in ["state_id", "store_id", "dept_id", "item_id"]:
            if col in df.columns:
                df[f"{col}_enc"] = pd.factorize(df[col])[0]
        return df
    
    def _create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create simple date-based features."""
        df["is_weekend"] = df["wday"].isin([1, 7]).astype(int)
        return df
    
    def _create_hierarchical_features(
        self, 
        df: pd.DataFrame, 
        hierarchical_levels: List[Dict]
    ) -> pd.DataFrame:
        """Create lag and rolling features at multiple hierarchical levels."""
        
        for level in hierarchical_levels:
            groupby_cols = level["groupby"]
            prefix = level["prefix"]
            
            
            if groupby_cols == ["id"]:
                # Bottom level: use original sales
                df = self._add_lag_rolling_features(
                    df, groupby_cols, "sales", prefix
                )
            else:
                # Aggregate level: create aggregated sales first
                agg_df = self._create_aggregated_sales(
                    df, groupby_cols, prefix
                )
                # Add lag/rolling on aggregated values
                agg_df = self._add_lag_rolling_features(
                    agg_df, groupby_cols, f"sales_{prefix}", prefix
                )
                # Merge back to main df
                keep_cols = groupby_cols + ["date"] + [
                    c for c in agg_df.columns if c.startswith(f"{prefix}_")
                ]
                df = df.merge(agg_df[keep_cols], on=groupby_cols + ["date"], how="left")
                
                del agg_df
                gc.collect()
            
        
        return df
    
    def _create_aggregated_sales(
        self, 
        df: pd.DataFrame, 
        groupby_cols: List[str],
        prefix: str
    ) -> pd.DataFrame:
        """Create aggregated sales for hierarchical level."""
        return (
            df.groupby(groupby_cols + ["date"], as_index=False)["sales"]
            .mean()
            .rename(columns={"sales": f"sales_{prefix}"})
        )
    
    def _add_lag_rolling_features(
        self, 
        df: pd.DataFrame, 
        groupby_cols: List[str],
        value_col: str,
        prefix: str
    ) -> pd.DataFrame:
        """Add lag and rolling features to dataframe."""
        df = df.sort_values(groupby_cols + ["date"])
        grp = df.groupby(groupby_cols, sort=False)[value_col]
        
        # Lag features
        for lag in self.lags:
            df[f"{prefix}_lag_{lag}"] = grp.shift(lag).values
        
        # Rolling features
        for window in self.rolling_windows:
            df[f"{prefix}_roll_{window}"] = (
                grp.shift(1).rolling(window).mean().values
            )
        
        return df
    
    def get_feature_names(self, hierarchical_levels: List[Dict]) -> List[str]:
        """Get list of all feature names."""
        base_feats = [
            "wday", "month", "year", "is_weekend", "snap",
            "sell_price", "state_id_enc", "store_id_enc", 
            "dept_id_enc", "item_id_enc"
        ]
        
        lag_feats = []
        roll_feats = []
        
        for level in hierarchical_levels:
            prefix = level["prefix"]
            lag_feats.extend([f"{prefix}_lag_{l}" for l in self.lags])
            roll_feats.extend([f"{prefix}_roll_{w}" for w in self.rolling_windows])
        
        return base_feats + lag_feats + roll_feats

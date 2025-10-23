"""Data loading and preprocessing"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


class M5DataLoader:
    """Load and preprocess M5 competition data."""
    
    def __init__(self, raw_dir: Path, history_days: int, test_horizon: int = 28):
        self.raw_dir = Path(raw_dir)
        self.history_days = history_days
        self.test_horizon = test_horizon
        
    def load_data(self) -> pd.DataFrame:
        """Load and merge all M5 data files."""
        print("Loading raw files...")
        
        # Load raw files
        calendar = self._load_calendar()
        prices = self._load_prices()
        sales = self._load_sales()
        
        # Reduce sales to required time window
        sales_small = self._filter_sales_by_date(sales, calendar)
        
        # Melt to long format
        long = self._melt_sales(sales_small)
        
        # Merge with calendar
        long = long.merge(
            calendar[["d", "date", "wm_yr_wk", "wday", "month", "year",
                     "snap_CA", "snap_TX", "snap_WI"]],
            on="d", 
            how="left"
        )
        long["date"] = pd.to_datetime(long["date"])
        
        # Merge with prices
        long = long.merge(
            prices, 
            on=["store_id", "item_id", "wm_yr_wk"], 
            how="left"
        )
                
        # Create unified SNAP feature
        long = self._create_snap_feature(long)
        
        # Keep only necessary columns
        long = long[[
            "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "date", "sales", "sell_price", "wday", "month", "year", "snap"
        ]].copy()
        
        # Final temporal filter
        long = self._apply_temporal_filter(long)
                
        return long
    
    def _load_calendar(self) -> pd.DataFrame:
        """Load calendar file."""
        return pd.read_csv(
            self.raw_dir / "calendar.csv",
            usecols=["date", "d", "wm_yr_wk", "wday", "month", "year",
                    "snap_CA", "snap_TX", "snap_WI"]
        )
    
    def _load_prices(self) -> pd.DataFrame:
        """Load sell prices file."""
        return pd.read_csv(
            self.raw_dir / "sell_prices.csv",
            usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"]
        )
    
    def _load_sales(self) -> pd.DataFrame:
        """Load sales training evaluation file."""
        return pd.read_csv(self.raw_dir / "sales_train_evaluation.csv")
    
    def _filter_sales_by_date(
        self, 
        sales: pd.DataFrame, 
        calendar: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter sales columns to required date range."""
        d_cols = [c for c in sales.columns if c.startswith("d_")]
        d2date = dict(zip(calendar["d"], calendar["date"]))
        dates = pd.to_datetime([d2date[d] for d in d_cols])
        last_date = dates.max()
                
        # Keep history_days + test_horizon
        keep_mask = dates >= (last_date - pd.Timedelta(days=self.history_days + self.test_horizon))
        keep_cols = [c for c, m in zip(d_cols, keep_mask) if m]
                
        return pd.concat([sales[sales.columns[:6]], sales[keep_cols]], axis=1)
    
    def _melt_sales(self, sales: pd.DataFrame) -> pd.DataFrame:
        """Convert sales from wide to long format."""
        return sales.melt(
            id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            var_name="d", 
            value_name="sales"
        )
    
    def _create_snap_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create unified SNAP feature based on state."""
        df["snap"] = 0
        df.loc[(df["state_id"] == "CA") & (df["snap_CA"] == 1), "snap"] = 1
        df.loc[(df["state_id"] == "TX") & (df["snap_TX"] == 1), "snap"] = 1
        df.loc[(df["state_id"] == "WI") & (df["snap_WI"] == 1), "snap"] = 1
        return df
    
    def _apply_temporal_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final temporal filter to data."""
        max_date = df["date"].max()
        cut_date = max_date - pd.Timedelta(days=self.test_horizon)
        hist_start = cut_date - pd.Timedelta(days=self.history_days)
                
        return df[
            (df["date"] >= hist_start) & (df["date"] <= max_date)
        ].copy()
    
    def split_train_test(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        max_date = df["date"].max()
        cut_date = max_date - pd.Timedelta(days=self.test_horizon)
        
        train_df = df[df["date"] <= cut_date].copy()
        test_df = df[df["date"] > cut_date].copy()
        
        return train_df, test_df

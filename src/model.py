"""Hierarchical LightGBM model for M5 forecasting - DEBUG VERSION."""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import gc


class HierarchicalLGBM:
    """Train multiple LightGBM models for hierarchical forecasting."""
    
    def __init__(
        self, 
        params: Dict, 
        num_boost_round: int,
        num_models: int = 28
    ):
        """
        Initialize hierarchical LightGBM trainer.
        
        Parameters
        ----------
        params : Dict
            LightGBM parameters
        num_boost_round : int
            Number of boosting rounds
        num_models : int
            Number of models to train (one per forecast horizon)
        """
        self.params = params
        self.num_boost_round = num_boost_round
        self.num_models = num_models
        self.models = {}
        
    def train(
        self, 
        train_df: pd.DataFrame, 
        feature_names: List[str]
    ) -> None:
        """
        Train one model per forecast horizon (non-recursive).
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training dataframe with features and target
        feature_names : List[str]
            List of feature column names
        """
        print(f"Training {self.num_models} models...")
        
        # Identify lag and rolling features
        lag_roll_feats = [f for f in feature_names 
                         if "_lag_" in f or "_roll_" in f]
                
        for h in tqdm(range(1, self.num_models + 1)):
            # Shift lag/rolling features by h for horizon h
            Xh = train_df.copy()
            grp = Xh.groupby("id", sort=False)
            
            for col in lag_roll_feats:
                Xh[col] = grp[col].shift(h).values
            
            # Drop rows with NaN features
            Xh = Xh.dropna(subset=feature_names)
            
            # Prepare data
            y = Xh["sales"].values
            X = Xh[feature_names].values
                        
            # Train model
            dtrain = lgb.Dataset(X, y)
            
            self.models[h] = lgb.train(
                params=self.params,
                train_set=dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=[dtrain],
                valid_names=["train"],
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            del Xh, X, y, dtrain
            gc.collect()
        
    
    def predict(
        self, 
        test_df: pd.DataFrame, 
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Generate predictions for test period.
        
        Parameters
        ----------
        test_df : pd.DataFrame
            Test dataframe
        feature_names : List[str]
            List of feature column names
            
        Returns
        -------
        pd.DataFrame
            Predictions in wide format (30490 x 28)
        """
        print(f"Predicting {self.num_models} days...")
        
        test_days = sorted(test_df["date"].unique())
        
        assert len(test_days) >= self.num_models
        test_days = test_days[:self.num_models]
        
        pred_list = []
        
        for h, day in tqdm(list(zip(range(1, self.num_models + 1), test_days))):
            Xtest = test_df[test_df["date"] == day].copy()
            Xmat = Xtest[feature_names].fillna(0).values
            
            pred = self.models[h].predict(Xmat)
            
            out = Xtest[["id"]].copy()
            out["h"] = h
            out["forecast"] = np.clip(pred, 0, None)
            
            pred_list.append(out)
        
        # Pivot to wide format
        pred_all = pd.concat(pred_list, axis=0)
        
        # Load sales to get correct ID order
        sales_path = Path("data/raw/sales_train_evaluation.csv")
        sales = pd.read_csv(sales_path)
        
        pivot = pred_all.pivot(
            index="id", 
            columns="h", 
            values="forecast"
        ).reindex(sales["id"].tolist()).fillna(0)
        
        pivot.columns = [f"F{i}" for i in range(1, self.num_models + 1)]
        
        return pivot

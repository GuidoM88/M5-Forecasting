#!/usr/bin/env python
"""Training script for hierarchical LightGBM forecasting - DEBUG VERSION."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
from src.config import Config
from src.data_loader import M5DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import HierarchicalLGBM
from src.evaluation import M5Evaluator


def main(config_path: str = "config/hierarchical_lgbm.yaml"):
    """Main training pipeline with DEBUG prints."""
    
    # Load configuration
    print("=" * 80)
    print("Hierarchical LightGBM Forecasting Pipeline - DEBUG")
    print("=" * 80)
    
    config = Config(config_path)
    
    print(f"\nConfig: {config.config_path}")
    print(f"- History days: {config.history_days}")
    print(f"- Lags: {config.lags}")
    print(f"- Rolling: {config.rolling_windows}")
    print(f"- Rounds: {config.num_boost_round}")
    print("=" * 80)
    
    # Create output directory
    config.output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[STEP 1] Loading data...")
    print("-" * 80)
    
    loader = M5DataLoader(
        config.raw_data_path,
        config.history_days,
        config.test_horizon
    )
    data = loader.load_data()
        
    # Feature engineering PRIMA dello split (come nel notebook)
    print("\n[STEP 2] Feature engineering...")
    print("-" * 80)
    
    fe = FeatureEngineer(config.lags, config.rolling_windows)
    
    print("Creating hierarchical features on full dataset...")
    data = fe.create_all_features(data, config.hierarchical_levels)
        
    # Split train/test DOPO feature engineering
    print("\n[STEP 3] Splitting train/test...")
    print("-" * 80)
    
    train_df, test_df = loader.split_train_test(data)
    
    feature_names = fe.get_feature_names(config.hierarchical_levels)
        
    # Drop NaN rows in training
    train_df = train_df.dropna(subset=feature_names)
    
    # Train models
    print("\n[STEP 3] Training models...")
    print("-" * 80)
    
    model = HierarchicalLGBM(
        params=config.model_params,
        num_boost_round=config.num_boost_round,
        num_models=config.num_models
    )
        
    model.train(train_df, feature_names)
    
    # Predict
    print("\n[STEP 4] Generating predictions...")
    print("-" * 80)
    
    forecast_df = model.predict(test_df, feature_names)
        
    # Evaluate
    print("\n[STEP 5] Evaluating...")
    print("-" * 80)
    
    # Load original sales for evaluation
    sales = pd.read_csv(config.raw_data_path / "sales_train_evaluation.csv")
    evaluator = M5Evaluator(sales)
    
    score = evaluator.evaluate(forecast_df)
    print(f"\n✅ Hierarchical WRMSSE: {score:.4f}")
    
    # Save results
    print("\n[STEP 6] Saving results...")
    print("-" * 80)
    
    forecast_path = config.output_path / config.get('output.forecast_filename')
    summary_path = config.output_path / config.get('output.summary_filename')
    
    forecast_df.to_pickle(forecast_path)
    print(f"Forecasts saved to: {forecast_path}")
    
    with open(summary_path, "wb") as f:
        pickle.dump({
            "wrmsse": float(score),
            "history_days": config.history_days,
            "lags": config.lags,
            "rolls": config.rolling_windows,
            "rounds": config.num_boost_round
        }, f)
    print(f"Summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train hierarchical LightGBM model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/hierarchical_lgbm.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    main(args.config)

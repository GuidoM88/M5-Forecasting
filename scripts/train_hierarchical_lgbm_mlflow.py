#!/usr/bin/env python
"""Training script with MLflow tracking for hierarchical LightGBM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
import time
import mlflow
import mlflow.lightgbm
from src.config import Config
from src.data_loader import M5DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import HierarchicalLGBM
from src.evaluation import M5Evaluator


def main(config_path: str = "config/hierarchical_lgbm.yaml"):
    """Main training pipeline with MLflow tracking."""
    
    # Load configuration
    config = Config(config_path)
    
    # Set MLflow experiment
    mlflow.set_experiment("M5_Hierarchical_Forecasting")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"LGBM_h{config.history_days}_r{config.num_boost_round}"):
        
        print("=" * 80)
        print("MLflow Hierarchical LightGBM Training Pipeline")
        print("=" * 80)
        
        start_time = time.time()
        
        # Log configuration parameters
        mlflow.log_params({
            "history_days": config.history_days,
            "test_horizon": config.test_horizon,
            "lags": str(config.lags),
            "rolling_windows": str(config.rolling_windows),
            "num_boost_round": config.num_boost_round,
            "num_models": config.num_models,
            "learning_rate": config.model_params.get("learning_rate"),
            "num_leaves": config.model_params.get("num_leaves"),
            "max_depth": config.model_params.get("max_depth"),
        })
        
        # Create output directories
        config.output_path.mkdir(parents=True, exist_ok=True)
        models_path = Path("models/hierarchical_lgbm")
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load data
        print("\n[STEP 1] Loading data...")
        print("-" * 80)
        
        loader = M5DataLoader(
            config.raw_data_path,
            config.history_days,
            config.test_horizon
        )
        data = loader.load_data()
        
        mlflow.log_metric("data_rows", len(data))
        print(f"Loaded {len(data):,} rows")
        
        # Step 2: Feature engineering (BEFORE split)
        print("\n[STEP 2] Feature engineering...")
        print("-" * 80)
        
        fe = FeatureEngineer(config.lags, config.rolling_windows)
        data = fe.create_all_features(data, config.hierarchical_levels)
        feature_names = fe.get_feature_names(config.hierarchical_levels)
        
        mlflow.log_metric("num_features", len(feature_names))
        print(f"Created {len(feature_names)} features")
        
        # Step 3: Split train/test
        print("\n[STEP 3] Splitting train/test...")
        print("-" * 80)
        
        train_df, test_df = loader.split_train_test(data)
        train_df = train_df.dropna(subset=feature_names)
        
        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("test_rows", len(test_df))
        print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
        
        # Step 4: Train models
        print("\n[STEP 4] Training models...")
        print("-" * 80)
        
        model = HierarchicalLGBM(
            params=config.model_params,
            num_boost_round=config.num_boost_round,
            num_models=config.num_models
        )
        
        training_start = time.time()
        model.train(train_df, feature_names)
        training_time = time.time() - training_start
        
        mlflow.log_metric("training_time_seconds", training_time)
        print(f"Training completed in {training_time:.1f} seconds")
        
        # Step 5: Predict
        print("\n[STEP 5] Generating predictions...")
        print("-" * 80)
        
        forecast_df = model.predict(test_df, feature_names)
        
        # Step 6: Evaluate
        print("\n[STEP 6] Evaluating...")
        print("-" * 80)
        
        sales = pd.read_csv(config.raw_data_path / "sales_train_evaluation.csv")
        evaluator = M5Evaluator(sales)
        wrmsse_score = evaluator.evaluate(forecast_df)
        
        # Log primary metric
        mlflow.log_metric("WRMSSE", wrmsse_score)
        print(f"\n✅ WRMSSE: {wrmsse_score:.4f}")
        
        # Step 7: Save artifacts
        print("\n[STEP 7] Saving artifacts...")
        print("-" * 80)
        
        # Save forecasts
        forecast_path = config.output_path / "forecasts.pkl"
        forecast_df.to_pickle(forecast_path)
        mlflow.log_artifact(str(forecast_path), "forecasts")
        
        # Save summary
        summary = {
            "wrmsse": float(wrmsse_score),
            "history_days": config.history_days,
            "lags": config.lags,
            "rolling_windows": config.rolling_windows,
            "num_boost_round": config.num_boost_round,
            "training_time": training_time,
            "num_features": len(feature_names)
        }
        
        summary_path = config.output_path / "summary.pkl"
        with open(summary_path, "wb") as f:
            pickle.dump(summary, f)
        mlflow.log_artifact(str(summary_path), "summary")
        
        # Save config
        mlflow.log_artifact(str(config.config_path), "config")
        
        # Log total time
        total_time = time.time() - start_time
        mlflow.log_metric("total_time_seconds", total_time)
        
        print(f"\nTotal pipeline time: {total_time:.1f} seconds")
        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with MLflow tracking")
    parser.add_argument(
        "--config",
        type=str,
        default="config/hierarchical_lgbm.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    main(args.config)

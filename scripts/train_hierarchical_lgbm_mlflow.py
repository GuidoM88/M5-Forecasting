"""Run the same backtest with optional MLflow tracking."""
import argparse
from src.config import Config
from src.pipeline import run_pipeline


def main():
    import mlflow
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/hierarchical_lgbm.yaml')
    parser.add_argument('--tracking-uri', default='sqlite:///mlflow.db')
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment('M5 Hierarchical Forecasting')
    with mlflow.start_run():
        _, summary = run_pipeline(args.config)
        mlflow.log_params({k: summary[k] for k in ['history_days', 'horizon', 'num_boost_round', 'cutoff']})
        mlflow.log_metrics({k: summary[k] for k in ['wrmsse', 'seasonal_naive_wrmsse', 'training_time']})
        mlflow.log_artifacts(str(Config(args.config).output_path))


if __name__ == '__main__':
    main()

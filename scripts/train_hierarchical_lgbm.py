"""Run a fixed-origin holdout backtest."""
import argparse
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/hierarchical_lgbm.yaml')
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == '__main__':
    main()

# M5 Forecasting - Production MLOps Pipeline

End-to-end production-ready time series forecasting system for retail sales prediction using hierarchical LightGBM models with MLflow tracking and FastAPI deployment.

## Performance

- WRMSSE Score: 0.6140 (150/5558, ~2.7%)
- Dataset: Walmart M5 Forecasting (30,490 products, 1,969 days)
- Forecast Horizon: 28 days ahead
- Model: Hierarchical LightGBM with bottom-up and grouped features

## Project Structure

M5-Forecasting
```
├── api/ REST API for model serving
│ ├── init.py
│ ├── main.py FastAPI application
│ ├── schemas.py Pydantic models for validation
│ └── predictor.py Model serving logic
├── config/ YAML configurations
│ └── hierarchical_lgbm.yaml Model and training parameters
├── data/ Data directory (gitignored)
│ ├── raw/ M5 CSV files (calendar, sell_prices, sales)
│ ├── baseline_results/
│ ├── conformal_results/
│ ├── lightgbm_results/
│ ├── multihorizon_results/
│ ├── predictability_results/
│ ├── tsb_results/
│ └── ttm_results/
├── mlruns/ MLflow experiment tracking (gitignored)
├── models/ Saved model artifacts (gitignored)
├── notebooks/ Jupyter notebooks for exploration
│ └── 08_hierarchical_LGBM.ipynb
├── outputs/ Model predictions and results (gitignored)
│ └── forecasts/
│ ├── forecasts.pkl
│ └── summary.pkl
├── scripts/ Execution scripts
│ ├── train_hierarchical_lgbm.py Standard training
│ └── train_with_mlflow.py Training with MLflow tracking
├── src/ Core package
│ ├── init.py
│ ├── config.py Configuration management
│ ├── data_loader.py M5 data loading and preprocessing
│ ├── evaluation.py WRMSSE metric calculation
│ ├── feature_engineering.py Lag/rolling features at multiple levels
│ └── model.py Hierarchical LightGBM implementation
├── .gitignore Git ignore rules
├── Dockerfile Docker containerization
├── docker-compose.yml Docker orchestration
├── README.md Project documentation (this file)
├── README_DOCKER.md Docker deployment guide
├── requirements.txt Python dependencies
└── setup.py Package installation configuration
```
## Features

Model Architecture: Bottom-level features (per product-store), Item-level aggregated features, Department-store cross features, State-store aggregated features, Lag features (7, 14, 28 days), Rolling mean features (7, 14, 28 days)

MLOps Pipeline: Modular Python package structure, YAML-based configuration, MLflow experiment tracking, REST API with FastAPI, Docker containerization, Comprehensive logging

## Installation

Requirements: Python 3.10+

Setup:

git clone https://github.com/YOUR_USERNAME/multivariate-retail-forecasting.git

cd multivariate-retail-forecasting

conda create -n mlops_env python=3.10

conda activate mlops_env

pip install -r requirements.txt

pip install -e .

python src/data/download.py 

## Usage

Training (standard):

python scripts/train_hierarchical_lgbm.py

Training with MLflow:

python scripts/train_with_mlflow.py

MLflow UI:

mlflow ui --backend-store-uri file:///$(pwd)/mlruns

Access at: http://localhost:5000

API Deployment (local):

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

API Deployment (Docker):

docker-compose up --build

API documentation: http://localhost:8000/docs

## API Endpoints

Health check:

curl http://localhost:8000/health

Model information:

curl http://localhost:8000/model/info

Generate forecasts:

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"item_ids": ["HOBBIES_1_001_CA_1_evaluation"]}'

## Configuration

Edit config/hierarchical_lgbm.yaml to customize parameters: history_days (600), test_horizon (28), lags ([7, 14, 28]), rolling_windows ([7, 14, 28]), num_boost_round (200), learning_rate (0.08), num_leaves (31), max_depth (8)

## Model Details

Training Process: Load M5 data with configurable history, Create hierarchical features (bottom + aggregated levels), Train 28 separate LightGBM models (one per forecast horizon), Generate recursive multi-step forecasts, Evaluate with WRMSSE metric

Feature Engineering Levels: Bottom-level (b_) Individual product-store time series, Item-level (it_) Product across all stores, Dept-Store (ds_) Department within store, State-Store (ss_) Store across state. Each level includes lag and rolling mean features.

## Performance Metrics

WRMSSE: 0.6140
Training Time: ~30 min (M1 Mac)
Prediction Time: <2 sec

## Tech Stack

Framework: Python 3.10, ML: LightGBM Pandas NumPy, MLOps: MLflow, API: FastAPI Uvicorn, Containerization: Docker Docker-Compose, Config: YAML

## Development

Code Quality: Modular package structure, Type hints, Comprehensive logging, YAML configuration, Clean separation of concerns

Testing:

pytest tests/

## Docker

Build and run:

docker-compose up --build

Stop:

docker-compose down

See README_DOCKER.md for detailed Docker instructions.

## Contributing

This project follows MLOps best practices: Modular code structure, Reproducible experiments with MLflow, Configuration management, API-first deployment, Containerization

## Acknowledgments

M5 Forecasting competition dataset (Kaggle/Walmart), Hierarchical forecasting methodology

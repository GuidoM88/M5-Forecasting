"""FastAPI application for M5 forecasting API."""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ForecastData,
    HealthResponse,
    ModelInfoResponse
)
from api.predictor import M5Predictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="M5 Hierarchical Forecasting API",
    description="REST API for generating retail sales forecasts using hierarchical LightGBM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (will be loaded on startup)
predictor = M5Predictor(
    forecasts_path="outputs/forecasts/forecasts.pkl",
    summary_path="outputs/forecasts/summary.pkl"
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting up API...")
    try:
        predictor.load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "M5 Hierarchical Forecasting API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        loaded=predictor.model_loaded,  # Changed from model_loaded
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information and metrics."""
    try:
        info = predictor.get_model_info()
        
        return ModelInfoResponse(
            name=info["model_name"],  # Changed from model_name
            version=info["model_version"],  # Changed from model_version
            training_date="2025-10-23",
            metrics={
                "WRMSSE": info["wrmsse"],
                "training_time_seconds": info["training_time"]
            },
            parameters={
                "history_days": info["history_days"],
                "num_boost_round": info["num_boost_round"],
                "lags": info["lags"],
                "rolling_windows": info["rolling_windows"]
            }
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Generate forecasts for specified item IDs.
    
    Returns 28-day forecast for each requested item.
    """
    try:
        # Get predictions
        predictions = predictor.predict(request.item_ids)
        
        # Format response
        forecast_data = [
            ForecastData(item_id=item_id, forecasts=forecasts)
            for item_id, forecasts in predictions.items()
        ]
        
        return PredictionResponse(
            status="success",
            message=f"Generated forecasts for {len(request.item_ids)} items",
            data=forecast_data,
            version="hierarchical_lgbm_v1"  # Changed from model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/items", tags=["Items"])
async def list_items(limit: int = 100, offset: int = 0):
    """
    List available item IDs.
    
    Parameters
    ----------
    limit : int
        Maximum number of items to return
    offset : int
        Offset for pagination
    """
    try:
        all_items = predictor.get_available_items()
        total = len(all_items)
        items = all_items[offset:offset + limit]
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items
        }
        
    except Exception as e:
        logger.error(f"Error listing items: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
from datetime import date


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    
    item_ids: List[str] = Field(
        ...,
        description="List of item IDs to predict (e.g., ['FOODS_1_001_CA_1_evaluation'])",
        min_length=1,
        examples=[["HOBBIES_1_001_CA_1_evaluation", "FOODS_1_001_TX_1_evaluation"]]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "item_ids": [
                    "HOBBIES_1_001_CA_1_evaluation",
                    "FOODS_1_001_TX_1_evaluation",
                    "HOUSEHOLD_1_001_WI_1_evaluation"
                ]
            }
        }
    )


class ForecastData(BaseModel):
    """Forecast data for a single item."""
    
    item_id: str = Field(..., description="Item ID")
    forecasts: List[float] = Field(..., description="28-day forecast values")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "item_id": "HOBBIES_1_001_CA_1_evaluation",
                "forecasts": [1.2, 1.3, 1.1, 1.4, 1.5, 1.6, 1.8, 2.0] + [0.0] * 20
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: List[ForecastData] = Field(..., description="List of forecasts")
    version: str = Field(..., description="Model version used")
    
    model_config = ConfigDict(
        protected_namespaces=(),  # Disable model_ namespace protection
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Predictions generated successfully",
                "data": [
                    {
                        "item_id": "HOBBIES_1_001_CA_1_evaluation",
                        "forecasts": [1.2, 1.3, 1.1] + [0.0] * 25
                    }
                ],
                "version": "hierarchical_lgbm_v1"
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    
    model_config = ConfigDict(
        protected_namespaces=(),  # Disable model_ namespace protection
        json_schema_extra={
            "example": {
                "status": "healthy",
                "loaded": True,
                "version": "1.0.0"
            }
        }
    )


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    training_date: str = Field(..., description="Training date")
    metrics: Dict[str, float] = Field(..., description="Model metrics")
    parameters: Dict[str, Any] = Field(..., description="Model parameters")  # Fixed: any -> Any
    
    model_config = ConfigDict(
        protected_namespaces=(),  # Disable model_ namespace protection
        json_schema_extra={
            "example": {
                "name": "Hierarchical LightGBM",
                "version": "v1",
                "training_date": "2025-10-23",
                "metrics": {"WRMSSE": 0.6171},
                "parameters": {"history_days": 600, "num_boost_round": 200}
            }
        }
    )

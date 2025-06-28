"""
FastAPI application for credit risk prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from typing import List
from src.predict import CreditRiskPredictor
from src.api.pydantic_models import CustomerTransaction, CreditRiskPrediction, BatchPredictionRequest, BatchPredictionResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting customer credit risk based on transaction data",
    version="1.0.0"
)

# Initialize predictor
predictor = CreditRiskPredictor()

@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    """
    return {
        "api_name": "Credit Risk Prediction API",
        "version": "1.0.0",
        "description": "API for predicting customer credit risk",
        "endpoints": {
            "/predict": "Single prediction endpoint",
            "/batch_predict": "Batch prediction endpoint"
        }
    }

@app.post("/predict", response_model=CreditRiskPrediction)
async def predict_credit_risk(customer_data: CustomerTransaction):
    """
    Predict credit risk for a single customer.
    
    Args:
        customer_data: Customer transaction data
        
    Returns:
        Credit risk prediction
    """
    try:
        result = predictor.predict(customer_data.dict())
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict_credit_risk(request: BatchPredictionRequest):
    """
    Predict credit risk for multiple customers in batch.
    
    Args:
        request: Batch prediction request containing multiple customer records
        
    Returns:
        Batch prediction results
    """
    try:
        predictions = []
        for customer in request.customers:
            result = predictor.predict(customer.dict())
            predictions.append(result)
            
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

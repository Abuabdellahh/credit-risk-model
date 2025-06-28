"""
Pydantic models for API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class CustomerTransaction(BaseModel):
    """
    Model for customer transaction data.
    """
    CustomerId: str = Field(..., description="Unique customer identifier")
    TransactionStartTime: str = Field(..., description="Transaction timestamp")
    Frequency: int = Field(..., description="Number of transactions")
    Monetary: float = Field(..., description="Total transaction value")
    Recency: int = Field(..., description="Days since last transaction")

class CreditRiskPrediction(BaseModel):
    """
    Model for credit risk prediction response.
    """
    customer_id: str = Field(..., description="Customer identifier")
    risk_probability: float = Field(..., description="Predicted risk probability", ge=0, le=1)
    risk_prediction: int = Field(..., description="Binary risk prediction", ge=0, le=1)
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionRequest(BaseModel):
    """
    Model for batch prediction request.
    """
    customers: List[CustomerTransaction] = Field(..., description="List of customer data")

class BatchPredictionResponse(BaseModel):
    """
    Model for batch prediction response.
    """
    predictions: List[CreditRiskPrediction] = Field(..., description="List of predictions")

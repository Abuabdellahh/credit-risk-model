"""
Credit risk prediction module.
"""

import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditRiskPredictor:
    """
    Class for making credit risk predictions.
    """
    
    def __init__(self, model_name: str = "credit_risk_model"):
        """
        Initialize the predictor with a specific model.
        
        Args:
            model_name: Name of the registered model in MLflow
        """
        self.model_name = model_name
        self.model = None
        
    def load_model(self, stage: str = "Production"):
        """
        Load the model from MLflow registry.
        
        Args:
            stage: Stage of the model to load (Production, Staging, etc.)
        """
        try:
            model_uri = f"models:/{self.model_name}/{stage}"
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from {model_uri}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Dictionary containing customer transaction data
            
        Returns:
            DataFrame ready for prediction
        """
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Calculate RFM metrics
            snapshot_date = pd.Timestamp.now()
            df['Recency'] = (snapshot_date - pd.to_datetime(df['TransactionStartTime'])).dt.days
            
            # Calculate additional features
            df['AvgTransactionAmount'] = df['Monetary'] / df['Frequency']
            df['TransactionFrequency'] = df['Frequency'] / df['Recency']
            
            # Select required features
            features = ['Recency', 'Frequency', 'Monetary', 'AvgTransactionAmount', 'TransactionFrequency']
            return df[features]
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single customer.
        
        Args:
            input_data: Dictionary containing customer data
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.model:
                self.load_model()
            
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Make prediction
            proba = self.model.predict_proba(X)[0][1]
            prediction = int(proba >= 0.5)
            
            return {
                "customer_id": input_data.get("CustomerId"),
                "risk_probability": float(proba),
                "risk_prediction": prediction,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

def predict_main():
    """
    Example usage of the predictor.
    """
    predictor = CreditRiskPredictor()
    
    # Example input data
    sample_input = {
        "CustomerId": "C12345",
        "TransactionStartTime": "2025-06-28",
        "Frequency": 10,
        "Monetary": 5000,
        "Recency": 30
    }
    
    try:
        result = predictor.predict(sample_input)
        print(f"Prediction result: {result}")
        
    except Exception as e:
        logger.error(f"Error in main prediction: {str(e)}")
        raise

if __name__ == "__main__":
    predict_main()

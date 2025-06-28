"""
Train credit risk prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import mlflow
import mlflow.sklearn
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data(data_path: str) -> pd.DataFrame:
    """
    Load processed data for training.
    
    Args:
        data_path: Path to processed data file
        
    Returns:
        DataFrame containing features and target
    """
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for modeling.
    
    Args:
        df: DataFrame containing all data
        
    Returns:
        Tuple of (X, y)
    """
    try:
        features = ['Recency', 'Frequency', 'Monetary', 'AvgTransactionAmount', 'TransactionFrequency']
        X = df[features]
        y = df['is_high_risk']
        return X, y
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise

def train_models(X_train, y_train, X_val, y_val):
    """
    Train and evaluate multiple models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
    """
    models = {
        'logistic': LogisticRegression(),
        'random_forest': RandomForestClassifier(),
        'gradient_boosting': GradientBoostingClassifier()
    }
    
    # Define hyperparameter grids
    param_grids = {
        'logistic': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2']
        },
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
    
    best_model = None
    best_score = 0
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Set experiment name
            mlflow.set_experiment("credit_risk_model")
            
            # Log parameters
            mlflow.log_param("model", model_name)
            
            # Perform grid search
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                scoring='roc_auc',
                cv=3,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred_proba = grid_search.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            # Log metrics
            mlflow.log_metric("best_auc", auc_score)
            mlflow.log_metric("best_params", str(grid_search.best_params_))
            
            # Save model if it's the best so far
            if auc_score > best_score:
                best_score = auc_score
                best_model = grid_search.best_estimator_
                
                # Log the best model
                mlflow.sklearn.log_model(
                    best_model,
                    "best_model",
                    registered_model_name="credit_risk_model"
                )
                
    return best_model, best_score

def train_main():
    """
    Main training function.
    """
    try:
        # Load data
        df = load_processed_data("data/processed/processed_data.csv")
        
        # Prepare features
        X, y = prepare_features(df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        best_model, best_score = train_models(X_train, y_train, X_val, y_val)
        
        logger.info(f"Training complete. Best model AUC: {best_score:.4f}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    train_main()

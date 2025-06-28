"""
Data processing and feature engineering pipeline for credit risk model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFMCalculator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate RFM (Recency, Frequency, Monetary) features.
    """
    
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date if snapshot_date else datetime.now()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """
        Calculate RFM metrics for each customer.
        
        Args:
            X: DataFrame containing transaction data
            
        Returns:
            DataFrame with RFM features
        """
        try:
            # Calculate Recency
            X['Recency'] = (self.snapshot_date - X['TransactionStartTime']).dt.days
            
            # Calculate Frequency and Monetary
            rfm = X.groupby('CustomerId').agg({
                'TransactionId': 'count',
                'Amount': 'sum'
            }).rename(columns={
                'TransactionId': 'Frequency',
                'Amount': 'Monetary'
            })
            
            # Merge RFM metrics back to original data
            X = X.merge(rfm, on='CustomerId', how='left')
            
            # Calculate additional features
            X['AvgTransactionAmount'] = X['Monetary'] / X['Frequency']
            X['TransactionFrequency'] = X['Frequency'] / X['Recency']
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error in RFM calculation: {str(e)}")
            raise

class ProxyTargetGenerator(BaseEstimator, TransformerMixin):
    """
    Generate proxy target variable using K-means clustering on RFM metrics.
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def fit(self, X, y=None):
        from sklearn.cluster import KMeans
        try:
            # Select features for clustering
            clustering_features = X[['Recency', 'Frequency', 'Monetary']]
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(clustering_features)
            
            # Perform K-means clustering
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            self.kmeans.fit(scaled_features)
            
            # Identify high-risk cluster
            cluster_centers = pd.DataFrame(self.kmeans.cluster_centers_, columns=clustering_features.columns)
            self.high_risk_cluster = cluster_centers['Frequency'].idxmin()
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error in fitting proxy target generator: {str(e)}")
            raise
            
    def transform(self, X):
        try:
            # Transform features using the same scaler
            clustering_features = X[['Recency', 'Frequency', 'Monetary']]
            scaled_features = self.kmeans.transform(clustering_features)
            
            # Assign cluster labels
            X['Cluster'] = self.kmeans.predict(scaled_features)
            
            # Create binary target variable
            X['is_high_risk'] = (X['Cluster'] == self.high_risk_cluster).astype(int)
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error in transforming proxy target: {str(e)}")
            raise

def create_data_pipeline():
    """
    Create the complete data processing pipeline.
    
    Returns:
        sklearn Pipeline object
    """
    pipeline = Pipeline([
        ('rfm_calculator', RFMCalculator()),
        ('proxy_target', ProxyTargetGenerator())
    ])
    
    return pipeline

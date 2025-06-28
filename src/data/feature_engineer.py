import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

class FeatureEngineer:
    """
    Class responsible for feature engineering and creation of the proxy target variable.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main method to engineer features and create proxy target variable.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Tuple of (features DataFrame, target DataFrame)
        """
        df = self._calculate_rfm_metrics(df)
        df = self._create_proxy_target(df)
        
        features = self._select_features(df)
        target = df[['is_high_risk']]
        
        return features, target

    def _calculate_rfm_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
        """
        # Define snapshot date as the latest transaction date
        snapshot_date = df['TransactionStartTime'].max()
        
        # Calculate Recency
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum'
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        })
        
        # Calculate additional features
        df = df.merge(rfm, on='CustomerId')
        df['AvgTransactionAmount'] = df['Monetary'] / df['Frequency']
        df['TransactionFrequency'] = df['Frequency'] / df['Recency']
        
        return df

    def _create_proxy_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create proxy target variable using K-Means clustering on RFM metrics.
        """
        from sklearn.cluster import KMeans
        
        # Select features for clustering
        clustering_features = ['Recency', 'Frequency', 'Monetary']
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(df[clustering_features])
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_features)
        
        # Identify high-risk cluster (typically cluster with low frequency and monetary value)
        cluster_stats = df.groupby('Cluster')[['Frequency', 'Monetary']].mean()
        high_risk_cluster = cluster_stats.idxmin().iloc[0]
        
        # Create binary target variable
        df['is_high_risk'] = (df['Cluster'] == high_risk_cluster).astype(int)
        
        return df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant features for modeling.
        """
        features = [
            'Recency', 'Frequency', 'Monetary',
            'AvgTransactionAmount', 'TransactionFrequency',
            'TransactionHour', 'TransactionDay', 'TransactionMonth'
        ]
        
        return df[features]

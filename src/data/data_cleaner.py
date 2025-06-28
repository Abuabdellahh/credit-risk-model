import pandas as pd
from typing import List
import logging
from datetime import datetime

class DataCleaner:
    """
    Class responsible for cleaning and preprocessing the raw data.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to clean the data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = self._handle_missing_values(df)
        df = self._convert_data_types(df)
        df = self._parse_dates(df)
        df = self._create_time_features(df)
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        """
        # Fill missing values with appropriate strategies
        df['Amount'].fillna(0, inplace=True)
        df['TransactionStartTime'].fillna('1900-01-01 00:00:00', inplace=True)
        
        # Drop rows with critical missing values
        df.dropna(subset=['AccountId', 'CustomerId'], inplace=True)
        
        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to appropriate data types.
        """
        type_conversions = {
            'TransactionId': str,
            'AccountId': str,
            'CustomerId': str,
            'Amount': float,
            'Value': float,
            'TransactionStartTime': str
        }
        
        for col, dtype in type_conversions.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and convert date columns.
        """
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from the transaction date.
        """
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        
        return df

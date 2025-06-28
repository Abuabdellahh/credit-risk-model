import pandas as pd
from typing import Union, Tuple
import logging
from pathlib import Path

class DataLoader:
    """
    Class responsible for loading and validating raw data.
    """
    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file and perform initial validation.
        
        Returns:
            pd.DataFrame: Loaded and validated data
        """
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            
            # Basic validation
            self._validate_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Perform basic data validation.
        """
        required_columns = [
            'TransactionId', 'AccountId', 'CustomerId', 'Amount', 
            'TransactionStartTime', 'ProductId', 'ProductCategory'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for duplicates
        if df.duplicated().sum() > 0:
            self.logger.warning("Duplicate rows found in the data")

        # Check for missing values in key columns
        for col in ['TransactionId', 'AccountId', 'CustomerId']:
            if df[col].isnull().sum() > 0:
                self.logger.warning(f"Missing values found in {col}")

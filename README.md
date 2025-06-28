# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops a credit risk probability model for Bati Bank's buy-now-pay-later service partnership with an eCommerce platform. The model uses behavioral transaction data to assess credit risk in the absence of traditional default labels.

## Credit Scoring Business Understanding

### How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord mandates that financial institutions maintain capital reserves proportional to their credit risk exposure. This regulatory framework directly influences our modeling approach in several ways:

**Risk Quantification Requirements:** Basel II requires quantifiable, auditable risk measures that can be directly translated into capital adequacy calculations. Our model must generate probability scores with clear mathematical foundations that regulators can validate.

**Model Validation Standards:** The accord demands comprehensive model validation including backtesting, stress testing, and ongoing performance monitoring. This necessitates interpretable models where decision pathways can be clearly documented and explained to regulatory bodies.

**Transparency and Governance:** Basel II emphasizes transparency in risk assessment processes. Our choice of interpretable algorithms like Logistic Regression with Weight of Evidence (WoE) transformations aligns with regulatory expectations for explainable AI in financial services.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks?

**Business Necessity:**
- **Time-to-Market:** Waiting for sufficient default observations would delay market entry by 12-24 months, missing competitive opportunities
- **Regulatory Compliance:** Basel II requires risk assessment capabilities before loan origination
- **Data-Driven Decision Making:** Enables systematic risk evaluation rather than subjective assessment

**Proxy Development Strategy:**
We use RFM (Recency, Frequency, Monetary) analysis combined with K-means clustering to identify disengaged customers as high-risk proxies, based on the assumption that low engagement patterns correlate with higher default probability.

**Potential Business Risks:**
1. **False Risk Classification:** Misclassifying good customers as high-risk, leading to lost revenue and customer dissatisfaction
2. **Regulatory Scrutiny:** Regulators may question proxy validity, requiring extensive documentation and justification
3. **Model Drift:** Proxy relationships may deteriorate as market conditions change, requiring continuous recalibration
4. **Adverse Selection:** Competitors may capture customers we incorrectly classify as high-risk

### What are the key trade-offs between simple, interpretable models versus complex, high-performance models in a regulated financial context?

**Interpretable Models (Logistic Regression + WoE):**

*Advantages:*
- Regulatory acceptance due to clear coefficient interpretation
- Easy communication of risk factors to stakeholders
- Traceable decision pathways for audit requirements
- Lower overfitting risk with limited proxy data
- Stable performance across different market conditions

*Disadvantages:*
- May miss complex non-linear relationships in data
- Requires extensive manual feature engineering
- Performance ceiling limitations
- May underperform in competitive accuracy metrics

**Complex Models (Gradient Boosting, Neural Networks):**

*Advantages:*
- Superior predictive performance and pattern recognition
- Automatic feature interaction discovery
- Higher ROC-AUC and precision metrics
- Better handling of complex, high-dimensional data

*Disadvantages:*
- "Black box" nature complicates regulatory explanation
- High overfitting risk with limited training data
- Requires specialized expertise for maintenance
- Difficult to validate individual prediction logic
- Potential regulatory rejection due to lack of interpretability

**Recommended Approach:** Implement a hybrid strategy using interpretable models for regulatory reporting and complex models for internal risk assessment, with ensemble methods to balance performance and interpretability.

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # CI/CD pipeline
├── data/                      # Data storage (gitignored)
│   ├── raw/                  # Original transaction data
│   └── processed/            # Cleaned, feature-engineered data
├── notebooks/
│   └── 1.0-eda.ipynb         # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Feature engineering pipeline
│   ├── train.py             # Model training scripts
│   ├── predict.py           # Inference pipeline
│   └── api/
│       ├── main.py          # FastAPI application
│       └── pydantic_models.py # API data models
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Data Description

The dataset contains transaction data from an eCommerce platform with the following key features:
- TransactionId, CustomerId, AccountId: Unique identifiers
- Amount, Value: Transaction monetary values
- TransactionStartTime: Timestamp data
- ProductCategory, ChannelId: Categorical features
- CountryCode, CurrencyCode: Geographic information
- FraudResult: Fraud indicator

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Data Processing
```bash
python src/data_processing.py
```

### Model Training
```bash
python src/train.py
```

### Running Tests
```bash
pytest tests/
```

## Current Progress

- [x] Project structure setup
- [x] Business understanding documentation
- [x] Initial EDA in Jupyter notebook
- [ ] Feature engineering pipeline
- [ ] RFM analysis and clustering
- [ ] Model training and evaluation
- [ ] API development
- [ ] Containerization and deployment

## Next Steps

1. Complete comprehensive EDA
2. Implement RFM-based proxy variable creation
3. Develop feature engineering pipeline
4. Train and evaluate multiple models
5. Set up MLflow for experiment tracking
6. Deploy API with Docker
7. Implement CI/CD pipeline

## References

- Basel II Capital Accord documentation
- Credit Risk Analysis methodologies
- Weight of Evidence and Information Value techniques
```

## 2. Basic EDA Notebook

```python:notebooks/1.0-eda.ipynb
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Credit Risk Data - Exploratory Data Analysis\n",
    "\n",
    "This notebook contains the initial exploratory data analysis for the credit risk modeling project."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from datetime import datetime\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Set plotting style\n",
    "plt.style.use('seaborn-v0_8')\n",
    "sns.set_palette(\"husl\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading and Overview"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "# df = pd.read_csv('../data/raw/transaction_data.csv')\n",
    "# print(f\"Dataset shape: {df.shape}\")\n",
    "# print(f\"\\nColumn names: {list(df.columns)}\")\n",
    "# df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Quality Assessment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for missing values\n",
    "# missing_data = df.isnull().sum()\n",
    "# missing_percent = (missing_data / len(df)) * 100\n",
    "# missing_df = pd.DataFrame({\n",
    "#     'Missing Count': missing_data,\n",
    "#     'Missing Percentage': missing_percent\n",
    "# })\n",
    "# print(\"Missing Data Summary:\")\n",
    "# print(missing_df[missing_df['Missing Count'] > 0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Summary Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Numerical features summary\n",
    "# numerical_cols = df.select_dtypes(include=[np.number]).columns\n",
    "# print(\"Numerical Features Summary:\")\n",
    "# df[numerical_cols].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Distribution Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Transaction amount distribution\n",
    "# plt.figure(figsize=(12, 4))\n",
    "# plt.subplot(1, 2, 1)\n",
    "# plt.hist(df['Amount'], bins=50, alpha=0.7)\n",
    "# plt.title('Transaction Amount Distribution')\n",
    "# plt.xlabel('Amount')\n",
    "# plt.ylabel('Frequency')\n",
    "\n",
    "# plt.subplot(1, 2, 2)\n",
    "# plt.hist(np.log1p(df['Amount']), bins=50, alpha=0.7)\n",
    "# plt.title('Log Transaction Amount Distribution')\n",
    "# plt.xlabel('Log(Amount + 1)')\n",
    "# plt.ylabel('Frequency')\n",
    "\n",
    "# plt.tight_layout()\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Customer Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Customer transaction patterns\n",
    "# customer_stats = df.groupby('CustomerId').agg({\n",
    "#     'TransactionId': 'count',\n",
    "#     'Amount': ['sum', 'mean', 'std'],\n",
    "#     'TransactionStartTime': ['min', 'max']\n",
    "# }).round(2)\n",
    "\n",
    "# customer_stats.columns = ['Transaction_Count', 'Total_Amount', 'Avg_Amount', 'Std_Amount', 'First_Transaction', 'Last_Transaction']\n",
    "# print(f\"Number of unique customers: {len(customer_stats)}\")\n",
    "# print(\"\\nCustomer Statistics Summary:\")\n",
    "# customer_stats.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Key Insights Summary\n",
    "\n",
    "Based on the initial exploration, here are the top insights:\n",
    "\n",
    "1. **Data Quality**: [To be filled after actual data analysis]\n",
    "2. **Customer Behavior**: [To be filled after actual data analysis]\n",
    "3. **Transaction Patterns**: [To be filled after actual data analysis]\n",
    "4. **Risk Indicators**: [To be filled after actual data analysis]\n",
    "5. **Feature Engineering Opportunities**: [To be filled after actual data analysis]\n",
    "\n",
    "## Next Steps\n",
    "\n",
    "1. Implement RFM analysis\n",
    "2. Create customer segmentation\n",
    "3. Develop proxy risk variable\n",
    "4. Feature engineering pipeline\n",
    "5. Model development and training"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

## 3. Requirements.txt

```txt:requirements.txt
# Data manipulation and analysis
pandas>=1.3.0
numpy>=1.21.0

# Machine learning
scikit-learn>=1.0.0
xgboost>=1.5.0

# Data visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Feature engineering
xverse>=0.0.5
woe>=0.1.1

# Model tracking and deployment
mlflow>=1.20.0
fastapi>=0.70.0
uvicorn>=0.15.0
pydantic>=1.8.0

# Testing and code quality
pytest>=6.2.0
flake8>=4.0.0
black>=21.0.0

# Utilities
python-dotenv>=0.19.0
joblib>=1.1.0
```

## 4. Basic Python Files

```python:src/__init__.py
"""
Credit Risk Model Package
"""

__version__ = "0.1.0"
__author__ = "Bati Bank Analytics Team"
```

```python:src/data_processing.py
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

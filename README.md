# Insurance Premium Prediction Model

A machine learning regression model to predict automobile insurance premiums with optimized performance.

## Features

- **Multiple ML Models**: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Best Model**: Lasso Regression with RMSE of 232.42
- **Interactive Dashboard**: Streamlit web application for predictions
- **Data Preprocessing**: Handles missing values and categorical encoding

## Model Performance

| Model | RMSE | R² Score |
|-------|------|----------|
| Lasso | 232.42 | 0.7000 |
| Linear Regression | 232.45 | 0.6999 |
| Ridge | 232.46 | 0.6999 |
| Gradient Boosting | 250.64 | 0.6512 |
| Random Forest | 265.40 | 0.6088 |
| XGBoost | 266.58 | 0.6054 |
| LightGBM | 269.04 | 0.5980 |

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn streamlit xgboost lightgbm
```

## Usage

### Train Models
```bash
python train_model.py
```

### Run Streamlit App
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Files

- `train_model.py` - Model training script
- `app.py` - Streamlit web application
- `challenging_insurance_dataset_regression.csv` - Insurance dataset
- `models/` - Saved models and preprocessing objects

## Features Used

- Vehicle age (years)
- Vehicle type (Hatchback, Sedan, SUV, Truck)
- Kilometers run
- Number of claims
- Region (Urban, Suburban, Rural)

## Target Variable

- Insurance Premium ($)

## Author

Sangeeta S K

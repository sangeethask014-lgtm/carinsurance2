import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import os

# Load data
df = pd.read_csv('challenging_insurance_dataset_regression.csv')

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Data preprocessing
df = df.drop('customer_id', axis=1)

# Convert to numeric types first
df['vehicle_age_years'] = pd.to_numeric(df['vehicle_age_years'], errors='coerce')
df['no_of_kilometers_run'] = pd.to_numeric(df['no_of_kilometers_run'], errors='coerce')
df['number_of_claims'] = pd.to_numeric(df['number_of_claims'], errors='coerce')
df['insurance_premium'] = pd.to_numeric(df['insurance_premium'], errors='coerce')

# Convert to object type for string columns to allow fillna
df['vehicle_type'] = df['vehicle_type'].astype('object')
df['region'] = df['region'].astype('object')

# Handle missing values
df['no_of_kilometers_run'] = df['no_of_kilometers_run'].fillna(df['no_of_kilometers_run'].median())
df['number_of_claims'] = df['number_of_claims'].fillna(df['number_of_claims'].median())
df['vehicle_type'] = df['vehicle_type'].fillna(df['vehicle_type'].mode()[0])
df['region'] = df['region'].fillna(df['region'].mode()[0])

# Encode categorical variables
le_vehicle = LabelEncoder()
le_region = LabelEncoder()
df['vehicle_type'] = le_vehicle.fit_transform(df['vehicle_type'])
df['region'] = le_region.fit_transform(df['region'])

# Prepare features and target
X = df.drop('insurance_premium', axis=1)
y = df['insurance_premium']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
}

results = {}
best_model_name = None
best_rmse = float('inf')

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

for name, model in models.items():
    # Train
    if name in ['Linear Regression', 'Ridge', 'Lasso']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'rmse': rmse, 'r2': r2, 'model': model}
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = name

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name} (RMSE: {best_rmse:.4f})")
print("="*60)

# Save best model and preprocessing objects
os.makedirs('models', exist_ok=True)

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(results[best_model_name]['model'], f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump({'vehicle_type': le_vehicle, 'region': le_region}, f)

# Save results to file
import json
results_json = {name: {'rmse': float(res['rmse']), 'r2': float(res['r2'])} for name, res in results.items()}
with open('models/results.json', 'w') as f:
    json.dump(results_json, f, indent=2)

print("\nModels saved to 'models/' directory")

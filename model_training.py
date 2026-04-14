import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('challenging_insurance_dataset_regression.csv')

# Data cleaning
df_clean = df.copy()

# Fill missing values in no_of_kilometers_run with median
df_clean['no_of_kilometers_run'].fillna(df_clean['no_of_kilometers_run'].median(), inplace=True)

# Fill missing values in number_of_claims with median
df_clean['number_of_claims'].fillna(df_clean['number_of_claims'].median(), inplace=True)

# Fill missing vehicle_type with mode
df_clean['vehicle_type'].fillna(df_clean['vehicle_type'].mode()[0], inplace=True)

# Fill missing region with mode
df_clean['region'].fillna(df_clean['region'].mode()[0], inplace=True)

# Remove outliers in no_of_kilometers_run (negative values)
df_clean = df_clean[df_clean['no_of_kilometers_run'] >= 0]

# Encode categorical variables
le_vehicle = LabelEncoder()
le_region = LabelEncoder()

df_clean['vehicle_type'] = le_vehicle.fit_transform(df_clean['vehicle_type'])
df_clean['region'] = le_region.fit_transform(df_clean['region'])

# Prepare features and target
X = df_clean[['vehicle_age_years', 'vehicle_type', 'no_of_kilometers_run', 
               'number_of_claims', 'region']]
y = df_clean['insurance_premium']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)

# Train LightGBM model
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    random_state=42,
    verbosity=-1
)
lgb_model.fit(X_train, y_train)

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Evaluate models
print("=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

# XGBoost
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"\nXGBoost Model:")
print(f"  RMSE: {rmse_xgb:.4f}")
print(f"  R² Score: {r2_xgb:.4f}")

# LightGBM
y_pred_lgb = lgb_model.predict(X_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
r2_lgb = r2_score(y_test, y_pred_lgb)
print(f"\nLightGBM Model:")
print(f"  RMSE: {rmse_lgb:.4f}")
print(f"  R² Score: {r2_lgb:.4f}")

# Gradient Boosting
y_pred_gb = gb_model.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)
print(f"\nGradient Boosting Model:")
print(f"  RMSE: {rmse_gb:.4f}")
print(f"  R² Score: {r2_gb:.4f}")

# Select best model (lowest RMSE)
models = {
    'XGBoost': (xgb_model, rmse_xgb),
    'LightGBM': (lgb_model, rmse_lgb),
    'GradientBoosting': (gb_model, rmse_gb)
}

best_model_name = min(models, key=lambda x: models[x][1])
best_model = models[best_model_name][0]
best_rmse = models[best_model_name][1]

print(f"\n{'=' * 60}")
print(f"BEST MODEL: {best_model_name}")
print(f"Best RMSE: {best_rmse:.4f}")
print(f"{'=' * 60}")

# Save models and preprocessors
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('le_vehicle.pkl', 'wb') as f:
    pickle.dump(le_vehicle, f)

with open('le_region.pkl', 'wb') as f:
    pickle.dump(le_region, f)

print("\nModels saved successfully!")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Page config
st.set_page_config(page_title="Insurance Premium Predictor", layout="wide")

# Load models and preprocessing objects
@st.cache_resource
def train_and_load_models():
    """Train models if they don't exist, otherwise load them"""
    models_dir = 'models'
    
    # Check if models exist
    if (os.path.exists(f'{models_dir}/best_model.pkl') and 
        os.path.exists(f'{models_dir}/scaler.pkl') and 
        os.path.exists(f'{models_dir}/label_encoders.pkl') and 
        os.path.exists(f'{models_dir}/results.json')):
        
        # Load existing models
        with open(f'{models_dir}/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{models_dir}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'{models_dir}/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open(f'{models_dir}/results.json', 'r') as f:
            results = json.load(f)
        return model, scaler, encoders, results
    
    # If models don't exist, train them
    st.info("Training models for the first time. This may take a moment...")
    
    # Load and preprocess data
    df = pd.read_csv('challenging_insurance_dataset_regression.csv')
    df = df.drop('customer_id', axis=1)
    
    # Convert to numeric types
    df['vehicle_age_years'] = pd.to_numeric(df['vehicle_age_years'], errors='coerce')
    df['no_of_kilometers_run'] = pd.to_numeric(df['no_of_kilometers_run'], errors='coerce')
    df['number_of_claims'] = pd.to_numeric(df['number_of_claims'], errors='coerce')
    df['insurance_premium'] = pd.to_numeric(df['insurance_premium'], errors='coerce')
    
    # Convert to object type for string columns
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
    
    # Train models
    models = {
        'Linear Regression': __import__('sklearn.linear_model', fromlist=['LinearRegression']).LinearRegression(),
        'Ridge': __import__('sklearn.linear_model', fromlist=['Ridge']).Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    
    results = {}
    best_model = None
    best_rmse = float('inf')
    best_model_name = None
    
    for name, model_obj in models.items():
        # Train
        if name in ['Linear Regression', 'Ridge', 'Lasso']:
            model_obj.fit(X_train_scaled, y_train)
            y_pred = model_obj.predict(X_test_scaled)
        else:
            model_obj.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'rmse': rmse, 'r2': r2}
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model_obj
            best_model_name = name
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save best model and preprocessing objects
    with open(f'{models_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(f'{models_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(f'{models_dir}/label_encoders.pkl', 'wb') as f:
        pickle.dump({'vehicle_type': le_vehicle, 'region': le_region}, f)
    
    results_json = {name: {'rmse': float(res['rmse']), 'r2': float(res['r2'])} for name, res in results.items()}
    with open(f'{models_dir}/results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    return best_model, scaler, {'vehicle_type': le_vehicle, 'region': le_region}, results_json

model, scaler, encoders, results = train_and_load_models()

# Title and description
st.title("🚗 Insurance Premium Predictor")
st.markdown("Predict automobile insurance premiums using optimized machine learning models")
st.markdown("---")

# Display model comparison
st.subheader("📊 Model Performance Comparison")
results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ['Model', 'RMSE', 'R² Score']
results_df = results_df.sort_values('RMSE')

col1, col2 = st.columns(2)

with col1:
    st.dataframe(results_df, use_container_width=True, hide_index=True)

with col2:
    best_model_name = results_df.iloc[0]['Model']
    best_rmse = results_df.iloc[0]['RMSE']
    best_r2 = results_df.iloc[0]['R² Score']
    
    st.metric("🏆 Best Model", best_model_name)
    st.metric("📉 Best RMSE", f"{best_rmse:.4f}")
    st.metric("📈 R² Score", f"{best_r2:.4f}")

st.markdown("---")

# Prediction section
st.subheader("🔮 Make a Prediction")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle & Claims Information")
    
    vehicle_age = st.slider(
        "Vehicle Age (years)",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.1
    )
    
    vehicle_type = st.selectbox(
        "Vehicle Type",
        options=encoders['vehicle_type'].classes_,
        index=0
    )
    
    kilometers = st.number_input(
        "Kilometers Run",
        min_value=0,
        max_value=200000,
        value=50000,
        step=1000
    )

with col2:
    st.subheader("Region Information")
    
    claims = st.slider(
        "Number of Claims",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5
    )
    
    region = st.selectbox(
        "Region",
        options=encoders['region'].classes_,
        index=0
    )

st.markdown("---")

# Prediction button
if st.button("🎯 Predict Premium", use_container_width=True):
    # Prepare input data
    vehicle_type_encoded = encoders['vehicle_type'].transform([vehicle_type])[0]
    region_encoded = encoders['region'].transform([region])[0]
    
    input_array = np.array([[vehicle_age, vehicle_type_encoded, kilometers, claims, region_encoded]])
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Display result
    st.markdown("---")
    st.success(f"✅ **Predicted Insurance Premium: ${prediction:.2f}**")
    
    col_result1, col_result2, col_result3 = st.columns(3)
    with col_result1:
        st.metric("Vehicle Age", f"{vehicle_age} years")
    with col_result2:
        st.metric("Type", vehicle_type)
    with col_result3:
        st.metric("Region", region)

# Sidebar with information
with st.sidebar:
    st.subheader("📊 Model Information")
    st.markdown("""
    **Training Details:**
    - Multiple advanced models trained
    - Best model selected based on lowest RMSE
    - Models: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM
    
    **Features Used:**
    - Vehicle age (years)
    - Vehicle type (categorical)
    - Kilometers run
    - Number of claims
    - Region (categorical)
    
    **Data Preprocessing:**
    - Missing values imputed with median/mode
    - Categorical variables encoded
    - Features standardized
    """)
    
    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Adjust vehicle information sliders
    2. Select vehicle type
    3. Enter kilometers run
    4. Set number of claims
    5. Choose region
    6. Click "Predict Premium"
    """)

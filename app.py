import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

# Page config
st.set_page_config(page_title="Insurance Premium Predictor", layout="wide")

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/results.json', 'r') as f:
        results = json.load(f)
    return model, scaler, encoders, results

model, scaler, encoders, results = load_models()

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

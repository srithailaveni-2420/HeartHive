import streamlit as st
import pandas as pd
import numpy as np
import torch
from model.transformer import TabularTransformer
from sklearn.preprocessing import StandardScaler
from explainability.shap_explainer import generate_shap_values
from recommendations.generator import generate_recommendations

# Load the model (this should be done once and cached for efficiency)
@st.cache_resource
def load_model():
    # Assuming you saved your trained model and scaler
    model = TabularTransformer(input_dim=12, embed_dim=32, num_heads=2, ff_dim=64, num_layers=1, dropout=0.1)
    model.load_state_dict(torch.load("heart_model.pth"))
    model.eval()
    
    scaler = StandardScaler()  # Assuming you've saved the scaler
    return model, scaler

# Load the model and scaler
model, scaler = load_model()

# Streamlit UI
st.title("HeartHive: Cardiovascular Disease Prediction")
st.write(
    "Enter your health data below and let HeartHive predict your risk of cardiovascular disease."
)

# User input
age = st.slider("Age", min_value=20, max_value=80, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (in cm)", min_value=100, max_value=220, value=170)
weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, value=70)

# Blood Pressure
ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=80, max_value=200, value=120)
ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=40, max_value=150, value=80)

# Lifestyle Indicators
cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3], index=0)
gluc = st.selectbox("Glucose Level", [1, 2, 3], index=0)
smoke = st.selectbox("Smoking", [0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
alco = st.selectbox("Alcohol Consumption", [0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
active = st.selectbox("Physical Activity", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")

# Prepare input data for prediction
input_data = {
    "age": [age],
    "gender": [1 if gender == "Male" else 0],
    "height": [height],
    "weight": [weight],
    "ap_hi": [ap_hi],
    "ap_lo": [ap_lo],
    "cholesterol": [cholesterol],
    "gluc": [gluc],
    "smoke": [smoke],
    "alco": [alco],
    "active": [active]
}
input_df = pd.DataFrame(input_data)

# Scale the input data
scaled_input = scaler.transform(input_df)

# Convert to torch tensor
input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

# Make prediction
with torch.no_grad():
    prediction = model(input_tensor).item()

# Display results
st.subheader("Prediction")
if prediction > 0.5:
    st.write("⚠️ High risk of cardiovascular disease!")
else:
    st.write("✅ Low risk of cardiovascular disease.")

# Explainability (SHAP)
st.subheader("Model Explanation")

# Generate SHAP values
shap_values = generate_shap_values(model, input_tensor)

# Visualize SHAP summary
import shap
shap.summary_plot(shap_values, input_df, feature_names=input_df.columns.tolist())

# Generate personalized recommendations based on SHAP
recommendations = generate_recommendations(0, shap_values, input_df)
st.subheader("Personalized Health Recommendations")
for rec in recommendations:
    st.write(f"👉 {rec}")

# Option to download explanation
shap_file = shap.summary_plot(shap_values, input_df, feature_names=input_df.columns.tolist(), show=False)
st.download_button("Download SHAP Summary", shap_file, file_name="shap_summary.png")

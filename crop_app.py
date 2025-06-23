import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('crop_model.pkl', 'rb'))

st.title("🌱 Crop Recommendation System")

st.markdown("Enter your soil and climate parameters below:")

# Input fields with default values
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=40)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=40)
temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.slider("Soil pH", 3.0, 10.0, 6.5)
rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 200.0)

# Prediction
if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")

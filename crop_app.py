import streamlit as st
import numpy as np
import pickle
import os

# ✅ SAFELY load the trained model
model_path = 'crop_model.pkl'

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please make sure crop_model.pkl is uploaded correctly.")
else:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

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

    # Predict when button clicked
    if st.button("Recommend Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        st.success(f"✅ Recommended Crop: **{prediction[0]}**")



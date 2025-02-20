import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model_path = "C:/Desktop/PRACTICE/LONDON WEATHER/temperature_predictor.pkl"
model = joblib.load(model_path)

# Streamlit UI
st.title("🌤️ London Weather Temperature Predictor")
st.write("Enter weather details to predict the mean temperature.")

# User input fields
cloud_cover = st.number_input("Cloud Cover (oktas)", min_value=0.0, max_value=8.0, value=3.0)
sunshine = st.number_input("Sunshine (hrs)", min_value=0.0, max_value=24.0, value=5.0)
global_radiation = st.number_input("Global Radiation (W/m²)", min_value=0.0, max_value=500.0, value=120.0)
max_temp = st.number_input("Max Temperature (°C)", min_value=-10.0, max_value=40.0, value=15.0)
min_temp = st.number_input("Min Temperature (°C)", min_value=-10.0, max_value=40.0, value=7.0)
precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.5)
pressure = st.number_input("Pressure (Pa)", min_value=900.0, max_value=1100.0, value=1010.0)

# Prediction button
if st.button("Predict Temperature"):
    # Prepare input for the model
    input_features = np.array([[cloud_cover, sunshine, global_radiation, max_temp, min_temp, precipitation, pressure]])
    input_df = pd.DataFrame(input_features, columns=['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'min_temp', 'precipitation', 'pressure'])
    prediction = model.predict(input_df)[0]  # Extract single prediction
    
    st.success(f"Predicted Mean Temperature: {prediction:.2f}°C")


# How This Works
# Loads the trained model from a .pkl file using pickle.load().
# Creates a Streamlit UI with sliders for weather conditions.
# Takes user inputs and converts them into a NumPy array ((1, 7) shape).
# Uses the trained model to predict the mean temperature.
# Displays the prediction using st.success().
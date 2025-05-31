import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("random_forest_model.pkl")
encoders = joblib.load("encoders.pkl")

def main():
    st.title("Predict Filling Time using Random Forest")

# User inputs
    shape = st.selectbox("Solder Ball Shape", encoders['Solder Ball shapes'].classes_)
    method = st.selectbox("Dispensing Method", encoders['Dispensing methods'].classes_)
    material = st.selectbox("Underfill Material", encoders['Underfill Material'].classes_)
    viscosity = st.slider("Viscosity (Pa.s)", 0.001, 1.0, 0.5)
    surface_tension = st.slider("Surface Tension (N/m)", 0.01, 0.1, 0.05)
    density = st.slider("Density (kg/m³)", 500, 2000, 1000)

    # Encode categorical inputs
    shape_encoded = encoders['Solder Ball shapes'].transform([shape])[0]
    method_encoded = encoders['Dispensing methods'].transform([method])[0]
    material_encoded = encoders['Underfill Material'].transform([material])[0]

    # Prepare features
    features = np.array([[shape_encoded, method_encoded, material_encoded, viscosity, surface_tension, density]])

    # Predict
    if st.button("Predict Filling Time"):
        prediction = model.predict(features)[0]
        st.success(f"Predicted Filling Time: {prediction:.2f} seconds")


if __name__ == "__main__":
    main()

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
    viscosity = st.number_input("Viscosity (Pa.s)")
    surface_tension = st.number_input("Surface Tension (N/m)")
    density = st.number_input("Density (kg/m³)")

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

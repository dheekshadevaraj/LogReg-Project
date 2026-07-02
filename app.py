import streamlit as st
import pickle
import numpy as np

# Load model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Diabetes Prediction")

st.title("🩺 Diabetes Prediction using Logistic Regression")

st.write("Enter the patient's medical information.")

pregnancies = st.number_input("Pregnancies", min_value=0, value=2)

glucose = st.number_input("Glucose", min_value=0, value=120)

blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, value=70)

skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, value=30)

insulin = st.number_input("Insulin", min_value=0, value=85)

bmi = st.number_input("BMI", min_value=0.0, value=32.5)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.45,
    format="%.2f"
)

age = st.number_input("Age", min_value=1, value=28)

if st.button("Predict"):

    features = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    probability = model.predict_proba(features_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Not Diabetic")

    st.write(f"Probability of Diabetes: **{probability[0][1]*100:.2f}%**")

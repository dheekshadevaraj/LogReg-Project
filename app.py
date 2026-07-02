import streamlit as st
import pickle

# Load model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Diabetes Progression Prediction")

st.title("🩺 Diabetes Progression Prediction")

st.write(
    "Predict the diabetes disease progression score based on patient features."
)

st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", -0.2, 0.2, 0.0)
sex = st.sidebar.slider("Sex", -0.2, 0.2, 0.0)
bmi = st.sidebar.slider("BMI", -0.2, 0.2, 0.0)
bp = st.sidebar.slider("Blood Pressure", -0.2, 0.2, 0.0)
s1 = st.sidebar.slider("S1", -0.2, 0.2, 0.0)
s2 = st.sidebar.slider("S2", -0.2, 0.2, 0.0)
s3 = st.sidebar.slider("S3", -0.2, 0.2, 0.0)
s4 = st.sidebar.slider("S4", -0.2, 0.2, 0.0)
s5 = st.sidebar.slider("S5", -0.2, 0.2, 0.0)
s6 = st.sidebar.slider("S6", -0.2, 0.2, 0.0)

if st.button("Predict"):

    prediction = model.predict([[
        age, sex, bmi, bp,
        s1, s2, s3, s4, s5, s6
    ]])

    st.success(
        f"Predicted Disease Progression Score: {prediction[0]:.2f}"
    )
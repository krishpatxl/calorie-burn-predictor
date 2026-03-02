import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Calorie Burn Predictor", page_icon="🔥")

st.title("Calorie Burn Predictor")
st.write("Enter your workout details below to estimate calories burned.")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["male", "female"])
    age = st.number_input("Age", min_value=1, max_value=120, value=21)
    height = st.number_input("Height (cm)", min_value=1.0, value=173.0)
    weight = st.number_input("Weight (kg)", min_value=1.0, value=75.0)
    duration = st.number_input("Duration (minutes)", min_value=1.0, value=30.0)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=1.0, value=140.0)
    body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, value=37.0)

    submitted = st.form_submit_button("Predict Calories")

if submitted:
    payload = {
        "gender": gender,
        "age": int(age),
        "height": float(height),
        "weight": float(weight),
        "duration": float(duration),
        "heart_rate": float(heart_rate),
        "body_temp": float(body_temp),
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        st.success(f"Estimated Calories Burned: {result['calories']} kcal")
    except Exception as e:
        st.error("Could not connect to API. Make sure FastAPI is running.")
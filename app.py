import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

st.title("Predictive Maintenance")
st.write("Predict engine condition using sensor measurements.")

# Load trained model
model = joblib.load("best_model.joblib")

engine_rpm = st.number_input("Engine RPM", value=1000.0)
lub_oil_pressure = st.number_input("Lub Oil Pressure", value=3.0)
fuel_pressure = st.number_input("Fuel Pressure", value=6.0)
coolant_pressure = st.number_input("Coolant Pressure", value=2.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", value=75.0)
coolant_temp = st.number_input("Coolant Temperature", value=80.0)

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Engine rpm": engine_rpm,
        "Lub oil pressure": lub_oil_pressure,
        "Fuel pressure": fuel_pressure,
        "Coolant pressure": coolant_pressure,
        "lub oil temp": lub_oil_temp,
        "Coolant temp": coolant_temp
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.warning("Engine condition predicted as Class 1")
    else:
        st.success("Engine condition predicted as Class 0")
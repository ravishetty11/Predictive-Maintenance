import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Predictive Engine Maintenance",
    page_icon="⚙  ",
    layout="centered"
)

# Title and description
st.title("⚙  Predictive Maintenance Classifier")
st.write("""
This interactive engine health assessment tool uses machine learning to predict potential failure conditions 
based on real-time sensor measurements.
""")

# Path to model
MODEL_PATH = "best_model.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

model = load_model()

if model is None:
    st.error(f"Model file '{MODEL_PATH}' was not found. Please ensure it is present in your repository.")
else:
    st.success("Model loaded successfully!")

    st.header("Input Engine Sensor Readings")
    
    # Generate user inputs with default values reflecting standard averages from EDA
    engine_rpm = st.number_input("Engine rpm", min_value=0, max_value=5000, value=790)
    lub_oil_press = st.number_input("Lub oil pressure", min_value=0.0, max_value=15.0, value=3.3, format="%.2f")
    fuel_press = st.number_input("Fuel pressure", min_value=0.0, max_value=30.0, value=6.6, format="%.2f")
    coolant_press = st.number_input("Coolant pressure", min_value=0.0, max_value=15.0, value=2.3, format="%.2f")
    lub_oil_temp = st.number_input("lub oil temp", min_value=0.0, max_value=150.0, value=77.6, format="%.2f")
    coolant_temp = st.number_input("Coolant temp", min_value=0.0, max_value=250.0, value=78.4, format="%.2f")

    # Convert inputs to a pandas DataFrame
    input_data = pd.DataFrame([{
        "Engine rpm": engine_rpm,
        "Lub oil pressure": lub_oil_press,
        "Fuel pressure": fuel_press,
        "Coolant pressure": coolant_press,
        "lub oil temp": lub_oil_temp,
        "Coolant temp": coolant_temp
    }])

    st.subheader("Captured Input Vector")
    st.dataframe(input_data)

    # Make real-time prediction
    if st.button("Predict Engine Condition"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        st.subheader("Prediction Results")
        if prediction == 1:
            st.error("☑  Engine Condition flagged as: ABNORMAL (Maintenance Recommended)")
            st.metric("Fault Probability", f"{prediction_proba[1]:.2%}")
        else:
            st.success("✅ Engine Condition flagged as: NORMAL (Healthy Operations)")
            st.metric("Normal Probability", f"{prediction_proba[0]:.2%}")

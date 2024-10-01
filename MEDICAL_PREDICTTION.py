import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('medical_cases_model.joblib')

# Title of the app
st.title("Medical Cases Prediction")

# Create tabs
tabs = st.tabs(["Predict", "About"])

# Prediction Tab
with tabs[0]:
    st.header("Predict Total Patients")

    # Input fields for the features
    surgical = st.number_input("Surgical Cases:", min_value=0, max_value=100)
    obs_gyn = st.number_input("OBS/GYN Cases:", min_value=0, max_value=100)
    neonate = st.number_input("Neonate Cases:", min_value=0, max_value=100)
    pediatrics = st.number_input("Pediatrics Cases:", min_value=0, max_value=100)
    covid_19 = st.number_input("COVID-19 Cases:", min_value=0, max_value=100)
    radiology = st.number_input("Radiology Cases:", min_value=0, max_value=100)
    rta = st.number_input("RTA Cases:", min_value=0, max_value=100)
    ph = st.number_input("PH Cases:", min_value=0, max_value=100)

    # Button to predict
    if st.button("Predict Total Patients"):
        # Prepare the input for prediction
        input_features = np.array([[surgical, obs_gyn, neonate, pediatrics, covid_19, radiology, rta, ph]])
        
        # Make prediction
        predicted_patients = model.predict(input_features)
        
        # Round the prediction for practical use
        rounded_prediction = np.round(predicted_patients[0])

        # Display the result
        st.success(f"Predicted Total Patients: {predicted_patients[0]:.2f}")
        st.info(f"For practical purposes, the predicted total patients can be rounded off to a whole number: **{int(rounded_prediction)}**.")

# About Tab
with tabs[1]:
    st.header("About This App")
    st.write("""
        This application predicts the total number of patients based on various medical case inputs. 
        It is designed to assist healthcare professionals in estimating patient load based on 
        specific medical case categories.
    """)
    st.write("### Developer Information")
    st.write("""
        - **Name**: [Marquline Opiyo]
        - **Email**: [marqulyneo@gmail.com]
    """)
    st.write("""
        Feel free to reach out for any inquiries or feedback regarding the application. 
        Your contributions and suggestions are highly appreciated!
    """)


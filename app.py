import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
pipeline = pickle.load(open("loan_pipeline.pkl", "rb"))

st.set_page_config(page_title="Loan Approval Predictor")
st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to predict loan approval")

# Collect user input dynamically
input_data = {}

for feature in pipeline.feature_names_in_:
    input_data[feature] = st.number_input(f"{feature}")

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = pipeline.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

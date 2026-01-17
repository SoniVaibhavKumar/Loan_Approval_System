import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("loan_model.pkl", "rb"))

st.set_page_config(page_title="Loan Approval Predictor")

st.title("🏦 Loan Approval Prediction System")

st.write("Enter applicant details to predict loan approval")

loan_amount = st.number_input("Loan Amount")
income = st.number_input("Applicant Income")
credit_score = st.number_input("Credit Score")
loan_term = st.number_input("Loan Term (months)")

if st.button("Predict"):
    input_data = np.array([[loan_amount, income, credit_score, loan_term]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

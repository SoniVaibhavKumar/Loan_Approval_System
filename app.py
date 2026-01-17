import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
pipeline = pickle.load(open("loan_pipeline.pkl", "rb"))

st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.write(
    "Enter applicant details below. "
    "The model will predict whether the loan is **Approved or Rejected**."
)

st.divider()

# Helper for binary inputs
def binary_input(label):
    return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0

input_data = {}

# Automatically generate inputs based on trained features
for feature in pipeline.feature_names_in_:
    feature_lower = feature.lower()

    if feature_lower in ["gender", "married", "education", "self_employed", "credit_history"]:
        input_data[feature] = binary_input(feature)

    elif "dependents" in feature_lower:
        input_data[feature] = st.number_input(feature, min_value=0, step=1)

    else:
        input_data[feature] = st.number_input(feature, min_value=0.0)

st.divider()

if st.button("🔍 Predict Loan Status"):
    input_df = pd.DataFrame([input_data])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ **Loan Approved**\n\nConfidence: **{probability:.2%}**")
    else:
        st.error(f"❌ **Loan Rejected**\n\nConfidence: **{probability:.2%}**")

st.caption("Built by Vaibhav Soni")

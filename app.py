import streamlit as st
import pickle
import pandas as pd

# ---------------- LOAD PIPELINE ----------------
pipeline = pickle.load(open("loan_pipeline.pkl", "rb"))
FEATURES = list(pipeline.feature_names_in_)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="wide"
)

# ---------------- THEME TOGGLE ----------------
with st.sidebar:
    st.title("⚙️ Settings")
    theme = st.radio("Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- HEADER ----------------
st.title("🏦 Loan Approval Prediction System")
st.write(
    "A machine learning–powered system to predict whether a loan will be **Approved or Rejected**."
)
st.divider()

# ---------------- INPUT SECTIONS ----------------
st.subheader("👤 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])
    loan_term = st.selectbox("Loan Term (Months)", [120, 180, 240, 300, 360])

st.subheader("💰 Financial Details")

col3, col4 = st.columns(2)

with col3:
    applicant_income = st.slider("Applicant Income", 0, 100000, 5000, step=500)
    coapplicant_income = st.slider("Co-applicant Income", 0, 100000, 0, step=500)

with col4:
    loan_amount = st.slider("Loan Amount", 0, 500000, 100000, step=5000)

st.divider()

# ---------------- BUILD SAFE INPUT ----------------
# Start with ALL features set to 0
input_dict = {feature: 0 for feature in FEATURES}

# Overwrite known features (must match training column names)
input_dict.update({
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": dependents,
    "Education": 1 if education == "Graduate" else 0,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": 1 if credit_history == "Good" else 0
})

input_df = pd.DataFrame([input_dict])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Loan Status"):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.success(f"✅ **Loan Approved**\n\nConfidence: **{probability:.2%}**")
    else:
        st.error(f"❌ **Loan Rejected**\n\nConfidence: **{probability:.2%}**")

# ---------------- FOOTER ----------------
st.caption("Built by Vaibhav Soni")

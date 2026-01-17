import streamlit as st
import pickle
import pandas as pd

# Load trained ML pipeline
pipeline = pickle.load(open("loan_pipeline.pkl", "rb"))

st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="centered"
)

# ---------------- HEADER ----------------
st.title("🏦 Loan Approval Prediction System")
st.write(
    "This application predicts whether a loan is likely to be **Approved or Rejected** "
    "based on applicant information."
)

st.divider()

# ---------------- HELPERS ----------------
def yes_no(label, help_text=""):
    return 1 if st.selectbox(label, ["No", "Yes"], help=help_text) == "Yes" else 0

# ---------------- SECTION 1 ----------------
st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    gender = yes_no("Male", "Select Yes for Male, No for Female")
    married = yes_no("Married", "Applicant marital status")
    education = yes_no("Graduate", "Yes if applicant is a graduate")

with col2:
    dependents = st.selectbox(
        "Number of Dependents",
        [0, 1, 2, 3],
        help="Number of dependents supported by the applicant"
    )
    self_employed = yes_no(
        "Self Employed",
        "Is the applicant self-employed?"
    )

st.divider()

# ---------------- SECTION 2 ----------------
st.subheader("💰 Financial Information")

col3, col4 = st.columns(2)

with col3:
    applicant_income = st.slider(
        "Applicant Monthly Income",
        min_value=0,
        max_value=100000,
        value=5000,
        step=500
    )

    loan_amount = st.slider(
        "Loan Amount Requested",
        min_value=0,
        max_value=500000,
        value=100000,
        step=5000
    )

with col4:
    coapplicant_income = st.slider(
        "Co-Applicant Monthly Income",
        min_value=0,
        max_value=100000,
        value=0,
        step=500
    )

    loan_term = st.selectbox(
        "Loan Term (Months)",
        [120, 180, 240, 300, 360],
        index=4
    )

credit_history = yes_no(
    "Good Credit History",
    "Yes if applicant has good credit repayment history"
)

st.divider()

# ---------------- INPUT DATAFRAME ----------------
input_data = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history
}

input_df = pd.DataFrame([input_data])

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

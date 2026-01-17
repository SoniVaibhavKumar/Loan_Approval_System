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

# ---------------- THEME TOGGLE (REAL FIX) ----------------
with st.sidebar:
    st.title("⚙️ Settings")
    theme = st.radio("Theme Mode", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        div[data-testid="stSidebar"] {
            background-color: #161b22;
        }
        label, span, p, h1, h2, h3 {
            color: #fafafa !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("## 🏦 Loan Approval Prediction System")
st.markdown(
    "A **machine learning–powered system** that predicts whether a loan will be "
    "**Approved or Rejected** based on applicant details."
)
st.divider()

# ---------------- INPUT CARD ----------------
st.markdown("### 👤 Applicant Profile")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])

with col2:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])

with col3:
    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700,
        help="Higher credit score improves approval chances"
    )
    loan_term = st.selectbox(
        "Loan Term (Months)",
        [120, 180, 240, 300, 360]
    )

st.divider()

# ---------------- FINANCIAL DETAILS ----------------
st.markdown("### 💰 Financial Information")

col4, col5, col6 = st.columns(3)

with col4:
    applicant_income = st.slider(
        "Applicant Monthly Income",
        0, 150000, 5000, step=500
    )

with col5:
    coapplicant_income = st.slider(
        "Co-Applicant Monthly Income",
        0, 150000, 0, step=500
    )

with col6:
    loan_amount = st.slider(
        "Loan Amount Requested",
        0, 500000, 100000, step=5000
    )

st.divider()

# ---------------- SAFE FEATURE BUILD ----------------
# Start with ALL features = 0
input_dict = {feature: 0 for feature in FEATURES}

# Map UI → Model features (must match training)
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
st.markdown("### 🔮 Prediction")

if st.button("🚀 Predict Loan Status", use_container_width=True):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ **Loan Approved**\n\nConfidence: **{probability:.2%}**")
    else:
        st.error(f"❌ **Loan Rejected**\n\nConfidence: **{probability:.2%}**")

st.divider()
st.caption("Built by Vaibhav Soni")

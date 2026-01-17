import streamlit as st
import pickle
import pandas as pd

# ================= LOAD MODEL =================
pipeline = pickle.load(open("loan_pipeline.pkl", "rb"))
FEATURES = list(pipeline.feature_names_in_)

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Loan Approval System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= THEME =================
with st.sidebar:
    st.title("⚙️ Settings")
    theme = st.radio("Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {
            background-color: #0b0f19;
            color: #ffffff;
        }
        div[data-testid="stSidebar"] {
            background-color: #111827;
        }
        label, p, h1, h2, h3, span {
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("## 🏦 Loan Approval Prediction System")
st.markdown(
    "An **AI-powered financial decision system** that evaluates loan eligibility "
    "based on applicant profile, income stability, and credit behavior."
)
st.divider()

# ================= APPLICANT PROFILE =================
st.markdown("### 👤 Applicant Profile")

c1, c2, c3, c4 = st.columns(4)

with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 65, 30)

with c2:
    married = st.selectbox("Marital Status", ["Married", "Single"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])

with c3:
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self Employed"])

with c4:
    residence_type = st.selectbox("Residence Type", ["Owned", "Rented", "Company Provided"])
    years_at_job = st.slider("Years at Current Job", 0, 40, 5)

st.divider()

# ================= FINANCIAL DETAILS =================
st.markdown("### 💰 Financial Details")

c5, c6, c7, c8 = st.columns(4)

with c5:
    applicant_income = st.slider("Applicant Monthly Income", 0, 200000, 5000, step=500)
    savings = st.slider("Savings Balance", 0, 1000000, 50000, step=10000)

with c6:
    coapplicant_income = st.slider("Co-Applicant Income", 0, 200000, 0, step=500)
    existing_loans = st.slider("Existing Loans (Count)", 0, 5, 0)

with c7:
    loan_amount = st.slider("Loan Amount Requested", 0, 1000000, 100000, step=10000)
    loan_term = st.selectbox("Loan Term (Months)", [120, 180, 240, 300, 360])

with c8:
    credit_score = st.slider("Credit Score", 300, 900, 700)
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])

st.divider()

# ================= ADDITIONAL RISK SIGNALS =================
st.markdown("### 📊 Additional Risk Indicators")

c9, c10, c11, c12 = st.columns(4)

with c9:
    property_owned = st.selectbox("Property Owned", ["Yes", "No"])
    vehicle_owned = st.selectbox("Vehicle Owned", ["Yes", "No"])

with c10:
    monthly_expenses = st.slider("Monthly Expenses", 0, 150000, 3000, step=500)
    dependents_income = st.selectbox("Dependents with Income", ["Yes", "No"])

with c11:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Home", "Education", "Business", "Personal", "Vehicle"]
    )
    employment_stability = st.selectbox("Employment Stability", ["High", "Medium", "Low"])

with c12:
    region_type = st.selectbox("Region Type", ["Urban", "Semi-Urban", "Rural"])
    bank_relationship = st.slider("Years with Bank", 0, 30, 5)

st.divider()

# ================= SAFE MODEL INPUT =================
# Start with all features as zero
input_dict = {feature: 0 for feature in FEATURES}

# Map only features used by model
input_dict.update({
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Married" else 0,
    "Dependents": dependents,
    "Education": 1 if education == "Graduate" else 0,
    "Self_Employed": 1 if employment_type == "Self Employed" else 0,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": 1 if credit_history == "Good" else 0
})

input_df = pd.DataFrame([input_dict])

# ================= PREDICTION =================
st.markdown("### 🔮 Loan Decision")

if st.button("🚀 Predict Loan Approval", use_container_width=True):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ **Loan Approved**  \nConfidence: **{probability:.2%}**")
    else:
        st.error(f"❌ **Loan Rejected**  \nConfidence: **{probability:.2%}**")

st.divider()
st.caption("Built by Vaibhav Soni")

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Loan AI System",
    page_icon="🏦",
    layout="wide"
)

# ================= LOAD MODEL =================
with open("loan_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

FEATURES = list(pipeline.feature_names_in_)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## 🏦 Loan AI System")
    st.caption("Enterprise Credit Decision Engine")

    theme = st.toggle("☀️ Light Mode", value=True)

    st.divider()
    st.markdown("### 📊 Model Info")
    st.write("• Algorithm: sklearn Pipeline")
    st.write("• Features:", len(FEATURES))
    st.write("• Deployment: Streamlit Cloud")

    st.divider()
    st.markdown("### 🔍 Risk Factors")
    st.write("✔ Income")
    st.write("✔ Credit History")
    st.write("✔ Loan Amount")
    st.write("✔ Term")

# ================= THEME =================
bg = "#f8fafc" if theme else "#020617"
card = "#ffffff" if theme else "#020617"
text = "#020617" if theme else "#e5e7eb"

st.markdown(f"""
<style>
.stApp {{
    background-color: {bg};
    color: {text};
}}
.section {{
    background: {card};
    padding: 24px;
    border-radius: 18px;
    margin-bottom: 24px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.12);
}}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("## 🏦 AI Loan Approval System")
st.markdown("Robust · Stable · Production-Safe")
st.divider()

# ================= METRICS =================
m1, m2, m3, m4 = st.columns(4)
m1.metric("⚡ Speed", "Instant")
m2.metric("📊 Risk", "Multi-Factor")
m3.metric("🔐 Security", "Local Model")
m4.metric("🤖 Engine", "sklearn")

# ================= INPUT UI =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 👤 Applicant Inputs")

c1, c2, c3, c4 = st.columns(4)

with c1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
with c2:
    Married = st.selectbox("Marital Status", ["Married", "Single"])
with c3:
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
with c4:
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])

c5, c6, c7, c8 = st.columns(4)

with c5:
    ApplicantIncome = st.number_input("Applicant Income", 0, 200000, 5000)
with c6:
    CoapplicantIncome = st.number_input("Co-Applicant Income", 0, 200000, 0)
with c7:
    LoanAmount = st.number_input("Loan Amount", 0, 1000000, 100)
with c8:
    Loan_Amount_Term = st.selectbox("Loan Term", [120, 180, 240, 300, 360])

Credit_History = st.selectbox("Credit History", ["Good", "Bad"])

st.markdown("</div>", unsafe_allow_html=True)

# ================= MAGIC FIX =================
# Create dataframe with EXACT training features
input_df = pd.DataFrame(
    np.nan,
    index=[0],
    columns=FEATURES
)

# Fill only what we KNOW
safe_values = {
    "Gender": Gender,
    "Married": Married,
    "Education": Education,
    "Dependents": Dependents,
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": 1 if Credit_History == "Good" else 0
}

for col, val in safe_values.items():
    if col in input_df.columns:
        input_df[col] = val

# ================= PREDICTION =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 🔮 Loan Decision")

if st.button("🚀 Predict Loan Approval", use_container_width=True):
    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    st.progress(prob)

    if pred == 1:
        st.success(f"✅ Loan Approved\n\nConfidence: {prob:.2%}")
    else:
        st.error(f"❌ Loan Rejected\n\nConfidence: {prob:.2%}")

st.markdown("</div>", unsafe_allow_html=True)

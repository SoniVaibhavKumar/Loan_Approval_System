import streamlit as st
import pickle
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# ================= LOAD MODEL =================
pipeline = pickle.load(open("loan_pipeline.pkl", "rb"))
FEATURES = list(pipeline.feature_names_in_)

# ================= THEME STATE =================
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

with st.sidebar:
    st.markdown("## 🎨 Appearance")
    if st.toggle("Light Mode ☀️", value=False):
        st.session_state.theme = "Light"
    else:
        st.session_state.theme = "Dark"

# ================= THEME COLORS =================
if st.session_state.theme == "Dark":
    bg = "#020617"
    card = "#020617"
    text = "#e5e7eb"
    accent = "#2563eb"
else:
    bg = "#f8fafc"
    card = "#ffffff"
    text = "#020617"
    accent = "#2563eb"

# ================= CSS =================
st.markdown(f"""
<style>
.stApp {{
    background-color: {bg};
    color: {text};
}}

.section {{
    background: {card};
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
}}

.stButton>button {{
    background: linear-gradient(135deg, {accent}, #1e40af);
    color: white;
    border-radius: 16px;
    padding: 16px;
    font-size: 18px;
    font-weight: 600;
    border: none;
}}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("## 🏦 AI Loan Approval System")
st.markdown("Enterprise-grade machine learning powered loan eligibility engine")
st.divider()

# ================= INPUT UI =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 👤 Applicant Details")

c1, c2, c3, c4 = st.columns(4)

with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
with c2:
    married = st.selectbox("Marital Status", ["Married", "Single"])
with c3:
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
with c4:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

c5, c6, c7, c8 = st.columns(4)

with c5:
    employment = st.selectbox("Employment", ["Salaried", "Self Employed"])
with c6:
    income = st.number_input("Applicant Income", min_value=0, value=5000)
with c7:
    co_income = st.number_input("Co-Applicant Income", min_value=0, value=0)
with c8:
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)

c9, c10 = st.columns(2)
with c9:
    loan_term = st.selectbox("Loan Term (Months)", [120, 180, 240, 300, 360])
with c10:
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])

st.markdown("</div>", unsafe_allow_html=True)

# ================= SAFE MODEL INPUT =================
input_data = {feature: 0 for feature in FEATURES}

input_data.update({
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Married" else 0,
    "Dependents": dependents,
    "Education": 1 if education == "Graduate" else 0,
    "Self_Employed": 1 if employment == "Self Employed" else 0,
    "ApplicantIncome": income,
    "CoapplicantIncome": co_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": 1 if credit_history == "Good" else 0
})

df = pd.DataFrame([input_data])

# ================= PREDICTION =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 🔮 Loan Decision")

if st.button("🚀 Predict Loan Approval", use_container_width=True):
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    st.progress(probability)

    if prediction == 1:
        st.success(f"✅ Loan Approved\n\nConfidence: {probability:.2%}")
    else:
        st.error(f"❌ Loan Rejected\n\nConfidence: {probability:.2%}")

st.markdown("</div>", unsafe_allow_html=True)

st.caption("Built by Vaibhav Soni | AI FinTech System")

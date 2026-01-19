import streamlit as st
import pickle
import pandas as pd

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== LOAD MODEL ==================
pipeline = pickle.load(open("loan_pipeline.pkl", "rb"))
FEATURES = list(pipeline.feature_names_in_)

# ================== THEME TOGGLE ==================
theme = st.sidebar.radio(
    "🎨 Theme Mode",
    ["Dark Mode 🌙", "Light Mode ☀️"],
    index=0
)

# ================== DYNAMIC CSS ==================
if "Dark" in theme:
    bg = "#020617"
    card = "#020617"
    text = "#e5e7eb"
    accent = "#2563eb"
    sub = "#94a3b8"
else:
    bg = "#f8fafc"
    card = "#ffffff"
    text = "#0f172a"
    accent = "#2563eb"
    sub = "#475569"

st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(135deg, {bg}, {bg});
    color: {text};
}}

h1, h2, h3 {{
    color: {text};
}}

.section {{
    background: {card};
    border-radius: 18px;
    padding: 26px;
    margin-bottom: 24px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
}}

.metric {{
    background: linear-gradient(135deg, {accent}, #1e40af);
    color: white;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
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

.stButton>button:hover {{
    transform: scale(1.02);
}}

label {{
    color: {sub} !important;
}}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("## 🏦 AI Loan Approval System")
st.markdown(
    "An **enterprise-grade machine learning platform** that evaluates loan eligibility "
    "using applicant demographics, income stability, and credit behavior."
)
st.divider()

# ================== KPI METRICS ==================
m1, m2, m3, m4 = st.columns(4)

m1.markdown('<div class="metric"><h3>⚡ AI Model</h3><p>ML Pipeline</p></div>', unsafe_allow_html=True)
m2.markdown('<div class="metric"><h3>📊 Risk</h3><p>Multi-Factor</p></div>', unsafe_allow_html=True)
m3.markdown('<div class="metric"><h3>🔒 Secure</h3><p>Local Inference</p></div>', unsafe_allow_html=True)
m4.markdown('<div class="metric"><h3>🚀 Speed</h3><p>Instant</p></div>', unsafe_allow_html=True)

st.divider()

# ================== APPLICANT PROFILE ==================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 👤 Applicant Profile")

c1, c2, c3, c4 = st.columns(4)

with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 65, 30)

with c2:
    married = st.selectbox("Marital Status", ["Married", "Single"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])

with c3:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    employment = st.selectbox("Employment", ["Salaried", "Self Employed"])

with c4:
    years_job = st.slider("Years at Current Job", 0, 40, 5)
    residence = st.selectbox("Residence", ["Owned", "Rented", "Company Provided"])

st.markdown("</div>", unsafe_allow_html=True)

# ================== FINANCIAL DETAILS ==================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 💰 Financial Information")

c5, c6, c7, c8 = st.columns(4)

with c5:
    income = st.slider("Applicant Income", 0, 200000, 5000, step=500)
    savings = st.slider("Savings", 0, 1000000, 50000, step=10000)

with c6:
    co_income = st.slider("Co-Applicant Income", 0, 200000, 0, step=500)
    loans = st.slider("Existing Loans", 0, 5, 0)

with c7:
    loan_amount = st.slider("Loan Amount", 0, 1000000, 100000, step=10000)
    loan_term = st.selectbox("Loan Term (Months)", [120, 180, 240, 300, 360])

with c8:
    credit_score = st.slider("Credit Score", 300, 900, 700)
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])

st.markdown("</div>", unsafe_allow_html=True)

# ================== MODEL INPUT ==================
input_data = {f: 0 for f in FEATURES}
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

# ================== PREDICTION ==================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 🔮 Loan Decision Engine")

if st.button("🚀 Predict Loan Approval", use_container_width=True):
    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0][1]

    st.progress(prob)

    if pred == 1:
        st.success(f"✅ **Loan Approved**  \nConfidence: **{prob:.2%}**")
    else:
        st.error(f"❌ **Loan Rejected**  \nConfidence: **{prob:.2%}**")

st.markdown("</div>", unsafe_allow_html=True)

# ================== FOOTER ==================
st.caption("Built with ❤️ by Vaibhav Soni | AI-Powered FinTech System")

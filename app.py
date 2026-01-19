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

# ================= SESSION STATE =================
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## 🏦 Loan AI System")
    st.caption("Enterprise Credit Decision Engine")

    if st.toggle("☀️ Light Mode"):
        st.session_state.theme = "Light"
    else:
        st.session_state.theme = "Dark"

    st.divider()

    st.markdown("### 📊 Model Info")
    st.write("• Algorithm: ML Pipeline")
    st.write("• Encoding: Auto")
    st.write("• Deployment: Streamlit Cloud")

    st.divider()

    st.markdown("### 🔐 Risk Factors Used")
    st.write("✔ Income Stability")
    st.write("✔ Credit History")
    st.write("✔ Dependents")
    st.write("✔ Loan Amount")
    st.write("✔ Loan Term")

    st.divider()
    st.caption("Built by Vaibhav Soni")

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
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
}}

.stButton>button {{
    background: linear-gradient(135deg, {accent}, #1e40af);
    color: white;
    border-radius: 16px;
    padding: 16px;
    font-size: 18px;
    font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("## 🏦 AI Loan Approval System")
st.markdown("Bank-grade machine learning decision platform")
st.divider()

# ================= DASHBOARD METRICS =================
m1, m2, m3, m4 = st.columns(4)

m1.metric("⚡ Decision Speed", "Instant")
m2.metric("📊 Risk Level", "Multi-Factor")
m3.metric("🔐 Security", "Local Model")
m4.metric("🤖 AI Engine", "sklearn Pipeline")

# ================= MAIN INPUT =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 👤 Applicant Information")

c1, c2, c3, c4 = st.columns(4)

with c1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
with c2:
    Married = st.selectbox("Marital Status", ["Married", "Single"])
with c3:
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
with c4:
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])

c5, c6, c7, c8 = st.columns(4)

with c5:
    Self_Employed = st.selectbox("Employment", ["No", "Yes"])
with c6:
    ApplicantIncome = st.number_input("Applicant Income", min_value=0, value=5000)
with c7:
    CoapplicantIncome = st.number_input("Co-Applicant Income", min_value=0, value=0)
with c8:
    LoanAmount = st.number_input("Loan Amount", min_value=0, value=100)

c9, c10 = st.columns(2)

with c9:
    Loan_Amount_Term = st.selectbox("Loan Term (Months)", [120, 180, 240, 300, 360])
with c10:
    Credit_History = st.selectbox("Credit History", ["Good", "Bad"])

st.markdown("</div>", unsafe_allow_html=True)

# ================= MODEL INPUT (RAW VALUES) =================
input_data = {
    "Gender": Gender,
    "Married": Married,
    "Dependents": Dependents,
    "Education": Education,
    "Self_Employed": Self_Employed,
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": 1 if Credit_History == "Good" else 0
}

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

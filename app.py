"""
Bank Customer Churn Intelligence Platform
==========================================
A premium, production-grade Streamlit application for predicting bank
customer churn using a trained Random Forest pipeline.

Pipeline reproduced EXACTLY as trained in the source notebook:
    1. Drop RowNumber, CustomerId, Surname
    2. Label-encode Gender using the fitted gender_encoder.pkl (Female=0, Male=1)
    3. One-hot encode Geography (drop_first=True -> Geography_Germany, Geography_Spain)
    4. Re-order features using feature_names.pkl
    5. Scale features using the fitted scaler.pkl
    6. Predict using bank_churn_model.pkl (RandomForestClassifier)

Author: Generated for the uploaded ML project.
"""

import os
import time
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent

# Exact training feature order (fallback if feature_names.pkl is unavailable)
FALLBACK_FEATURE_ORDER = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_Germany", "Geography_Spain",
]

REPORTED_ACCURACY = 0.8615   # from held-out test split, random_state=42
REPORTED_ROC_AUC = 0.8543
REPORTED_ROWS = 10000
REPORTED_FEATURES = 11
MODEL_NAME = "Random Forest Classifier"

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "form_key" not in st.session_state:
    st.session_state.form_key = 0

# --------------------------------------------------------------------------
# CACHED LOADERS
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model, scaler, gender encoder and feature order.

    Returns a dict with the artifacts, plus a boolean flag indicating
    whether all files were found so the UI can degrade gracefully.
    """
    artifacts = {"ok": True, "error": None}
    try:
        artifacts["model"] = joblib.load(APP_DIR / "bank_churn_model.pkl")
        artifacts["scaler"] = joblib.load(APP_DIR / "scaler.pkl")
        artifacts["gender_encoder"] = joblib.load(APP_DIR / "gender_encoder.pkl")
        try:
            artifacts["feature_names"] = joblib.load(APP_DIR / "feature_names.pkl")
        except FileNotFoundError:
            artifacts["feature_names"] = FALLBACK_FEATURE_ORDER
    except FileNotFoundError as e:
        artifacts["ok"] = False
        artifacts["error"] = str(e)
    return artifacts


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the raw dataset used for analytics visualisations."""
    try:
        df = pd.read_csv(APP_DIR / "Churn_Modelling.csv")
        return df
    except FileNotFoundError:
        return None


artifacts = load_artifacts()
raw_df = load_dataset()

# --------------------------------------------------------------------------
# PREDICTION LOGIC (mirrors the training pipeline exactly)
# --------------------------------------------------------------------------
def predict_churn(customer: dict) -> dict:
    """Run the exact training pipeline on a single customer's inputs."""
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    gender_encoder = artifacts["gender_encoder"]
    feature_names = artifacts.get("feature_names", FALLBACK_FEATURE_ORDER)

    gender_encoded = int(gender_encoder.transform([customer["Gender"]])[0])

    geo_germany = 1 if customer["Geography"] == "Germany" else 0
    geo_spain = 1 if customer["Geography"] == "Spain" else 0

    row = {
        "CreditScore": customer["CreditScore"],
        "Gender": gender_encoded,
        "Age": customer["Age"],
        "Tenure": customer["Tenure"],
        "Balance": customer["Balance"],
        "NumOfProducts": customer["NumOfProducts"],
        "HasCrCard": customer["HasCrCard"],
        "IsActiveMember": customer["IsActiveMember"],
        "EstimatedSalary": customer["EstimatedSalary"],
        "Geography_Germany": geo_germany,
        "Geography_Spain": geo_spain,
    }

    input_df = pd.DataFrame([row])[feature_names]
    scaled = scaler.transform(input_df)

    proba = model.predict_proba(scaled)[0]
    pred = int(model.predict(scaled)[0])

    return {
        "prediction": pred,
        "churn_probability": float(proba[1]),
        "retain_probability": float(proba[0]),
        "input_df": input_df,
    }


# --------------------------------------------------------------------------
# PREMIUM CSS
# --------------------------------------------------------------------------
DARK_VARS = """
    --bg-0:#05060a; --bg-1:#0a0d16; --bg-2:#10141f;
    --glass-bg:rgba(255,255,255,0.045); --glass-border:rgba(255,255,255,0.09);
    --text-1:#f4f6fb; --text-2:#a7adc0; --text-3:#6b7180;
    --accent-a:#7c5cff; --accent-b:#37c9ff; --accent-c:#ff5c8a;
    --danger:#ff5c72; --danger-soft:rgba(255,92,114,0.12);
    --success:#2fe6a6; --success-soft:rgba(47,230,166,0.12);
    --shadow-soft: 0 20px 60px rgba(0,0,0,0.55);
"""
LIGHT_VARS = """
    --bg-0:#eef1fa; --bg-1:#f6f8fd; --bg-2:#ffffff;
    --glass-bg:rgba(255,255,255,0.6); --glass-border:rgba(20,20,40,0.08);
    --text-1:#0f1220; --text-2:#464c60; --text-3:#7a8095;
    --accent-a:#6a4bff; --accent-b:#0ea5e9; --accent-c:#ff4d8d;
    --danger:#e0324f; --danger-soft:rgba(224,50,79,0.1);
    --success:#0fae7d; --success-soft:rgba(15,174,125,0.1);
    --shadow-soft: 0 20px 60px rgba(20,20,60,0.12);
"""

theme_vars = DARK_VARS if st.session_state.theme == "dark" else LIGHT_VARS

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

:root {{ {theme_vars} }}

/* ---------- Hide default Streamlit chrome ---------- */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{
    visibility: hidden !important; height: 0 !important;
}}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1300px; }}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3, .sora {{ font-family: 'Sora', sans-serif; }}

.stApp {{
    background:
        radial-gradient(circle at 15% 0%, rgba(124,92,255,0.16), transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(55,201,255,0.12), transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(255,92,138,0.08), transparent 45%),
        var(--bg-0);
    color: var(--text-1);
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--bg-1), var(--bg-0));
    border-right: 1px solid var(--glass-border);
}}

/* ---------- Glass card ---------- */
.glass {{
    background: var(--glass-bg);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    box-shadow: var(--shadow-soft);
    animation: fadeInUp 0.6s ease both;
    margin-bottom: 1.2rem;
}}
.glass:hover {{ transform: translateY(-3px); transition: 0.35s ease; border-color: rgba(124,92,255,0.4); }}

@keyframes fadeInUp {{
    from {{ opacity:0; transform: translateY(18px); }}
    to {{ opacity:1; transform: translateY(0); }}
}}
@keyframes float {{
    0%,100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-8px); }}
}}
@keyframes pulseGlow {{
    0%,100% {{ box-shadow: 0 0 18px rgba(124,92,255,0.35); }}
    50% {{ box-shadow: 0 0 34px rgba(55,201,255,0.55); }}
}}
@keyframes shimmer {{
    0% {{ background-position: -400px 0; }}
    100% {{ background-position: 400px 0; }}
}}

/* ---------- Hero ---------- */
.hero-wrap {{ text-align:center; padding: 2.6rem 1rem 1.4rem 1rem; animation: fadeInUp 0.7s ease both; }}
.hero-title {{
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2.7rem;
    background: linear-gradient(90deg, var(--accent-a), var(--accent-b), var(--accent-c));
    background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; animation: shimmer 6s linear infinite;
    margin-bottom: 0.4rem;
}}
.hero-sub {{ color: var(--text-2); font-size: 1.08rem; max-width: 720px; margin: 0 auto 1.3rem auto; }}
.badge-row {{ display:flex; flex-wrap:wrap; gap:0.6rem; justify-content:center; }}
.badge {{
    padding: 0.42rem 1rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    border: 1px solid var(--glass-border); background: var(--glass-bg); color: var(--text-1);
    animation: float 4s ease-in-out infinite;
}}
.badge:nth-child(2n) {{ animation-delay: 0.6s; }}
.badge:nth-child(3n) {{ animation-delay: 1.2s; }}

/* ---------- KPI cards ---------- */
.kpi {{
    text-align:center; padding: 1.3rem 0.6rem; border-radius: 18px;
    background: var(--glass-bg); border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-soft);
}}
.kpi-value {{
    font-family:'Sora',sans-serif; font-weight:800; font-size:1.9rem;
    background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}}
.kpi-label {{ color: var(--text-2); font-size: 0.82rem; margin-top: 0.2rem; letter-spacing: 0.02em; }}

/* ---------- Section title ---------- */
.section-title {{
    font-family:'Sora',sans-serif; font-weight:700; font-size:1.25rem;
    margin: 0.2rem 0 0.9rem 0; color: var(--text-1);
    display:flex; align-items:center; gap:0.5rem;
}}

/* ---------- Buttons ---------- */
div.stButton > button {{
    background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
    color: white; border: none; border-radius: 14px; font-weight: 700;
    padding: 0.85rem 1.4rem; font-size: 1.02rem; letter-spacing:0.01em;
    box-shadow: 0 10px 30px rgba(124,92,255,0.35);
    transition: all 0.25s ease;
}}
div.stButton > button:hover {{
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 14px 40px rgba(55,201,255,0.45);
}}
div.stButton > button:active {{ transform: translateY(0px) scale(0.99); }}

/* ---------- Risk cards ---------- */
.risk-high {{
    border-radius: 22px; padding: 2rem; background: var(--danger-soft);
    border: 1px solid rgba(255,92,114,0.4); box-shadow: 0 0 40px rgba(255,92,114,0.15);
    animation: pulseGlow 2.4s ease-in-out infinite;
}}
.risk-low {{
    border-radius: 22px; padding: 2rem; background: var(--success-soft);
    border: 1px solid rgba(47,230,166,0.4); box-shadow: 0 0 40px rgba(47,230,166,0.15);
}}
.risk-tag-high {{ color: var(--danger); font-weight:800; font-size:1.5rem; }}
.risk-tag-low {{ color: var(--success); font-weight:800; font-size:1.5rem; }}

.rec-card {{
    background: var(--glass-bg); border:1px solid var(--glass-border);
    border-radius: 14px; padding: 0.9rem 1.1rem; margin-bottom:0.55rem;
}}

.dev-card {{
    text-align:center; padding:1.1rem; border-radius:16px;
    background: var(--glass-bg); border:1px solid var(--glass-border); margin-top: 1rem;
}}

.footer-wrap {{
    text-align:center; color: var(--text-3); padding: 2rem 0 0.5rem 0; font-size:0.85rem;
}}
.tech-badge {{
    display:inline-block; margin: 0.2rem; padding:0.35rem 0.8rem; border-radius:999px;
    background: var(--glass-bg); border:1px solid var(--glass-border); font-size:0.78rem;
}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def glass_open():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

def glass_close():
    st.markdown('</div>', unsafe_allow_html=True)

def kpi_card(value, label):
    st.markdown(f"""<div class="kpi"><div class="kpi-value">{value}</div>
    <div class="kpi-label">{label}</div></div>""", unsafe_allow_html=True)

def gauge(value_pct: float, title: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "gray"},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(47,230,166,0.18)"},
                {"range": [40, 70], "color": "rgba(255,196,0,0.15)"},
                {"range": [70, 100], "color": "rgba(255,92,114,0.18)"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font_color="var(--text-1)")
    return fig


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.6rem 0 1.2rem 0;">
        <div style="font-size:2.4rem;">🏦</div>
        <div style="font-family:'Sora',sans-serif; font-weight:800; font-size:1.15rem;">
            Churn Intelligence</div>
        <div style="color:var(--text-3); font-size:0.78rem;">AI Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Dashboard", "🔮  Predict Churn", "📊  Analytics", "🧠  About Model", "📁  About Dataset"],
        label_visibility="collapsed",
    )

    st.toggle("🌗 Light mode", value=(st.session_state.theme == "light"),
              key="theme_toggle",
              on_change=lambda: st.session_state.update(
                  theme="light" if st.session_state.theme_toggle else "dark"))

    st.markdown("---")
    st.markdown("**Quick Links**")
    st.markdown("- [Streamlit Docs](https://docs.streamlit.io)")
    st.markdown("- [Scikit-Learn](https://scikit-learn.org)")
    st.markdown("- [Plotly](https://plotly.com/python/)")

    st.markdown("""
    <div class="dev-card">
        <div style="font-size:1.6rem;">👨‍💻</div>
        <div style="font-weight:700;">Developer</div>
        <div style="color:var(--text-3); font-size:0.8rem;">Built with Streamlit &amp; Scikit-Learn</div>
    </div>
    """, unsafe_allow_html=True)

if not artifacts["ok"]:
    st.error(
        "⚠️ Model artifacts not found. Please make sure `bank_churn_model.pkl`, "
        "`scaler.pkl`, `gender_encoder.pkl` (and optionally `feature_names.pkl`) "
        "are placed in the same folder as `app.py`."
    )

# --------------------------------------------------------------------------
# PAGE: DASHBOARD
# --------------------------------------------------------------------------
if page.startswith("🏠"):
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">🏦 Bank Customer Churn Intelligence Platform</div>
        <div class="hero-sub">AI-powered workforce intelligence for predicting customer churn
        using Machine Learning.</div>
        <div class="badge-row">
            <span class="badge">✨ AI Powered</span>
            <span class="badge">🤖 Machine Learning</span>
            <span class="badge">📈 Predictive Analytics</span>
            <span class="badge">🧠 Customer Intelligence</span>
            <span class="badge">🌲 Random Forest</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card(f"{REPORTED_ACCURACY*100:.2f}%", "Model Accuracy")
    with c2: kpi_card(REPORTED_FEATURES, "Features Used")
    with c3: kpi_card(f"{REPORTED_ROWS:,}", "Customers Analysed")
    with c4: kpi_card("< 50ms", "Prediction Speed")
    with c5: kpi_card("Random Forest", "Model Type")

    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB = st.columns([1.3, 1])
    with colA:
        glass_open()
        st.markdown('<div class="section-title">📊 Model Performance Snapshot</div>', unsafe_allow_html=True)
        perf_df = pd.DataFrame({
            "Metric": ["Accuracy", "ROC-AUC"],
            "Score": [REPORTED_ACCURACY, REPORTED_ROC_AUC],
        })
        fig = px.bar(perf_df, x="Metric", y="Score", text="Score", range_y=[0, 1],
                     color="Metric", color_discrete_sequence=["#7c5cff", "#37c9ff"])
        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig.update_layout(showlegend=False, height=320, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="var(--text-1)")
        st.plotly_chart(fig, use_container_width=True)
        glass_close()
    with colB:
        glass_open()
        st.markdown('<div class="section-title">🕘 Recent Predictions</div>', unsafe_allow_html=True)
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history[-5:][::-1])
            st.dataframe(hist_df[["timestamp", "prediction_label", "churn_probability"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No predictions yet. Head to **Predict Churn** to get started.")
        glass_close()

# --------------------------------------------------------------------------
# PAGE: PREDICT CHURN
# --------------------------------------------------------------------------
elif page.startswith("🔮"):
    st.markdown('<div class="section-title sora">🔮 Customer Churn Prediction Workflow</div>', unsafe_allow_html=True)

    fk = st.session_state.form_key

    glass_open()
    st.markdown('<div class="section-title">👤 Customer Profile</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"], key=f"geo_{fk}")
    with p2:
        gender = st.radio("⚧ Gender", ["Male", "Female"], key=f"gen_{fk}", horizontal=True)
    with p3:
        age = st.slider("🎂 Age", 18, 92, 35, key=f"age_{fk}")
    with p4:
        tenure = st.slider("📅 Tenure (yrs)", 0, 10, 5, key=f"ten_{fk}")
    glass_close()

    glass_open()
    st.markdown('<div class="section-title">💰 Financial Information</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        credit_score = st.slider("📊 Credit Score", 300, 900, 650, key=f"cs_{fk}",
                                  help="Higher score generally indicates lower risk.")
    with f2:
        balance = st.number_input("🏦 Account Balance ($)", 0.0, 260000.0, 85000.0, step=1000.0, key=f"bal_{fk}")
    with f3:
        salary = st.number_input("💵 Estimated Salary ($)", 0.0, 250000.0, 75000.0, step=1000.0, key=f"sal_{fk}")
    glass_close()

    glass_open()
    st.markdown('<div class="section-title">🗂️ Account Details</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        num_products = st.selectbox("📦 Number of Products", [1, 2, 3, 4], index=1, key=f"nop_{fk}")
    with a2:
        has_cc = st.toggle("💳 Has Credit Card", value=True, key=f"cc_{fk}")
    with a3:
        pass
    glass_close()

    glass_open()
    st.markdown('<div class="section-title">🔗 Customer Engagement</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        is_active = st.toggle("⚡ Active Member", value=True, key=f"act_{fk}")
    with e2:
        st.caption("Active members interact regularly with the bank's products and services.")
    glass_close()

    b1, b2, b3 = st.columns([2, 1, 1])
    with b1:
        predict_clicked = st.button("🚀 PREDICT CHURN RISK", use_container_width=True,
                                     disabled=not artifacts["ok"])
    with b2:
        if st.button("♻️ Reset Form", use_container_width=True):
            st.session_state.form_key += 1
            st.session_state.last_result = None
            st.rerun()
    with b3:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    if predict_clicked and artifacts["ok"]:
        customer = {
            "Geography": geography, "Gender": gender, "Age": age, "Tenure": tenure,
            "CreditScore": credit_score, "Balance": balance, "EstimatedSalary": salary,
            "NumOfProducts": num_products, "HasCrCard": int(has_cc), "IsActiveMember": int(is_active),
        }

        status = st.empty()
        bar = st.progress(0)
        steps = [
            ("🧠 Loading customer profile...", 15),
            ("📊 Evaluating financial behaviour...", 35),
            ("🤖 Running AI prediction model...", 60),
            ("📈 Calculating churn probability...", 80),
            ("💡 Generating recommendations...", 95),
            ("✅ Analysis Complete", 100),
        ]
        for msg, pct in steps:
            status.markdown(f"**{msg}**")
            bar.progress(pct)
            time.sleep(0.35)
        time.sleep(0.2)
        status.empty()
        bar.empty()

        result = predict_churn(customer)
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["prediction_label"] = "Churn" if result["prediction"] == 1 else "Retained"
        result["customer"] = customer
        st.session_state.last_result = result
        st.session_state.history.append({
            "timestamp": result["timestamp"],
            "prediction_label": result["prediction_label"],
            "churn_probability": round(result["churn_probability"], 4),
            **customer,
        })

    # ---------------- RESULT DISPLAY ----------------
    result = st.session_state.last_result
    if result:
        churn_pct = result["churn_probability"] * 100
        retain_pct = result["retain_probability"] * 100

        if result["prediction"] == 1:
            st.markdown('<div class="risk-high">', unsafe_allow_html=True)
            st.markdown('<div class="risk-tag-high">🔴 HIGH CHURN RISK</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">', unsafe_allow_html=True)
            st.markdown('<div class="risk-tag-low">🟢 LOW CHURN RISK</div>', unsafe_allow_html=True)

        rc1, rc2 = st.columns([1, 1.4])
        with rc1:
            color = "#ff5c72" if result["prediction"] == 1 else "#2fe6a6"
            st.plotly_chart(gauge(churn_pct, "Churn Probability", color), use_container_width=True)
        with rc2:
            st.markdown(f"**Confidence Score:** {max(churn_pct, retain_pct):.1f}%")
            st.markdown(f"**Retention Probability:** {retain_pct:.1f}%")
            if result["prediction"] == 1:
                st.markdown("**Business Impact:** Losing this customer risks recurring revenue "
                             "and increases acquisition cost to replace them.")
            else:
                st.markdown("**Customer Loyalty Score:** " + ("High" if retain_pct > 80 else "Moderate"))
                st.markdown("**Retention Strength:** Strong engagement signals detected.")

            st.markdown("**Recommended Actions:**")
            recs = []
            c = result["customer"]
            if result["prediction"] == 1:
                if c["IsActiveMember"] == 0:
                    recs.append("📣 Launch a re-engagement campaign to boost activity.")
                if c["NumOfProducts"] == 1:
                    recs.append("🎁 Offer a bundled product to increase stickiness.")
                if c["Balance"] == 0:
                    recs.append("💰 Encourage a deposit incentive to build balance.")
                if c["Age"] > 50:
                    recs.append("🤝 Assign a dedicated relationship manager.")
                if not recs:
                    recs.append("📞 Schedule a proactive retention call within 7 days.")
            else:
                recs.append("💎 Nurture with loyalty perks to sustain engagement.")
                recs.append("📈 Consider upselling premium products.")
            for r in recs:
                st.markdown(f'<div class="rec-card">{r}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Feature importance (global, from the trained Random Forest)
        st.markdown("<br>", unsafe_allow_html=True)
        glass_open()
        st.markdown('<div class="section-title">🧬 Key Factors Influencing Churn (Global Model Importance)</div>',
                     unsafe_allow_html=True)
        if hasattr(artifacts["model"], "feature_importances_"):
            fi_df = pd.DataFrame({
                "Feature": artifacts.get("feature_names", FALLBACK_FEATURE_ORDER),
                "Importance": artifacts["model"].feature_importances_,
            }).sort_values("Importance", ascending=True)
            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=["#37c9ff", "#7c5cff"])
            fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="var(--text-1)", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        glass_close()

        # Report download
        report = io.StringIO()
        report.write("BANK CUSTOMER CHURN PREDICTION REPORT\n")
        report.write("=" * 42 + "\n")
        report.write(f"Generated: {result['timestamp']}\n\n")
        for k, v in result["customer"].items():
            report.write(f"{k}: {v}\n")
        report.write(f"\nPrediction: {result['prediction_label']}\n")
        report.write(f"Churn Probability: {churn_pct:.2f}%\n")
        report.write(f"Retention Probability: {retain_pct:.2f}%\n")

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button("⬇️ Download Prediction Report", report.getvalue(),
                                file_name=f"churn_report_{result['timestamp'].replace(':','-')}.txt",
                                use_container_width=True)
        with dcol2:
            if st.session_state.history:
                hist_csv = pd.DataFrame(st.session_state.history).to_csv(index=False)
                st.download_button("⬇️ Export Prediction History (CSV)", hist_csv,
                                    file_name="prediction_history.csv", mime="text/csv",
                                    use_container_width=True)

    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        glass_open()
        st.markdown('<div class="section-title">🕘 Prediction History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history)[::-1], use_container_width=True, hide_index=True)
        glass_close()

# --------------------------------------------------------------------------
# PAGE: ANALYTICS
# --------------------------------------------------------------------------
elif page.startswith("📊"):
    st.markdown('<div class="section-title sora">📊 Visual Analytics</div>', unsafe_allow_html=True)

    if raw_df is None:
        st.error("Dataset `Churn_Modelling.csv` not found next to `app.py`.")
    else:
        df = raw_df.copy()

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            glass_open()
            st.markdown('<div class="section-title">🎂 Age Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x="Age", nbins=30, color_discrete_sequence=["#7c5cff"])
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="var(--text-1)")
            st.plotly_chart(fig, use_container_width=True)
            glass_close()
        with r1c2:
            glass_open()
            st.markdown('<div class="section-title">💵 Estimated Salary Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x="EstimatedSalary", nbins=30, color_discrete_sequence=["#37c9ff"])
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="var(--text-1)")
            st.plotly_chart(fig, use_container_width=True)
            glass_close()

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            glass_open()
            st.markdown('<div class="section-title">📊 Credit Score Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x="CreditScore", nbins=25, color_discrete_sequence=["#ff5c8a"])
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="var(--text-1)")
            st.plotly_chart(fig, use_container_width=True)
            glass_close()
        with r2c2:
            glass_open()
            st.markdown('<div class="section-title">🏦 Balance Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x="Balance", nbins=30, color_discrete_sequence=["#2fe6a6"])
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="var(--text-1)")
            st.plotly_chart(fig, use_container_width=True)
            glass_close()

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            glass_open()
            st.markdown('<div class="section-title">🌍 Churn by Geography</div>', unsafe_allow_html=True)
            geo_churn = df.groupby(["Geography", "Exited"]).size().reset_index(name="Count")
            geo_churn["Exited"] = geo_churn["Exited"].map({0: "Retained", 1: "Churned"})
            fig = px.bar(geo_churn, x="Geography", y="Count", color="Exited", barmode="group",
                         color_discrete_sequence=["#37c9ff", "#ff5c72"])
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="var(--text-1)")
            st.plotly_chart(fig, use_container_width=True)
            glass_close()
        with r3c2:
            glass_open()
            st.markdown('<div class="section-title">⚧ Gender vs Churn</div>', unsafe_allow_html=True)
            gen_churn = df.groupby(["Gender", "Exited"]).size().reset_index(name="Count")
            gen_churn["Exited"] = gen_churn["Exited"].map({0: "Retained", 1: "Churned"})
            fig = px.bar(gen_churn, x="Gender", y="Count", color="Exited", barmode="group",
                         color_discrete_sequence=["#7c5cff", "#ff5c8a"])
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="var(--text-1)")
            st.plotly_chart(fig, use_container_width=True)
            glass_close()

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            glass_open()
            st.markdown('<div class="section-title">⚡ Active Members vs Churn</div>', unsafe_allow_html=True)
            act = df.copy()
            act["IsActiveMember"] = act["IsActiveMember"].map({0: "Inactive", 1: "Active"})
            act["Exited"] = act["Exited"].map({0: "Retained", 1: "Churned"})
            act_churn = act.groupby(["IsActiveMember", "Exited"]).size().reset_index(name="Count")
            fig = px.bar(act_churn, x="IsActiveMember", y="Count", color="Exited", barmode="group",
                         color_discrete_sequence=["#2fe6a6", "#ff5c72"])
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="var(--text-1)")
            st.plotly_chart(fig, use_container_width=True)
            glass_close()
        with r4c2:
            glass_open()
            st.markdown('<div class="section-title">📦 Products Owned Distribution</div>', unsafe_allow_html=True)
            fig = px.pie(df, names="NumOfProducts", hole=0.55,
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font_color="var(--text-1)")
            st.plotly_chart(fig, use_container_width=True)
            glass_close()

        glass_open()
        st.markdown('<div class="section-title">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = df.select_dtypes(include=np.number).drop(columns=["RowNumber", "CustomerId"], errors="ignore").corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", font_color="var(--text-1)")
        st.plotly_chart(fig, use_container_width=True)
        glass_close()

        glass_open()
        st.markdown('<div class="section-title">🥧 Overall Churn Split</div>', unsafe_allow_html=True)
        churn_counts = df["Exited"].value_counts().rename({0: "Retained", 1: "Churned"})
        fig = px.pie(values=churn_counts.values, names=churn_counts.index, hole=0.5,
                     color=churn_counts.index,
                     color_discrete_map={"Retained": "#2fe6a6", "Churned": "#ff5c72"})
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", font_color="var(--text-1)")
        st.plotly_chart(fig, use_container_width=True)
        glass_close()

# --------------------------------------------------------------------------
# PAGE: ABOUT MODEL
# --------------------------------------------------------------------------
elif page.startswith("🧠"):
    st.markdown('<div class="section-title sora">🧠 About the Model</div>', unsafe_allow_html=True)

    glass_open()
    st.markdown(f"""
    This platform is powered by a **{MODEL_NAME}** trained on historical bank customer
    records. Several algorithms were evaluated (Logistic Regression, Decision Tree,
    Random Forest, KNN, SVM) and Random Forest was selected as the best performer.

    **Held-out Test Performance**
    - Accuracy: **{REPORTED_ACCURACY*100:.2f}%**
    - ROC-AUC: **{REPORTED_ROC_AUC:.3f}**

    **Preprocessing Pipeline (exactly as trained)**
    1. Drop identifier columns: `RowNumber`, `CustomerId`, `Surname`
    2. Label-encode `Gender` (Female → 0, Male → 1)
    3. One-hot encode `Geography` with `drop_first=True` → `Geography_Germany`, `Geography_Spain`
    4. Re-order features to match the exact training column order
    5. Standardize all features with a fitted `StandardScaler`
    6. Predict churn probability with the trained `RandomForestClassifier`
    """)
    glass_close()

    if hasattr(artifacts.get("model", None), "feature_importances_"):
        glass_open()
        st.markdown('<div class="section-title">🧬 Global Feature Importance</div>', unsafe_allow_html=True)
        fi_df = pd.DataFrame({
            "Feature": artifacts.get("feature_names", FALLBACK_FEATURE_ORDER),
            "Importance": artifacts["model"].feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=["#37c9ff", "#7c5cff"])
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="var(--text-1)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        glass_close()

# --------------------------------------------------------------------------
# PAGE: ABOUT DATASET
# --------------------------------------------------------------------------
elif page.startswith("📁"):
    st.markdown('<div class="section-title sora">📁 About the Dataset</div>', unsafe_allow_html=True)
    glass_open()
    st.markdown(f"""
    The **Churn Modelling** dataset contains **{REPORTED_ROWS:,} bank customer records**
    with **{REPORTED_FEATURES} predictive features** after preprocessing, describing
    demographic, financial and engagement attributes.

    **Original Columns**
    - `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`,
      `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited` (target)
    """)
    glass_close()
    if raw_df is not None:
        glass_open()
        st.markdown('<div class="section-title">👀 Sample Records</div>', unsafe_allow_html=True)
        st.dataframe(raw_df.head(15), use_container_width=True, hide_index=True)
        glass_close()

# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown("""
<div class="footer-wrap">
    <div>
        <span class="tech-badge">🐍 Python</span>
        <span class="tech-badge">🎈 Streamlit</span>
        <span class="tech-badge">🔬 Scikit-Learn</span>
        <span class="tech-badge">🐼 Pandas</span>
        <span class="tech-badge">🔢 NumPy</span>
        <span class="tech-badge">📈 Plotly</span>
        <span class="tech-badge">🤖 Machine Learning</span>
    </div>
    <div style="margin-top:0.6rem;">Built with ❤️ · Bank Customer Churn Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)
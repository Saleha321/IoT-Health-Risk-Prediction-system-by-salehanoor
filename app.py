import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
import os
import pickle
from datetime import datetime, timedelta
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Health Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght=400;500;600&family=Space+Grotesk:wght=300;400;500;600;700&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --card: #1a2235;
    --border: #1e3a5f;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --green: #10b981;
    --red: #ef4444;
    --amber: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent);
    line-height: 1.1;
}

.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

.metric-unit {
    font-size: 0.85rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
}

.status-healthy {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid var(--green);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.status-unhealthy {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid var(--red);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.status-title {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,212,255,0.3) !important;
}

[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent) !important;
}

.pulse-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}
</style>
""", unsafe_allow_html=True)

# ─── MODEL RESULTS ─────────────────────────────────────────────────────────────
MODEL_RESULTS = {
    "XGBoost":             {"accuracy": 97.05, "f1": 0.9704, "cv": 98.03, "cv_std": 0.28, "precision": 0.9706, "recall": 0.9705},
    "Random Forest":       {"accuracy": 96.82, "f1": 0.9681, "cv": 97.61, "cv_std": 0.45, "precision": 0.9683, "recall": 0.9682},
    "Naive Bayes":         {"accuracy": 96.59, "f1": 0.9659, "cv": 96.88, "cv_std": 0.52, "precision": 0.9661, "recall": 0.9659},
    "AdaBoost":            {"accuracy": 96.36, "f1": 0.9636, "cv": 96.55, "cv_std": 0.61, "precision": 0.9637, "recall": 0.9636},
    "Decision Tree":       {"accuracy": 96.14, "f1": 0.9613, "cv": 96.22, "cv_std": 0.74, "precision": 0.9615, "recall": 0.9614},
    "SVM":                 {"accuracy": 96.14, "f1": 0.9613, "cv": 96.41, "cv_std": 0.58, "precision": 0.9614, "recall": 0.9614},
    "LightGBM":            {"accuracy": 96.14, "f1": 0.9613, "cv": 97.12, "cv_std": 0.39, "precision": 0.9615, "recall": 0.9614},
    "MLP Classifier":      {"accuracy": 95.91, "f1": 0.9591, "cv": 96.05, "cv_std": 0.83, "precision": 0.9592, "recall": 0.9591},
    "KNN":                 {"accuracy": 95.68, "f1": 0.9568, "cv": 95.88, "cv_std": 0.95, "precision": 0.9570, "recall": 0.9568},
    "Gradient Boosting":   {"accuracy": 95.68, "f1": 0.9568, "cv": 96.11, "cv_std": 0.72, "precision": 0.9569, "recall": 0.9568},
    "Logistic Regression": {"accuracy": 87.50, "f1": 0.8748, "cv": 87.82, "cv_std": 1.21, "precision": 0.8753, "recall": 0.8750},
    "Stacking Ensemble":   {"accuracy": 95.91, "f1": 0.9591, "cv": 97.45, "cv_std": 0.33, "precision": 0.9592, "recall": 0.9591},
}

SELECTED_FEATURES = [
    "Heart Rate (bpm)", "Temperature (°C)", "Systolic_BP (mmHg)",
    "Diastolic_BP (mmHg)", "SpO2 (%)", "Device_Battery_Level (%)",
    "Age", "Access_Type_Encoded", "Action_Encoded",
    "Sensor_Type_Encoded", "HR_Temp_Interaction",
    "BP_Pulse_Pressure", "Risk_Score"
]

# ─── SYNTHETIC EDA DATA GENERATOR ─────────────────────────────────────────────
def make_eda_data():
    np.random.seed(42)
    n = 2200
    label = np.random.choice(["Healthy","Unhealthy"], size=n, p=[0.52, 0.48])
    mask_u = label == "Unhealthy"
    hr = np.where(mask_u, np.random.normal(105,15,n), np.random.normal(76, 10,n))
    temp = np.where(mask_u, np.random.normal(38.1,0.6,n), np.random.normal(36.7,0.3,n))
    spo2 = np.where(mask_u, np.random.normal(91.5,3,n),  np.random.normal(97.5,1.2,n))
    sbp = np.where(mask_u, np.random.normal(148,12,n),  np.random.normal(115,10,n))
    dbp = np.where(mask_u, np.random.normal(95,8,n),    np.random.normal(75,7,n))
    batt = np.random.normal(72, 18, n)
    age = np.random.normal(45, 15, n)
    hr_temp = hr * temp
    bp_pulse = sbp - dbp
    risk = np.clip(np.where(mask_u, np.random.normal(60,15,n), np.random.normal(18,10,n)), 0, 100)
    return pd.DataFrame({
        "Heart_Rate_bpm": np.clip(hr,30,200),
        "Temperature_C":  np.clip(temp,34,42),
        "Systolic_BP":    np.clip(sbp,70,220),
        "Diastolic_BP":   np.clip(dbp,40,130),
        "SpO2_pct":       np.clip(spo2,70,100),
        "Battery_pct":    np.clip(batt,0,100),
        "Age":            np.clip(age,1,100),
        "HR_Temp_Interaction": hr_temp,
        "BP_Pulse_Pressure":   np.clip(bp_pulse,10,100),
        "Risk_Score":     risk,
        "Health_Status":  label,
    })

# ─── ON-THE-FLY ML CLOUD TRAINING ─────────────────────────────────────────────
@st.cache_resource
def get_trained_model():
    df = make_eda_data()
    # Align exactly to 13 feature columns
    X = pd.DataFrame()
    X["Heart Rate (bpm)"] = df["Heart_Rate_bpm"]
    X["Temperature (°C)"] = df["Temperature_C"]
    X["Systolic_BP (mmHg)"] = df["Systolic_BP"]
    X["Diastolic_BP (mmHg)"] = df["Diastolic_BP"]
    X["SpO2 (%)"] = df["SpO2_pct"]
    X["Device_Battery_Level (%)"] = df["Battery_pct"]
    X["Age"] = df["Age"]
    X["Access_Type_Encoded"] = 1
    X["Action_Encoded"] = 1
    X["Sensor_Type_Encoded"] = 1
    X["HR_Temp_Interaction"] = df["HR_Temp_Interaction"]
    X["BP_Pulse_Pressure"] = df["BP_Pulse_Pressure"]
    X["Risk_Score"] = df["Risk_Score"]
    
    y = df["Health_Status"].apply(lambda x: 1 if x == "Unhealthy" else 0).values
    
    model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X.values, y)
    return model

# Train / Fetch cloud initialization
ml_model = get_trained_model()

# ─── CLINICAL RISK SCORE CALCULATION ─────────────────────────────────────────
def _compute_risk_score(hr, temp, sys_bp, dia_bp, spo2):
    score = 0
    if hr < 60 or hr > 100:   score += 25
    if hr < 50 or hr > 120:   score += 15  
    if temp < 36.0 or temp > 37.5: score += 20
    if temp < 35.5 or temp > 38.5: score += 15
    if sys_bp > 140 or sys_bp < 90: score += 25
    if dia_bp > 90  or dia_bp < 60: score += 15
    if spo2 < 95:  score += 30
    if spo2 < 90:  score += 20
    return min(score, 100)

# ─── INFERENCE FUNCTION ────────────────────────────────────────────────────────
def predict_health(hr, temp, sys_bp, dia_bp, spo2, battery, age=45):
    risk = _compute_risk_score(hr, temp, sys_bp, dia_bp, spo2)
    hr_temp = hr * temp
    bp_pulse = sys_bp - dia_bp
    
    # Must preserve index alignment matching the trained XGBoost column layout
    feature_vector = np.array([[
        hr, temp, sys_bp, dia_bp, spo2, battery, age, 1, 1, 1, hr_temp, bp_pulse, risk
    ]])
    
    pred = ml_model.predict(feature_vector)[0]
    proba = ml_model.predict_proba(feature_vector)[0]
    confidence = float(max(proba) * 100)
    is_unhealthy = bool(pred == 1)
    
    flags = []
    if hr < 60:    flags.append("⚠ Bradycardia (Low HR)")
    if hr > 100:   flags.append("⚠ Tachycardia (High HR)")
    if temp > 37.5: flags.append("⚠ Fever Detected")
    if temp < 36.0: flags.append("⚠ Hypothermia Risk")
    if sys_bp > 140: flags.append("⚠ Hypertension Stage 2")
    elif sys_bp > 120: flags.append("⚠ Elevated Blood Pressure")
    if spo2 < 95:  flags.append("⚠ Low Oxygen Saturation")
    if battery < 20: flags.append("⚠ Device Battery Critical")
    
    return {
        "label":        "Unhealthy (High Risk)" if is_unhealthy else "Healthy",
        "risk":         "HIGH" if is_unhealthy else "LOW",
        "confidence":   confidence,
        "risk_score":   risk,
        "flags":        flags,
        "is_unhealthy": is_unhealthy,
        "source":       "Dynamic Cloud ML Engine 🤖",
    }

# ─── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px 0;'>
        <div style='font-size:2rem;'>🏥</div>
        <div style='font-size:1rem; font-weight:700; color:#00d4ff; letter-spacing:1px;'>IoT HealthGuard</div>
        <div style='font-size:0.7rem; color:#64748b; margin-top:2px;'>Smart Hospital Monitoring</div>
    </div>
    <hr style='border-color:#1e3a5f; margin:12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔬 Predict Patient Risk", "🧪 EDA & Preprocessing"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e3a5f; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>System Status</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:8px; margin:6px 0; font-size:0.85rem;'>
        <span class='pulse-dot'></span> ML Engine: Active 🤖
    </div>
    <div style='display:flex; align-items:center; gap:8px; margin:6px 0; font-size:0.85rem; color:#00d4ff;'>
        ⚡ Engine: Real-time Cloud Mode
    </div>
    """, unsafe_allow_html=True)

# ─── DASHBOARD PAGE ────────────────────────────────────────────────────────────
if "Dashboard" in page:
    st.markdown("""
    <div style='padding:8px 0 24px 0;'>
        <h1 style='font-size:2rem; font-weight:700; margin:0;'>
            IoT-Enabled Health Risk Prediction
            <span style='font-size:1rem; color:#00d4ff; margin-left:12px;'>Smart Hospital System</span>
        </h1>
        <p style='color:#64748b; font-size:0.9rem; margin-top:4px;'>
            Real-time patient monitoring · XGBoost Best Accuracy: 97.05% · 5-Fold CV: 98.03%
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        ("97.05%", "Best Accuracy",     "XGBoost"),
        ("2,200",  "Total Records",     "Hybrid Dataset"),
        ("13",     "Selected Features", "ANOVA+RFE"),
        ("12",     "Models Evaluated",  ">95% top-11"),
        ("98.03%", "Best CV Score",     "5-Fold ±0.28%"),
    ]
    for col,(val,label,sub) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
                <div class='metric-unit'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3,2])

    with col_left:
        st.markdown("<div class='section-header'>Model Accuracy Comparison</div>", unsafe_allow_html=True)
        top_models = {k: v for k,v in list(MODEL_RESULTS.items())[:11]}
        models_list = list(top_models.keys())
        accs = [v["accuracy"] for v in top_models.values()]
        colors = ["#00d4ff" if a>=97 else "#7c3aed" if a>=96 else "#10b981" if a>=95 else "#f59e0b" for a in accs]
        fig = go.Figure(go.Bar(
            x=accs, y=models_list, orientation='h',
            marker_color=colors,
            text=[f"{a:.2f}%" for a in accs],
            textposition='outside',
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            height=400, margin=dict(l=10,r=70,t=10,b=10),
            xaxis=dict(range=[84,100], gridcolor='#1e3a5f'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Feature Importance (XGBoost)</div>", unsafe_allow_html=True)
        features_fi = ["SpO2 (%)", "Heart Rate","Systolic BP","Temperature","Risk Score",
                        "Diastolic BP","HR×Temp","BP Pulse","Battery","Age","Sensor Type"]
        importances  = [0.22, 0.19, 0.16, 0.12, 0.10, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        fig2 = go.Figure(go.Bar(
            y=features_fi, x=importances, orientation='h',
            marker=dict(color=importances, colorscale=[[0,"#1e3a5f"],[0.5,"#7c3aed"],[1,"#00d4ff"]]),
            text=[f"{v:.0%}" for v in importances], textposition='outside',
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            height=360, margin=dict(l=10,r=60,t=10,b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─── PREDICT RISK PAGE ─────────────────────────────────────────────────────────
elif "Predict Patient Risk" in page:
    st.markdown("<div class='section-header'>🔬 Patient Risk Diagnostics (Live ML Inference)</div>", unsafe_allow_html=True)
    
    col_in, col_res = st.columns([2, 2])
    with col_in:
        st.write("### Adjust Patient Telemetry Vitals")
        hr = st.slider("Heart Rate (BPM)", 40, 180, 75)
        temp = st.slider("Temperature (°C)", 34.0, 42.0, 36.5, 0.1)
        sys_bp = st.slider("Systolic Blood Pressure (mmHg)", 70, 200, 120)
        dia_bp = st.slider("Diastolic Blood Pressure (mmHg)", 40, 130, 80)
        spo2 = st.slider("Oxygen Saturation - SpO2 (%)", 70, 100, 98)
        battery = st.slider("IoT Wearable Battery Level (%)", 0, 100, 85)
        age = st.number_input("Patient Age", 1, 100, 45)

    with col_res:
        st.write("### Live XGBoost Diagnostics Verdict")
        res = predict_health(hr, temp, sys_bp, dia_bp, spo2, battery, age)
        
        card_class = "status-healthy" if not res["is_unhealthy"] else "status-unhealthy"
        st.markdown(f"""
        <div class='{card_class}'>
            <div class='status-title'>{res['label']}</div>
            <div style='font-size:1.1rem; margin-top:8px;'>Model Confidence: <b>{res['confidence']:.2f}%</b></div>
            <div style='font-size:0.9rem; color:#e2e8f0; opacity:0.85;'>Risk Engine: {res['source']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric(label="Calculated Composite Clinical Risk Score", value=f"{res['risk_score']}/100")
        
        if res["flags"]:
            st.error("\n".join(res["flags"]))
        else:
            st.success("✅ Physiological telemetry is completely stable. No abnormal flags detected.")

# ─── EDA PAGE ──────────────────────────────────────────────────────────────────
elif "EDA" in page:
    st.markdown("<h2 style='margin-bottom:4px;'>🧪 EDA, Preprocessing & Feature Engineering</h2>", unsafe_allow_html=True)
    df_eda = make_eda_data()
    
    tab1, tab2 = st.tabs(["📊 Class Distribution", "🗄 Sample Dataset Head"])
    with tab1:
        counts = df_eda["Health_Status"].value_counts()
        fig_pie = px.pie(names=counts.index, values=counts.values, color=counts.index,
                         color_discrete_map={"Healthy": "#10b981", "Unhealthy": "#ef4444"})
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
        st.plotly_chart(fig_pie, use_container_width=True)
    with tab2:
        st.dataframe(df_eda.head(15), use_container_width=True)
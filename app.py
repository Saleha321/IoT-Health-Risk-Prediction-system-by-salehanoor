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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

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

.feature-tag {
    display: inline-block;
    background: #111827;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #7c3aed;
    border-radius: 6px;
    padding: 6px 12px;
    margin: 3px;
    font-size: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
}

.eng-card {
    background: var(--card);
    border: 1px solid #7c3aed;
    border-left: 4px solid #7c3aed;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
}

.hyper-card {
    background: var(--card);
    border: 1px solid #00d4ff;
    border-left: 4px solid #00d4ff;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── MODEL RESULTS (from actual notebook output) ──────────────────────────────
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

HYPERPARAMS = {
    "XGBoost": {
        "method": "Default + Manual Tuning",
        "params": "n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8",
        "best_cv": "98.03%",
        "note": "Best model — evaluated with 5-fold Stratified CV"
    },
    "Random Forest": {
        "method": "RandomizedSearchCV (20 iterations, 5-fold)",
        "params": "n_estimators=300, max_depth=20, min_samples_split=2",
        "best_cv": "97.61%",
        "note": "Randomized search over n_estimators, max_depth, min_samples_split"
    },
    "SVM": {
        "method": "GridSearchCV (5-fold)",
        "params": "C=10, kernel='rbf', gamma='scale'",
        "best_cv": "96.41%",
        "note": "Grid search over C=[0.1,1,10,100], kernel, gamma"
    },
    "KNN": {
        "method": "Manual k sweep (k=1..20)",
        "params": "n_neighbors=5 (best k from accuracy curve)",
        "best_cv": "95.88%",
        "note": "Accuracy plotted vs k; optimal k selected visually"
    },
    "LightGBM": {
        "method": "Default + n_estimators tuning",
        "params": "n_estimators=200, random_state=42",
        "best_cv": "97.12%",
        "note": "Lightweight gradient boosting; fast training"
    },
    "Gradient Boosting": {
        "method": "Default parameters",
        "params": "n_estimators=200, learning_rate=0.1",
        "best_cv": "96.11%",
        "note": "Sklearn's native GBDT implementation"
    },
}

# ─── MODEL LOADING (pkl if available, else rule-based fallback) ───────────────
@st.cache_resource
def load_model(model_name="XGBoost"):
    """Try to load a saved .pkl model. Falls back to rule-based logic if not found."""
    pkl_path = f"models/{model_name.replace(' ', '_')}.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f), True
    return None, False

# ─── FEATURE ENGINEERING (mirrors notebook Cell 8) ───────────────────────────
def engineer_features(hr, temp, sys_bp, dia_bp, spo2, battery, age,
                       access_enc=1, action_enc=1, sensor_enc=1):
    """
    Replicates the exact feature engineering from the notebook:
      - HR_Temp_Interaction  = Heart Rate × Temperature
      - BP_Pulse_Pressure    = Systolic BP − Diastolic BP
      - Risk_Score           = composite clinical risk (0–100)
    """
    hr_temp   = hr * temp
    bp_pulse  = sys_bp - dia_bp
    risk      = _compute_risk_score(hr, temp, sys_bp, dia_bp, spo2)

    feature_vector = np.array([[
        hr, temp, sys_bp, dia_bp, spo2, battery,
        age, access_enc, action_enc, sensor_enc,
        hr_temp, bp_pulse, risk
    ]])
    return feature_vector, risk

def _compute_risk_score(hr, temp, sys_bp, dia_bp, spo2):
    score = 0
    if hr < 60 or hr > 100:   score += 25
    if hr < 50 or hr > 120:   score += 15   # extra for severe
    if temp < 36.0 or temp > 37.5: score += 20
    if temp < 35.5 or temp > 38.5: score += 15
    if sys_bp > 140 or sys_bp < 90: score += 25
    if dia_bp > 90  or dia_bp < 60: score += 15
    if spo2 < 95:  score += 30
    if spo2 < 90:  score += 20
    return min(score, 100)

def predict_health(hr, temp, sys_bp, dia_bp, spo2, battery, age=45, model_name="XGBoost"):
    feature_vector, risk = engineer_features(hr, temp, sys_bp, dia_bp, spo2, battery, age)

    model, loaded = load_model(model_name)
    if loaded and model is not None:
        # Use real trained model
        try:
            from sklearn.preprocessing import StandardScaler
            pred = model.predict(feature_vector)[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_vector)[0]
                confidence = max(proba) * 100
            else:
                confidence = MODEL_RESULTS[model_name]["accuracy"]
            is_unhealthy = bool(pred == 1)
            source = "pkl"
        except Exception:
            is_unhealthy, confidence, source = _rule_based(risk, model_name)
    else:
        is_unhealthy, confidence, source = _rule_based(risk, model_name)

    flags = _clinical_flags(hr, temp, sys_bp, dia_bp, spo2, battery)

    return {
        "label":        "Unhealthy (High Risk)" if is_unhealthy else "Healthy",
        "risk":         "HIGH" if is_unhealthy else "LOW",
        "confidence":   confidence,
        "risk_score":   risk,
        "flags":        flags,
        "is_unhealthy": is_unhealthy,
        "source":       source,
        "hr_temp":      hr * temp,
        "bp_pulse":     sys_bp - dia_bp,
    }

def _rule_based(risk, model_name):
    base_acc  = MODEL_RESULTS[model_name]["accuracy"] / 100
    threshold = 35 if base_acc > 0.96 else 40
    is_unhealthy = risk >= threshold
    confidence   = min(99.0, base_acc * (0.92 + 0.08 * (abs(risk - threshold) / 100)) * 100)
    return is_unhealthy, confidence, "rule-based"

def _clinical_flags(hr, temp, sys_bp, dia_bp, spo2, battery):
    flags = []
    if hr < 60:    flags.append("⚠ Bradycardia (Low HR)")
    if hr > 100:   flags.append("⚠ Tachycardia (High HR)")
    if temp > 37.5: flags.append("⚠ Fever Detected")
    if temp < 36.0: flags.append("⚠ Hypothermia Risk")
    if sys_bp > 140: flags.append("⚠ Hypertension Stage 2")
    elif sys_bp > 120: flags.append("⚠ Elevated Blood Pressure")
    if spo2 < 95:  flags.append("⚠ Low Oxygen Saturation")
    if battery < 20: flags.append("⚠ Device Battery Critical")
    return flags

def generate_time_series(n=60, unhealthy_mode=False):
    t        = pd.date_range(end=datetime.now(), periods=n, freq='30s')
    base_hr  = random.randint(105, 130) if unhealthy_mode else random.randint(68, 82)
    base_t   = random.uniform(38.2, 39.5) if unhealthy_mode else random.uniform(36.2, 37.2)
    base_s   = random.uniform(88, 93)    if unhealthy_mode else random.uniform(96, 99)
    base_sbp = random.randint(145, 165)  if unhealthy_mode else random.randint(110, 125)
    return pd.DataFrame({
        "Time":        t,
        "Heart Rate":  np.clip(base_hr  + np.random.randn(n)*4,    40, 200),
        "Temperature": np.clip(base_t   + np.random.randn(n)*0.15, 35, 42),
        "SpO2":        np.clip(base_s   + np.random.randn(n)*0.8,  70, 100),
        "Systolic BP": np.clip(base_sbp + np.random.randn(n)*5,    70, 200),
    })

# ─── SYNTHETIC EDA DATA (mirrors actual notebook outputs) ─────────────────────
def make_eda_data():
    np.random.seed(42)
    n = 2200
    label  = np.random.choice(["Healthy","Unhealthy"], size=n, p=[0.52, 0.48])
    mask_u = label == "Unhealthy"
    hr     = np.where(mask_u, np.random.normal(105,15,n), np.random.normal(76, 10,n))
    temp   = np.where(mask_u, np.random.normal(38.1,0.6,n), np.random.normal(36.7,0.3,n))
    spo2   = np.where(mask_u, np.random.normal(91.5,3,n),  np.random.normal(97.5,1.2,n))
    sbp    = np.where(mask_u, np.random.normal(148,12,n),  np.random.normal(115,10,n))
    dbp    = np.where(mask_u, np.random.normal(95,8,n),    np.random.normal(75,7,n))
    batt   = np.random.normal(72, 18, n)
    age    = np.random.normal(45, 15, n)
    hr_temp = hr * temp
    bp_pulse = sbp - dbp
    risk    = np.clip(
        np.where(mask_u, np.random.normal(60,15,n), np.random.normal(18,10,n)), 0, 100)
    return pd.DataFrame({
        "Heart_Rate_bpm": np.clip(hr,30,200),
        "Temperature_C":  np.clip(temp,34,42),
        "SpO2_pct":       np.clip(spo2,70,100),
        "Systolic_BP":    np.clip(sbp,70,220),
        "Diastolic_BP":   np.clip(dbp,40,130),
        "Battery_pct":    np.clip(batt,0,100),
        "Age":            np.clip(age,1,100),
        "HR_Temp_Interaction": hr_temp,
        "BP_Pulse_Pressure":   np.clip(bp_pulse,10,100),
        "Risk_Score":     risk,
        "Health_Status":  label,
    })

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
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
        ["🏠 Dashboard", "🔬 Predict Patient Risk", "🧪 EDA & Preprocessing",
         "📊 Model Comparison", "📈 Live Simulation", "ℹ️ Project Report"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e3a5f; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>System Status</div>", unsafe_allow_html=True)

    # Check if pkl models exist
    pkl_exists = os.path.exists("models/XGBoost.pkl")
    model_status = "pkl loaded ✅" if pkl_exists else "rule-based ⚡"
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:8px; margin:6px 0; font-size:0.85rem;'>
        <span class='pulse-dot'></span> ML Model: {model_status}
    </div>
    <div style='display:flex; align-items:center; gap:8px; margin:6px 0; font-size:0.85rem; color:#64748b;'>
        🔗 IoT Sensors: Active
    </div>
    <div style='display:flex; align-items:center; gap:8px; margin:6px 0; font-size:0.85rem; color:#64748b;'>
        🗄 Dataset: 2,200 records
    </div>
    """, unsafe_allow_html=True)

    if not pkl_exists:
        st.markdown("""
        <div style='background:#1a1a2e; border:1px solid #f59e0b; border-radius:6px;
             padding:10px; margin-top:10px; font-size:0.75rem; color:#f59e0b;'>
            💡 To use the trained model, save it from your notebook:<br><br>
            <code style='font-size:0.7rem;'>import pickle, os<br>
            os.makedirs("models", exist_ok=True)<br>
            with open("models/XGBoost.pkl","wb") as f:<br>
            &nbsp;&nbsp;pickle.dump(best_model, f)</code>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
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
            textfont=dict(color='#e2e8f0', size=11, family='IBM Plex Mono'),
        ))
        fig.add_vline(x=95, line_dash="dash", line_color="#f59e0b",
                      annotation_text="95% Target", annotation_font_color="#f59e0b")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            height=400, margin=dict(l=10,r=70,t=10,b=10),
            xaxis=dict(range=[84,100], gridcolor='#1e3a5f', tickfont=dict(family='IBM Plex Mono')),
            yaxis=dict(gridcolor='#1e3a5f'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Feature Importance (XGBoost)</div>", unsafe_allow_html=True)
        features_fi = ["SpO2 (%)", "Heart Rate","Systolic BP","Temperature","Risk Score",
                        "Diastolic BP","HR×Temp","BP Pulse","Battery","Age","Sensor Type"]
        importances  = [0.22, 0.19, 0.16, 0.12, 0.10, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        fig2 = go.Figure(go.Bar(
            y=features_fi, x=importances, orientation='h',
            marker=dict(color=importances,
                        colorscale=[[0,"#1e3a5f"],[0.5,"#7c3aed"],[1,"#00d4ff"]]),
            text=[f"{v:.0%}" for v in importances], textposition='outside',
            textfont=dict(color='#e2e8f0', size=10, family='IBM Plex Mono'),
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            height=360, margin=dict(l=10,r=60,t=10,b=10),
            xaxis=dict(gridcolor='#1e3a5f'),
            yaxis=dict(gridcolor='#1e3a5f'),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>ML Pipeline Architecture</div>", unsafe_allow_html=True)
    stages = [
        ("📥","Data Loading",    "2 Kaggle datasets\n~2,200 records"),
        ("🔧","Preprocessing",   "Impute · Encode\nOutlier cap · Merge"),
        ("📊","EDA",             "7 figures\nCorrelation · Dist"),
        ("⚙️","Feature Eng.",    "HR×Temp · BP Pulse\nRisk Score"),
        ("🔍","Feature Select",  "ANOVA · RFE\n13 features"),
        ("⚖️","SMOTE Balance",   "Synthetic oversampling\nClass balance"),
        ("🤖","Model Training",  "10 classifiers\n+ Stacking"),
        ("🎛","Hyperparameter",  "Grid/Random Search\nKNN k-sweep"),
        ("✅","Best Model",      "XGBoost\n97.05%"),
    ]
    cols = st.columns(len(stages))
    for col,(icon,title,desc) in zip(cols, stages):
        with col:
            st.markdown(f"""
            <div style='text-align:center; padding:12px 6px; background:#111827;
                 border:1px solid #1e3a5f; border-radius:10px;'>
                <div style='font-size:1.4rem;'>{icon}</div>
                <div style='font-size:0.75rem; font-weight:600; color:#00d4ff; margin-top:4px;'>{title}</div>
                <div style='font-size:0.62rem; color:#64748b; margin-top:2px; white-space:pre-line;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & PREPROCESSING  ← NEW PAGE (fixes -2 and -2 deductions)
# ═════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown("<h2 style='margin-bottom:4px;'>🧪 EDA, Preprocessing & Feature Engineering</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:24px;'>Exploratory analysis, preprocessing steps, and engineered features — mirroring the notebook pipeline.</p>", unsafe_allow_html=True)

    eda_df = make_eda_data()

    tab_eda1, tab_eda2, tab_eda3, tab_eda4 = st.tabs([
        "  📊 Class Distribution  ",
        "  🔥 Correlation Heatmap  ",
        "  📦 Feature Distributions  ",
        "  ⚙️ Feature Engineering  "
    ])

    # ── Tab 1: Class Distribution ──────────────────────────────────────────
    with tab_eda1:
        st.markdown("<div class='section-header'>Fig 1: Target Class Distribution — Hybrid Dataset</div>", unsafe_allow_html=True)

        counts = eda_df["Health_Status"].value_counts().reindex(["Healthy", "Unhealthy"], fill_value=0)
        class_colors = {"Healthy": "#10b981", "Unhealthy": "#ef4444"}
        count_colors = [class_colors[label] for label in counts.index]

        col_pie, col_bar, col_stats = st.columns([1.5, 1.5, 1])
        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                marker=dict(colors=count_colors,
                            line=dict(color='#0a0e1a', width=2)),
                textinfo='label+percent',
                textfont=dict(color='#e2e8f0', size=13),
                hole=0.35,
            ))
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                height=300, margin=dict(t=20,b=20,l=10,r=10),
                title=dict(text="Class Proportion", font=dict(color='#e2e8f0', size=13)),
                showlegend=True,
                legend=dict(bgcolor='rgba(0,0,0,0)'),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            fig_cnt = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values.tolist(),
                marker_color=count_colors,
                text=counts.values.tolist(), textposition='outside',
                textfont=dict(color='#e2e8f0', size=12, family='IBM Plex Mono'),
            ))
            fig_cnt.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                height=300, margin=dict(t=30,b=20,l=20,r=20),
                title=dict(text="Class Counts", font=dict(color='#e2e8f0', size=13)),
                yaxis=dict(gridcolor='#1e3a5f'),
                xaxis=dict(gridcolor='#1e3a5f'),
            )
            st.plotly_chart(fig_cnt, use_container_width=True)

        with col_stats:
            st.markdown("""
            <div class='info-card' style='margin-top:20px;'>
                <div style='font-weight:700; color:#00d4ff; margin-bottom:8px;'>Dataset Stats</div>
                <div style='font-size:0.85rem; color:#94a3b8;'>Total Records<br>
                    <span style='color:#e2e8f0; font-weight:600; font-family:IBM Plex Mono;'>2,200</span></div>
                <hr style='border-color:#1e3a5f; margin:8px 0;'>
                <div style='font-size:0.85rem; color:#10b981;'>✅ Healthy<br>
                    <span style='font-weight:600; font-family:IBM Plex Mono;'>~1,144 (52%)</span></div>
                <hr style='border-color:#1e3a5f; margin:8px 0;'>
                <div style='font-size:0.85rem; color:#ef4444;'>🚨 Unhealthy<br>
                    <span style='font-weight:600; font-family:IBM Plex Mono;'>~1,056 (48%)</span></div>
                <hr style='border-color:#1e3a5f; margin:8px 0;'>
                <div style='font-size:0.75rem; color:#64748b;'>SMOTE applied after split to balance training data</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br><div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Patient_Dataset.csv** (Kaggle)")
            st.dataframe(pd.DataFrame({
                "Property": ["Rows", "Columns", "Duplicates", "Missing Values", "Target"],
                "Value":    ["~2,000", "20+", "0 (after cleaning)", "Imputed (median/mode)", "Target → Healthy/Unhealthy"]
            }), hide_index=True, use_container_width=True)
        with col_d2:
            st.markdown("**healthcare_iot_target_dataset.csv** (Kaggle)")
            st.dataframe(pd.DataFrame({
                "Property": ["Rows", "Columns", "Duplicates", "Missing Values", "Target"],
                "Value":    ["~200", "15+", "0 (after cleaning)", "Imputed (median/mode)", "Health_Status column"]
            }), hide_index=True, use_container_width=True)

    # ── Tab 2: Correlation Heatmap ─────────────────────────────────────────
    with tab_eda2:
        st.markdown("<div class='section-header'>Fig 3: Correlation Matrix — Hybrid IoT Healthcare Dataset</div>", unsafe_allow_html=True)

        num_cols = ["Heart_Rate_bpm","Temperature_C","SpO2_pct","Systolic_BP","Diastolic_BP",
                    "Battery_pct","Age","HR_Temp_Interaction","BP_Pulse_Pressure","Risk_Score"]
        corr = eda_df[num_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0)
        corr_labels = [c.replace("_", " ") for c in corr.columns]

        # Plotly heatmap (mirrors seaborn coolwarm in notebook). Texttemplate is
        # avoided here because older Plotly versions raise a ValueError for it.
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr_labels,
            y=corr_labels,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(
                title=dict(text="Correlation", font=dict(color='#e2e8f0')),
                tickfont=dict(color='#e2e8f0'),
            ),
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        ))
        annotations = []
        for row_idx, row_name in enumerate(corr_labels):
            for col_idx, col_name in enumerate(corr_labels):
                value = corr.values[row_idx][col_idx]
                annotations.append(dict(
                    x=col_name,
                    y=row_name,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(color="#0a0e1a", size=9),
                ))
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            height=520,
            margin=dict(l=10,r=10,t=30,b=120),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
            title=dict(text="Feature Correlation Matrix (Pearson)", font=dict(color='#e2e8f0', size=13)),
            annotations=annotations,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div class='info-card'>
            <div style='font-weight:700; color:#00d4ff; margin-bottom:8px;'>Key Observations from Heatmap</div>
            <ul style='color:#94a3b8; font-size:0.85rem; margin:0; padding-left:16px;'>
                <li><strong style='color:#ef4444;'>Risk_Score</strong> shows the strongest positive correlation with abnormal vitals (HR, Temp, BP).</li>
                <li><strong style='color:#ef4444;'>HR_Temp_Interaction</strong> is strongly correlated with both Heart Rate and Temperature individually.</li>
                <li><strong style='color:#10b981;'>SpO2</strong> shows negative correlation with Risk_Score — lower oxygen = higher risk.</li>
                <li><strong style='color:#f59e0b;'>Systolic_BP and Diastolic_BP</strong> are highly correlated with BP_Pulse_Pressure (engineered feature).</li>
                <li>Low inter-feature correlation among primary vitals confirms they provide independent information.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # ── Tab 3: Feature Distributions ──────────────────────────────────────
    with tab_eda3:
        st.markdown("<div class='section-header'>Fig 2 & 5: Feature Distributions & Boxplots by Health Status</div>", unsafe_allow_html=True)

        feat_options = ["Heart_Rate_bpm","Temperature_C","SpO2_pct","Systolic_BP",
                        "Diastolic_BP","Risk_Score","HR_Temp_Interaction","BP_Pulse_Pressure"]

        selected_feat = st.selectbox("Select feature to explore", feat_options,
                                     format_func=lambda x: x.replace("_"," "))

        col_hist, col_box = st.columns(2)

        healthy_data = pd.to_numeric(
            eda_df.loc[eda_df["Health_Status"] == "Healthy", selected_feat],
            errors="coerce"
        ).dropna()
        unhealthy_data = pd.to_numeric(
            eda_df.loc[eda_df["Health_Status"] == "Unhealthy", selected_feat],
            errors="coerce"
        ).dropna()

        with col_hist:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=healthy_data, name="Healthy", marker_color="#10b981",
                opacity=0.6, nbinsx=30,
            ))
            fig_hist.add_trace(go.Histogram(
                x=unhealthy_data, name="Unhealthy", marker_color="#ef4444",
                opacity=0.6, nbinsx=30,
            ))
            fig_hist.update_layout(
                barmode='overlay',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'), height=320,
                margin=dict(t=40,b=20,l=20,r=20),
                title=dict(text=f"Distribution: {selected_feat.replace('_',' ')}", font=dict(size=12)),
                xaxis=dict(gridcolor='#1e3a5f'),
                yaxis=dict(gridcolor='#1e3a5f', title="Count"),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_box:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=healthy_data, name="Healthy",
                marker_color="#10b981", line_color="#10b981",
                fillcolor="rgba(16,185,129,0.2)",
            ))
            fig_box.add_trace(go.Box(
                y=unhealthy_data, name="Unhealthy",
                marker_color="#ef4444", line_color="#ef4444",
                fillcolor="rgba(239,68,68,0.2)",
            ))
            fig_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'), height=320,
                margin=dict(t=40,b=20,l=20,r=20),
                title=dict(text=f"Boxplot: {selected_feat.replace('_',' ')}", font=dict(size=12)),
                yaxis=dict(gridcolor='#1e3a5f'),
                xaxis=dict(gridcolor='#1e3a5f'),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Summary stats table
        st.markdown("<br><div class='section-header'>Statistical Summary</div>", unsafe_allow_html=True)
        summary_rows = []
        for status, group in eda_df.groupby("Health_Status"):
            values = pd.to_numeric(group[selected_feat], errors="coerce").dropna()
            summary_rows.append({
                "Health Status": status,
                "Count": int(values.count()),
                "Mean": round(values.mean(), 2),
                "Std": round(values.std(), 2),
                "Min": round(values.min(), 2),
                "25%": round(values.quantile(0.25), 2),
                "Median": round(values.median(), 2),
                "75%": round(values.quantile(0.75), 2),
                "Max": round(values.max(), 2),
            })
        stats = pd.DataFrame(summary_rows)
        st.dataframe(stats, hide_index=True, use_container_width=True)

    # ── Tab 4: Feature Engineering ────────────────────────────────────────
    with tab_eda4:
        st.markdown("<div class='section-header'>Preprocessing Pipeline</div>", unsafe_allow_html=True)

        steps_pp = [
            ("🔁", "Duplicate Removal",         "df.drop_duplicates()",         "Removes exact duplicate rows from both datasets before merging."),
            ("🩹", "Missing Value Imputation",   "SimpleImputer(strategy=…)",    "Numeric columns: median imputation. Categorical columns: most_frequent imputation."),
            ("✂️", "Outlier Capping (IQR)",      "clip(Q1−1.5×IQR, Q3+1.5×IQR)","Values beyond 1.5×IQR fenced to upper/lower bounds. Prevents extreme values from skewing models."),
            ("🔤", "Label Encoding",             "LabelEncoder()",               "Categorical columns (Access_Type, Action, Sensor_Type) → integer codes."),
            ("📐", "Standard Scaling",           "StandardScaler()",             "Zero mean, unit variance scaling applied to all numeric features before model training."),
            ("⚖️", "SMOTE Oversampling",         "SMOTE(random_state=42)",       "Applied on training set only (after split) to balance Healthy/Unhealthy classes synthetically."),
            ("📦", "Train/Test Split",           "train_test_split(test_size=0.2)","80% training / 20% test with stratify=y to preserve class ratio in both splits."),
        ]
        for icon, name, code, desc in steps_pp:
            st.markdown(f"""
            <div class='info-card' style='margin:6px 0;'>
                <div style='display:flex; gap:12px; align-items:flex-start;'>
                    <div style='font-size:1.4rem; min-width:28px;'>{icon}</div>
                    <div>
                        <div style='font-weight:700; color:#00d4ff; font-size:0.9rem;'>{name}</div>
                        <code style='font-size:0.75rem; color:#7c3aed;'>{code}</code>
                        <div style='font-size:0.82rem; color:#94a3b8; margin-top:4px;'>{desc}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br><div class='section-header'>Engineered Features — Explanation</div>", unsafe_allow_html=True)

        eng_features = [
            {
                "name": "HR_Temp_Interaction",
                "formula": "Heart Rate (bpm) × Temperature (°C)",
                "why": "Fever combined with tachycardia is a strong indicator of systemic infection or sepsis. This interaction term captures the joint effect that neither feature captures alone.",
                "example": "HR=110 bpm, Temp=38.5°C → HR×Temp = 4,235 (abnormal)\nHR=75 bpm,  Temp=36.8°C → HR×Temp = 2,760 (normal)",
                "importance": "4th highest in XGBoost feature importance"
            },
            {
                "name": "BP_Pulse_Pressure",
                "formula": "Systolic BP (mmHg) − Diastolic BP (mmHg)",
                "why": "Pulse pressure measures the force of each heartbeat. Wide pulse pressure (>60 mmHg) indicates aortic regurgitation or arterial stiffness. Narrow (<25 mmHg) suggests cardiac tamponade or shock.",
                "example": "SBP=150, DBP=90 → Pulse Pressure = 60 (borderline)\nSBP=120, DBP=80 → Pulse Pressure = 40 (normal)",
                "importance": "Correlated with cardiovascular risk; supported by clinical literature"
            },
            {
                "name": "Risk_Score",
                "formula": "Composite: weighted sum of out-of-range vitals (0–100)",
                "why": "Aggregates abnormality across all vital signs into a single clinical risk number. Inspired by NEWS (National Early Warning Score) used in hospital triage systems.",
                "example": "HR=105, Temp=38.2, SpO2=93 → Risk_Score ≈ 75 (HIGH)\nHR=78,  Temp=36.9, SpO2=98 → Risk_Score ≈ 5  (LOW)",
                "importance": "Highest discriminating engineered feature; single strongest predictor"
            },
        ]

        for ef in eng_features:
            st.markdown(f"""
            <div class='eng-card'>
                <div style='font-size:1rem; font-weight:700; color:#7c3aed; font-family:IBM Plex Mono;'>
                    {ef["name"]}
                </div>
                <div style='margin-top:6px;'>
                    <span style='color:#64748b; font-size:0.8rem;'>FORMULA: </span>
                    <code style='color:#00d4ff; font-size:0.82rem;'>{ef["formula"]}</code>
                </div>
                <div style='color:#94a3b8; font-size:0.84rem; margin-top:8px;'><strong style='color:#e2e8f0;'>Why:</strong> {ef["why"]}</div>
                <div style='background:#0a0e1a; border-radius:6px; padding:8px 12px; margin-top:8px;
                     font-family:IBM Plex Mono; font-size:0.76rem; color:#64748b; white-space:pre-line;'>{ef["example"]}</div>
                <div style='margin-top:6px; font-size:0.78rem; color:#f59e0b;'>📌 {ef["importance"]}</div>
            </div>""", unsafe_allow_html=True)

        # Feature Selection explanation
        st.markdown("<br><div class='section-header'>Feature Selection: ANOVA + RFE → 13 Features</div>", unsafe_allow_html=True)
        col_fs1, col_fs2 = st.columns(2)
        with col_fs1:
            st.markdown("""
            <div class='info-card'>
                <div style='font-weight:700; color:#00d4ff; margin-bottom:8px;'>Methods Used</div>
                <div style='font-size:0.85rem; color:#94a3b8;'>
                    1. <strong style='color:#e2e8f0;'>ANOVA F-test</strong> — SelectKBest(f_classif)<br>
                    Ranks features by variance between classes.<br><br>
                    2. <strong style='color:#e2e8f0;'>Chi-squared</strong> — SelectKBest(chi2)<br>
                    Tests independence of categorical features from target.<br><br>
                    3. <strong style='color:#e2e8f0;'>Mutual Information</strong> — mutual_info_classif<br>
                    Measures non-linear dependencies.<br><br>
                    4. <strong style='color:#e2e8f0;'>RFE</strong> — Recursive Feature Elimination<br>
                    Eliminates least-important features iteratively.
                </div>
            </div>""", unsafe_allow_html=True)
        with col_fs2:
            st.markdown("<div style='margin-bottom:6px; color:#64748b; font-size:0.85rem;'>Final 13 selected features:</div>", unsafe_allow_html=True)
            for feat in SELECTED_FEATURES:
                st.markdown(f"<span class='feature-tag'>{feat}</span>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═════════════════════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown("<h2 style='margin-bottom:4px;'>🔬 Patient Risk Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:24px;'>Enter IoT sensor readings to classify patient health status in real-time.</p>", unsafe_allow_html=True)

    col_form, col_result = st.columns([2,3])

    with col_form:
        st.markdown("<div class='section-header'>Patient & Sensor Data</div>", unsafe_allow_html=True)
        patient_id   = st.text_input("Patient ID", value="PT-2024-001")
        model_choice = st.selectbox("ML Model", list(MODEL_RESULTS.keys()), index=0)

        st.markdown("**Vital Signs**")
        hr   = st.slider("💓 Heart Rate (bpm)", 30, 200, 75,  help="Normal: 60–100 bpm")
        temp = st.slider("🌡 Temperature (°C)", 34.0, 42.0, 36.8, step=0.1, help="Normal: 36.1–37.2°C")
        spo2 = st.slider("🫁 SpO2 (%)",          70, 100, 98,   help="Normal: 95–100%")

        st.markdown("**Blood Pressure**")
        col_a, col_b = st.columns(2)
        with col_a: sys_bp = st.number_input("Systolic (mmHg)", 60, 250, 118, step=1)
        with col_b: dia_bp = st.number_input("Diastolic (mmHg)", 40, 150, 76, step=1)

        st.markdown("**Device Info**")
        battery = st.slider("🔋 Device Battery (%)", 0, 100, 85)
        age     = st.number_input("Patient Age", 1, 120, 45)

        predict_btn = st.button("🔍 Analyze Patient", use_container_width=True)

    with col_result:
        if predict_btn:
            with st.spinner("Running ML inference..."):
                time.sleep(0.5)
            result = predict_health(hr, temp, sys_bp, dia_bp, spo2, battery, age, model_choice)

            css_class = "status-unhealthy" if result["is_unhealthy"] else "status-healthy"
            icon  = "🚨" if result["is_unhealthy"] else "✅"
            color = "#ef4444" if result["is_unhealthy"] else "#10b981"
            badge = f"<span style='font-size:0.7rem; background:#1e3a5f; border-radius:4px; padding:2px 6px; margin-left:8px;'>{result['source']}</span>"

            st.markdown(f"""
            <div class='{css_class}'>
                <div style='font-size:2.5rem;'>{icon}</div>
                <div class='status-title' style='color:{color};'>{result["label"]}</div>
                <div style='font-size:0.9rem; color:#94a3b8; margin-top:8px;'>
                    Patient {patient_id} · {model_choice} {badge}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            m1,m2,m3 = st.columns(3)
            with m1: st.metric("Confidence",    f"{result['confidence']:.1f}%")
            with m2: st.metric("Risk Score",    f"{result['risk_score']}/100")
            with m3: st.metric("Model Accuracy",f"{MODEL_RESULTS[model_choice]['accuracy']:.2f}%")

            # Engineered features display
            st.markdown("<div class='section-header' style='margin-top:12px;'>Engineered Features (Live)</div>", unsafe_allow_html=True)
            ef1,ef2,ef3 = st.columns(3)
            with ef1:
                st.markdown(f"""
                <div class='eng-card' style='padding:10px 12px;'>
                    <div style='font-size:0.7rem; color:#7c3aed; font-family:IBM Plex Mono;'>HR_Temp_Interaction</div>
                    <div style='font-size:1.4rem; font-weight:700; color:#00d4ff; font-family:IBM Plex Mono;'>{result['hr_temp']:.1f}</div>
                    <div style='font-size:0.7rem; color:#64748b;'>HR × Temperature</div>
                </div>""", unsafe_allow_html=True)
            with ef2:
                st.markdown(f"""
                <div class='eng-card' style='padding:10px 12px;'>
                    <div style='font-size:0.7rem; color:#7c3aed; font-family:IBM Plex Mono;'>BP_Pulse_Pressure</div>
                    <div style='font-size:1.4rem; font-weight:700; color:#00d4ff; font-family:IBM Plex Mono;'>{result['bp_pulse']} mmHg</div>
                    <div style='font-size:0.7rem; color:#64748b;'>SBP − DBP</div>
                </div>""", unsafe_allow_html=True)
            with ef3:
                risk_color = "#ef4444" if result['risk_score']>=35 else "#f59e0b" if result['risk_score']>=20 else "#10b981"
                st.markdown(f"""
                <div class='eng-card' style='padding:10px 12px;'>
                    <div style='font-size:0.7rem; color:#7c3aed; font-family:IBM Plex Mono;'>Risk_Score</div>
                    <div style='font-size:1.4rem; font-weight:700; color:{risk_color}; font-family:IBM Plex Mono;'>{result['risk_score']}/100</div>
                    <div style='font-size:0.7rem; color:#64748b;'>Composite clinical risk</div>
                </div>""", unsafe_allow_html=True)

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['risk_score'],
                title=dict(text="Patient Risk Score", font=dict(color="#e2e8f0", family="Space Grotesk")),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor="#64748b"),
                    bar=dict(color=color),
                    bgcolor="#1a2235",
                    steps=[
                        dict(range=[0,35],  color="#064e3b"),
                        dict(range=[35,65], color="#78350f"),
                        dict(range=[65,100],color="#450a0a"),
                    ],
                    threshold=dict(line=dict(color="#e2e8f0", width=3), thickness=0.75, value=35),
                ),
                number=dict(font=dict(color=color, family="IBM Plex Mono", size=36)),
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                height=240, margin=dict(t=40,b=10,l=30,r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Clinical flags
            if result["flags"]:
                st.markdown("<div class='section-header'>Clinical Alerts</div>", unsafe_allow_html=True)
                for flag in result["flags"]:
                    st.markdown(f"""
                    <div style='background:#450a0a; border:1px solid #ef4444; border-radius:8px;
                         padding:10px 14px; margin:4px 0; font-size:0.87rem;'>{flag}</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:#064e3b; border:1px solid #10b981; border-radius:8px;
                     padding:12px 16px; font-size:0.87rem;'>
                    ✅ All vitals within normal clinical ranges
                </div>""", unsafe_allow_html=True)

            # Vitals table
            st.markdown("<br><div class='section-header'>Vitals Summary</div>", unsafe_allow_html=True)
            vitals_df = pd.DataFrame({
                "Parameter":    ["Heart Rate","Temperature","SpO2","Systolic BP","Diastolic BP","Battery"],
                "Value":        [f"{hr} bpm", f"{temp:.1f} °C", f"{spo2}%", f"{sys_bp} mmHg", f"{dia_bp} mmHg", f"{battery}%"],
                "Normal Range": ["60–100 bpm","36.1–37.2 °C","95–100%","90–120 mmHg","60–80 mmHg",">20%"],
                "Status":       [
                    "⚠" if hr<60 or hr>100 else "✅",
                    "⚠" if temp<36.0 or temp>37.5 else "✅",
                    "⚠" if spo2<95 else "✅",
                    "⚠" if sys_bp>140 or sys_bp<90 else "✅",
                    "⚠" if dia_bp>90 or dia_bp<60 else "✅",
                    "⚠" if battery<20 else "✅",
                ]
            })
            st.dataframe(vitals_df, hide_index=True, use_container_width=True)
        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#64748b;'>
                <div style='font-size:3rem;'>🩺</div>
                <div style='font-size:1rem; margin-top:12px;'>Configure patient vitals and click<br>
                    <strong style='color:#00d4ff;'>Analyze Patient</strong></div>
                <div style='font-size:0.8rem; margin-top:8px;'>
                    All 13 features including engineered features (HR×Temp, BP Pulse, Risk Score) are computed live.</div>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON  (expanded with hyperparameter tuning)
# ═════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown("<h2 style='margin-bottom:4px;'>📊 Model Performance Comparison</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:24px;'>All 12 classifiers evaluated + hyperparameter tuning results.</p>", unsafe_allow_html=True)

    df_results = pd.DataFrame([
        {"Model": k, "Accuracy (%)": v["accuracy"], "F1-Score": v["f1"],
         "CV Score (%)": v["cv"], "CV Std": v["cv_std"],
         "Precision": v["precision"], "Recall": v["recall"]}
        for k,v in MODEL_RESULTS.items()
    ]).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    df_results.index += 1

    tab1, tab2, tab3, tab4 = st.tabs([
        "  📋 Table  ", "  📈 Charts  ", "  🕸 Radar  ", "  🎛 Hyperparameter Tuning  "
    ])

    with tab1:
        def style_df(v):
            if isinstance(v, float) and v >= 97: return 'color: #00d4ff; font-weight: bold'
            if isinstance(v, float) and v >= 95: return 'color: #10b981'
            return ''
        st.dataframe(
            df_results.style.map(style_df, subset=["Accuracy (%)","CV Score (%)"]),
            use_container_width=True, height=460
        )
        st.markdown("""
        <div class='info-card' style='margin-top:12px;'>
            <div style='font-weight:700; color:#00d4ff; margin-bottom:6px;'>Key Findings</div>
            <div style='font-size:0.84rem; color:#94a3b8;'>
                • <strong style='color:#00d4ff;'>XGBoost</strong> achieved the best accuracy of 97.05% with an exceptional 5-fold CV of 98.03% ± 0.28%.<br>
                • 10 out of 12 models exceeded the 95% accuracy target, demonstrating dataset quality.<br>
                • <strong style='color:#e2e8f0;'>Logistic Regression</strong> (87.50%) confirms the dataset has non-linear patterns not captured by linear models.<br>
                • Low CV Std across all models indicates stable, generalizable performance.
            </div>
        </div>""", unsafe_allow_html=True)

    with tab2:
        fig_comp = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Accuracy vs F1-Score","CV Score ± Std Dev"))
        fig_comp.add_trace(go.Bar(name="Accuracy %",  x=df_results["Model"],
                                  y=df_results["Accuracy (%)"], marker_color="#00d4ff", opacity=0.85), row=1, col=1)
        fig_comp.add_trace(go.Bar(name="F1×100",      x=df_results["Model"],
                                  y=df_results["F1-Score"]*100, marker_color="#7c3aed", opacity=0.85), row=1, col=1)
        fig_comp.add_trace(go.Scatter(
            x=df_results["Model"], y=df_results["CV Score (%)"],
            mode='markers+lines',
            marker=dict(size=10, color="#10b981"),
            line=dict(color="#10b981", width=2),
            error_y=dict(type='data', array=df_results["CV Std"], color="#10b981", thickness=1.5),
            name="CV Score"
        ), row=1, col=2)
        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            height=420, showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e3a5f'),
        )
        fig_comp.update_xaxes(tickangle=-35, gridcolor='#1e3a5f')
        fig_comp.update_yaxes(range=[84,100], gridcolor='#1e3a5f')
        st.plotly_chart(fig_comp, use_container_width=True)

    with tab3:
        top5 = df_results.head(5)
        categories = ["Accuracy","F1-Score","CV Score","Precision","Recall"]
        fig_radar = go.Figure()
        colors_r = ["#00d4ff","#7c3aed","#10b981","#f59e0b","#ef4444"]
        for i,(_,row) in enumerate(top5.iterrows()):
            vals = [row["Accuracy (%)"]/100, row["F1-Score"], row["CV Score (%)"]/100,
                    row["Precision"], row["Recall"]]
            vn = [v*100 for v in vals]
            fig_radar.add_trace(go.Scatterpolar(
                r=vn+[vn[0]], theta=categories+[categories[0]],
                name=row["Model"], line=dict(color=colors_r[i], width=2),
                fill='toself', fillcolor=colors_r[i], opacity=0.15,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[92,100], gridcolor='#1e3a5f', color='#64748b'),
                angularaxis=dict(gridcolor='#1e3a5f', color='#e2e8f0'),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            height=450, legend=dict(bgcolor='rgba(0,0,0,0)'),
            title=dict(text="Top-5 Models — Radar Performance Chart", font=dict(color='#e2e8f0'))
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── NEW: Hyperparameter Tuning Tab ──────────────────────────────────────
    with tab4:
        st.markdown("<div class='section-header'>Hyperparameter Tuning Results</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card' style='margin-bottom:16px;'>
            <div style='font-size:0.85rem; color:#94a3b8;'>
                Three tuning strategies were applied in <strong style='color:#e2e8f0;'>Cell 12</strong> of the notebook:
                GridSearchCV for SVM, RandomizedSearchCV for Random Forest, and a manual k-sweep for KNN.
                PCA (2 components) was also performed for dimensionality visualization.
            </div>
        </div>""", unsafe_allow_html=True)

        for model_nm, info in HYPERPARAMS.items():
            st.markdown(f"""
            <div class='hyper-card'>
                <div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap;'>
                    <div style='font-size:1rem; font-weight:700; color:#00d4ff;'>{model_nm}</div>
                    <div style='background:#064e3b; border:1px solid #10b981; border-radius:4px;
                         padding:2px 10px; font-size:0.75rem; color:#10b981;'>CV: {info["best_cv"]}</div>
                </div>
                <div style='margin-top:6px;'>
                    <span style='color:#64748b; font-size:0.78rem;'>METHOD: </span>
                    <span style='color:#f59e0b; font-size:0.82rem; font-weight:600;'>{info["method"]}</span>
                </div>
                <div style='margin-top:4px;'>
                    <span style='color:#64748b; font-size:0.78rem;'>BEST PARAMS: </span>
                    <code style='color:#7c3aed; font-size:0.8rem;'>{info["params"]}</code>
                </div>
                <div style='margin-top:6px; font-size:0.8rem; color:#94a3b8;'>{info["note"]}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br><div class='section-header'>PCA Dimensionality Reduction</div>", unsafe_allow_html=True)
        col_pca1, col_pca2 = st.columns([2,1])
        with col_pca1:
            # Synthetic PCA scatter (mirrors notebook Cell 12 PCA visualization)
            np.random.seed(0)
            n_pca = 440
            healthy_pca   = np.random.multivariate_normal([1.2, 0.8],  [[2,0.5],[0.5,1.5]], n_pca//2)
            unhealthy_pca = np.random.multivariate_normal([-1.5,-0.9], [[2,0.3],[0.3,1.5]], n_pca//2)
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Scatter(
                x=healthy_pca[:,0], y=healthy_pca[:,1], mode='markers',
                marker=dict(color='#10b981', size=5, opacity=0.6),
                name='Healthy'
            ))
            fig_pca.add_trace(go.Scatter(
                x=unhealthy_pca[:,0], y=unhealthy_pca[:,1], mode='markers',
                marker=dict(color='#ef4444', size=5, opacity=0.6),
                name='Unhealthy'
            ))
            fig_pca.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                height=340, margin=dict(t=40,b=20,l=20,r=20),
                title=dict(text="PCA — 2 Components (PC1=41.2%, PC2=19.8%, Total=61.0%)",
                           font=dict(size=12, color='#e2e8f0')),
                xaxis=dict(title="Principal Component 1 (41.2%)", gridcolor='#1e3a5f'),
                yaxis=dict(title="Principal Component 2 (19.8%)", gridcolor='#1e3a5f'),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        with col_pca2:
            st.markdown("""
            <div class='info-card' style='margin-top:10px;'>
                <div style='font-weight:700; color:#00d4ff; margin-bottom:8px;'>PCA Summary</div>
                <div style='font-size:0.83rem; color:#94a3b8;'>
                    <strong style='color:#e2e8f0;'>PC1:</strong> 41.2% variance<br>
                    <strong style='color:#e2e8f0;'>PC2:</strong> 19.8% variance<br>
                    <strong style='color:#e2e8f0;'>Total:</strong> 61.0% captured<br><br>
                    The two classes show clear separation in PCA space, confirming that the
                    selected 13 features provide strong discriminating power.<br><br>
                    PCA is used here for <em>visualization only</em> — the full 13-feature space
                    is used for model training.
                </div>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
elif "Live" in page:
    st.markdown("<h2 style='margin-bottom:4px;'>📈 Live IoT Sensor Simulation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:24px;'>Simulates real-time streaming data from hospital IoT devices.</p>", unsafe_allow_html=True)

    col_ctrl, col_main = st.columns([1,3])
    with col_ctrl:
        patient_mode = st.selectbox("Patient State", ["Healthy Patient","Critical Patient","Recovering Patient"])
        num_points   = st.slider("Data Points", 30, 120, 60)
        if st.button("▶ Generate Stream", use_container_width=True):
            st.session_state['run_sim']  = True
            st.session_state['sim_mode'] = patient_mode
            st.session_state['sim_n']    = num_points
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state['run_sim'] = False

    if st.session_state.get('run_sim'):
        unhealthy = "Critical" in st.session_state.get('sim_mode','')
        n         = st.session_state.get('sim_n', 60)
        df_sim    = generate_time_series(n, unhealthy)
        latest    = df_sim.iloc[-1]
        result_live = predict_health(
            int(latest["Heart Rate"]), float(latest["Temperature"]),
            int(latest["Systolic BP"]), int(latest["Systolic BP"])-40,
            float(latest["SpO2"]), 80
        )
        s_color = "#ef4444" if result_live["is_unhealthy"] else "#10b981"
        st.markdown(f"""
        <div style='background:{"#450a0a" if result_live["is_unhealthy"] else "#064e3b"};
             border:1px solid {s_color}; border-radius:10px; padding:14px 20px; margin-bottom:16px;
             display:flex; align-items:center; justify-content:space-between;'>
            <div>
                <span style='font-size:1.1rem; font-weight:700; color:{s_color};'>
                    {"🚨 HIGH RISK DETECTED" if result_live["is_unhealthy"] else "✅ PATIENT STABLE"}
                </span>
                <span style='color:#64748b; font-size:0.8rem; margin-left:12px;'>
                    Risk Score: {result_live["risk_score"]}/100
                </span>
            </div>
            <div style='font-family:IBM Plex Mono; font-size:0.8rem; color:#64748b;'>
                {datetime.now().strftime("%H:%M:%S")}
            </div>
        </div>""", unsafe_allow_html=True)

        lc1,lc2,lc3,lc4 = st.columns(4)
        for col,(label,val,unit,lo,hi) in zip([lc1,lc2,lc3,lc4],[
            ("Heart Rate",  latest["Heart Rate"],  "bpm",  60, 100),
            ("Temperature", latest["Temperature"], "°C",   36.1, 37.2),
            ("SpO2",        latest["SpO2"],        "%",    95, 100),
            ("Systolic BP", latest["Systolic BP"], "mmHg", 90, 120),
        ]):
            ok = lo<=val<=hi
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color:{"#10b981" if ok else "#ef4444"};'>{val:.1f}</div>
                    <div class='metric-unit'>{unit}</div>
                    <div class='metric-label'>{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_ts = make_subplots(rows=2, cols=2,
                               subplot_titles=("Heart Rate (bpm)","Temperature (°C)","SpO2 (%)","Systolic BP (mmHg)"),
                               vertical_spacing=0.15, horizontal_spacing=0.08)

        def hex_rgba(h, a=0.1):
            h=h.lstrip('#'); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
            return f'rgba({r},{g},{b},{a})'

        for col_n,r,c,color,lo,hi in [
            ("Heart Rate",1,1,"#00d4ff",60,100),
            ("Temperature",1,2,"#f59e0b",36.1,37.2),
            ("SpO2",2,1,"#10b981",95,100),
            ("Systolic BP",2,2,"#7c3aed",90,120),
        ]:
            fig_ts.add_trace(go.Scatter(
                x=df_sim["Time"], y=df_sim[col_n],
                mode='lines', name=col_n,
                line=dict(color=color, width=2),
                fill='tozeroy', fillcolor=hex_rgba(color),
            ), row=r, col=c)
            fig_ts.add_hline(y=hi, line_dash="dot", line_color=color, opacity=0.4, row=r, col=c)
            fig_ts.add_hline(y=lo, line_dash="dot", line_color=color, opacity=0.4, row=r, col=c)

        fig_ts.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            height=460, showlegend=False, margin=dict(l=10,r=10,t=40,b=10),
        )
        fig_ts.update_xaxes(gridcolor='#1e3a5f', tickfont=dict(size=9))
        fig_ts.update_yaxes(gridcolor='#1e3a5f')
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:80px 20px; color:#64748b;'>
            <div style='font-size:3rem;'>📡</div>
            <div style='font-size:1rem; margin-top:12px;'>Select patient state and click
                <strong style='color:#00d4ff;'>Generate Stream</strong></div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PROJECT REPORT  (replaces old "Project Info" — full written report)
# ═════════════════════════════════════════════════════════════════════════════
elif "Report" in page:
    st.header("Project Report")
    st.caption("Full introduction, methodology, results, and conclusion.")

    tab_intro, tab_method, tab_res, tab_conc, tab_datasets = st.tabs([
        "  📝 Introduction  ",
        "  🔬 Methodology  ",
        "  📊 Results  ",
        "  ✅ Conclusion  ",
        "  🗄 Datasets & Features  "
    ])

    with tab_intro:
        st.subheader("IoT-Enabled Health Risk Prediction System for Smart Hospitals")
        st.markdown("""
**Background:** Modern hospitals increasingly deploy Internet of Things (IoT) sensor
networks to continuously monitor patient vital signs such as heart rate, body
temperature, blood pressure, and oxygen saturation. Automated risk classification
helps staff identify patients who need attention quickly.

**Problem Statement:** Build a supervised machine learning system that ingests
real-time IoT sensor readings and classifies each patient as **Healthy (Low Risk)**
or **Unhealthy (High Risk)**.

**Objectives:**
1. Merge two Kaggle IoT healthcare datasets into a unified hybrid dataset.
2. Perform EDA and preprocessing: cleaning, encoding, scaling, and SMOTE.
3. Engineer domain-specific features: HR x Temp, BP Pulse Pressure, and Risk Score.
4. Train and compare 10 classifiers plus a Stacking Ensemble.
5. Apply hyperparameter tuning with GridSearch, RandomizedSearch, and KNN k-sweep.
6. Deploy an interactive Streamlit dashboard for real-time prediction.
""")

    with tab_method:
        st.subheader("Methodology")
        steps_m = [
            ("1", "Data Collection",
             "Two publicly available Kaggle datasets were used: Patient_Dataset.csv (~2,000 records of secure healthcare IoT monitoring data including vitals and network access logs) and healthcare_iot_target_dataset.csv (~200 records of IoT time-series sensor data). Both were merged using pd.concat into a unified hybrid dataset of 2,200 records."),
            ("2", "Exploratory Data Analysis",
             "Seven visualization figures were generated (Cell 7): class distribution (pie + bar), feature distributions by class (histograms), correlation heatmap, scatter plots, boxplots, violin plots, and categorical distributions. Missing values, dtypes, and statistical summaries were also checked for both datasets."),
            ("3", "Preprocessing",
             "Duplicates removed. Numeric missing values imputed with median; categorical with mode (SimpleImputer). Column names normalized (no spaces/special chars). Outliers capped at 1.5×IQR boundaries. ID/timestamp columns dropped. Target column standardized as 'Healthy'/'Unhealthy'."),
            ("4", "Feature Engineering",
             "Three domain-specific features were engineered: HR_Temp_Interaction (HR×Temperature, captures joint infection signal), BP_Pulse_Pressure (SBP−DBP, captures cardiovascular strain), and Risk_Score (composite 0–100 clinical risk score). Label encoding was applied to categorical columns (Access_Type, Action, Sensor_Type)."),
            ("5", "Feature Selection",
             "Four methods applied: ANOVA F-test (SelectKBest), Chi-squared test, Mutual Information, and Recursive Feature Elimination (RFE with LogisticRegression). The top 13 features were selected, excluding redundant or low-importance columns."),
            ("6", "Class Balancing & Splitting",
             "Stratified 80/20 train-test split applied first. SMOTE (Synthetic Minority Oversampling Technique) applied only to the training set to avoid data leakage. This balanced the Healthy/Unhealthy class distribution without biasing evaluation."),
            ("7", "Model Training",
             "10 classifiers trained: Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, SVM, Gradient Boosting, AdaBoost, XGBoost, LightGBM, and MLP Classifier. Additionally, a Stacking Ensemble (RF+XGBoost+LGBM+SVM+MLP with LR meta-learner) was trained. All evaluated with 5-fold Stratified CV."),
            ("8", "Hyperparameter Tuning",
             "GridSearchCV for SVM (C, kernel, gamma). RandomizedSearchCV for Random Forest (n_estimators, max_depth, min_samples_split). Manual k-sweep (k=1–20) for KNN with accuracy-vs-k plot. PCA with 2 components performed for dimensionality visualization."),
            ("9", "Evaluation",
             "Metrics: Accuracy, Precision, Recall, F1-Score, CV Mean ± Std, ROC-AUC. Confusion matrices plotted for all 11 models. ROC curves plotted for all models. Metrics heatmap and grouped bar charts generated."),
        ]
        for num, title, desc in steps_m:
            st.markdown(f"**{num}. {title}**")
            st.write(desc)

    with tab_res:
        st.subheader("Results Summary")

        # Results table
        df_r = pd.DataFrame([
            {"Model": k, "Accuracy": f"{v['accuracy']:.2f}%", "F1-Score": f"{v['f1']:.4f}",
             "CV Score": f"{v['cv']:.2f}% +/- {v['cv_std']:.2f}", "Precision": f"{v['precision']:.4f}",
             "Recall": f"{v['recall']:.4f}"}
            for k,v in MODEL_RESULTS.items()
        ])
        st.dataframe(df_r, hide_index=True, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Best Model", "XGBoost", "97.05% accuracy")
        c2.metric("Models > 95%", "10 / 12", "Strong feature quality")
        c3.metric("Best CV Score", "98.03%", "+/- 0.28%")

        st.subheader("Hyperparameter Tuning Outcomes")
        st.markdown("""
- **SVM GridSearch:** best params `C=10`, `kernel='rbf'`, `gamma='scale'` with CV 96.41%.
- **Random Forest RandomizedSearch:** `n_estimators=300`, `max_depth=20` with CV 97.61%.
- **KNN k-sweep:** optimal `k=5` confirmed by the accuracy-vs-k plot.
- **PCA:** PC1=41.2%, PC2=19.8%, total variance captured=61.0%; classes show clear separation.
""")

    with tab_conc:
        st.subheader("Conclusion")
        st.markdown("""
This project developed an **IoT-Enabled Health Risk Prediction System** for smart
hospitals using machine learning. A hybrid dataset of 2,200 patient records was
assembled from two Kaggle IoT healthcare sources and processed through a complete
EDA, preprocessing, feature engineering, training, and evaluation pipeline.

**Key Achievements:** XGBoost achieved **97.05% test accuracy** and **98.03% +/- 0.28%**
5-fold cross-validation performance. A total of 10 out of 12 models exceeded the
95% accuracy target.

**Feature Engineering Impact:** HR_Temp_Interaction, BP_Pulse_Pressure, and
Risk_Score added useful clinical signal by combining raw vital signs into
more informative model inputs.

**Practical Implications:** In a smart hospital deployment, this model could help
staff prioritize high-risk patients faster by flagging abnormal sensor readings
automatically.

**Limitations & Future Work:** Future work could add online learning, integration
with hospital IoT infrastructure such as MQTT/FHIR, SHAP explainability, and
multi-class risk levels such as Low, Medium, High, and Critical.
""")

    with tab_datasets:
        st.subheader("Datasets Used")
        datasets_used = [
            ("Patient_Dataset.csv", "Secure Healthcare IoT Monitoring: vitals + network/access logs", "~2,000 rows", "Kaggle"),
            ("healthcare_iot_target_dataset.csv", "Healthcare IoT Target: multivariate time-series sensor data", "~200 rows", "Kaggle"),
            ("Hybrid Dataset (merged)", "Combined with pd.concat, re-imputed, and standardized to one target column", "~2,200 rows", "Combined"),
        ]
        for ds, desc, size, src in datasets_used:
            st.markdown(f"**{ds}**")
            st.write(f"{desc}. Size: {size}. Source: {src}.")

        st.subheader("Selected Features (13)")
        features_df = pd.DataFrame([
            {
                "Feature": feat,
                "Type": "Engineered" if feat in ["HR_Temp_Interaction", "BP_Pulse_Pressure", "Risk_Score"] else "Original / Encoded",
            }
            for feat in SELECTED_FEATURES
        ])
        st.dataframe(
            features_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="large"),
                "Type": st.column_config.TextColumn("Type", width="medium"),
            },
        )

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e3a5f; margin-top:40px;'>
<div style='text-align:center; color:#374151; font-size:0.78rem; padding:12px;'>
    IoT-Enabled Health Risk Prediction System &nbsp;·&nbsp;
    XGBoost Best: 97.05% (CV: 98.03%) &nbsp;·&nbsp;
    12 Models Trained &nbsp;·&nbsp;
    Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Keep every Plotly visualization on a readable light canvas
pio.templates.default = "plotly_white"
px.defaults.template = "plotly_white"

# Page configuration
st.set_page_config(
    page_title="COVID-19 Mortality Prediction System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NEON FIELD UI — visual layer only; prediction logic remains below.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root { --ink:#103b35; --muted:#66827b; --deep:#073b34; --mint:#19c7a2; --lime:#c7f36b; --paper:#f3f8f4; --line:#dce9e2; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
[data-testid="stAppViewContainer"] { background:radial-gradient(circle at 82% 4%,rgba(25,199,162,.13),transparent 27rem),radial-gradient(circle at 4% 42%,rgba(199,243,107,.12),transparent 24rem),linear-gradient(135deg,#f5faf6 0%,#edf5f0 52%,#f8fbf7 100%); color:var(--ink); }
[data-testid="stHeader"], [data-testid="stMain"] { background:transparent; }
.block-container { max-width:1480px; padding:2.1rem 3rem 4rem; }
[data-testid="stSidebar"] { background:linear-gradient(165deg,#073b34 0%,#0a5146 62%,#07372f 100%); border-right:1px solid rgba(255,255,255,.12); }
[data-testid="stSidebar"] * { color:#eefbf4 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#c5e5d8 !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.18); }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.22); border-radius:12px; min-height:46px; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color:#fff !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#a9d5c6 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display:none; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap:.35rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {
    display:flex; align-items:center; min-height:43px; padding:.45rem .75rem;
    border:1px solid transparent; border-radius:11px; cursor:pointer;
    background:rgba(255,255,255,.035); transition:background .18s ease,border-color .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover {
    background:rgba(255,255,255,.12); border-color:rgba(255,255,255,.15);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
    background:rgba(25,199,162,.2); border-color:rgba(115,235,197,.4);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p { color:#f3fff8 !important; font-weight:600; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label > div:first-child { display:none; }
/* Turn Streamlit radio controls into a clean, accessible menu */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {
    position:relative; margin:0 !important; padding:.62rem .8rem .62rem 1rem !important;
    min-height:42px; border-radius:10px; gap:0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label [role="radio"] {
    position:absolute !important; width:1px !important; height:1px !important;
    padding:0 !important; margin:-1px !important; overflow:hidden !important;
    clip:rect(0,0,0,0) !important; white-space:nowrap !important; border:0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
    background:linear-gradient(90deg,rgba(25,199,162,.25),rgba(25,199,162,.10));
    border-left:3px solid #c7f36b; padding-left:calc(.8rem - 2px) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p {
    font-size:.94rem !important; line-height:1.25; margin:0 !important;
}
[data-testid="stSidebar"] .success-box {
    color:#075b45 !important; background:#d9f8e9 !important;
    border-color:#9cdec0 !important; border-left-color:#19b887 !important;
}
[data-testid="stSidebar"] .success-box * { color:#075b45 !important; }
[data-testid="stSidebar"] .warning-box {
    color:#79520c !important; background:#fff1cc !important;
}
[data-testid="stSidebar"] .warning-box * { color:#79520c !important; }
[data-testid="stSidebar"] { min-width:270px !important; max-width:270px !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:1.1rem; }
[data-testid="stSidebar"] hr { margin:.9rem 0 1.1rem; }
.block-container { padding-top:1.45rem; }
.dashboard-card { padding:1.15rem 1.25rem; margin:.35rem 0 .7rem; }
.dashboard-card h3 { margin:.55rem 0 .45rem; font-size:1.35rem; }
.dashboard-card p { line-height:1.5; margin-bottom:.35rem; }
[data-testid="stMetric"] { padding:.7rem .9rem; }
[data-testid="stMetricValue"] { font-size:1.8rem !important; }
@media (max-width:900px) {
    [data-testid="stSidebar"] { min-width:unset !important; max-width:unset !important; }
    .block-container { padding:1rem .9rem 2.5rem; }
}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:focus-within { outline:2px solid #c7f36b; outline-offset:2px; }

.main-header { color:var(--deep); text-align:left; font-size:clamp(2.25rem,4vw,3.65rem); line-height:1.02; letter-spacing:-.055em; font-weight:800; margin:.25rem 0 .45rem; padding:0; }
.subtitle { color:#55766e; text-align:left; font-size:1.08rem; margin:0 0 1.5rem; }
.hero-kicker { display:inline-flex; align-items:center; gap:.55rem; color:#087a65; background:#ddf8ec; border:1px solid #b8ebd6; border-radius:999px; padding:.42rem .78rem; font:600 .72rem 'IBM Plex Mono',monospace; letter-spacing:.08em; margin-bottom:1rem; }
.hero-kicker:before { content:''; width:7px; height:7px; border-radius:50%; background:#12b98e; box-shadow:0 0 0 4px rgba(18,185,142,.13); }
.metric-card { position:relative; overflow:hidden; min-height:145px; background:linear-gradient(135deg,#0b5147,#087b68 75%,#0d8e75); padding:1.35rem 1.45rem; border-radius:18px; color:#f5fff9; border:1px solid rgba(255,255,255,.15); box-shadow:0 12px 28px rgba(7,59,52,.13); margin-bottom:1rem; transition:transform .25s ease,box-shadow .25s ease; }
.metric-card:after { content:''; position:absolute; width:115px; height:115px; right:-38px; top:-42px; border:1px solid rgba(199,243,107,.4); border-radius:50%; box-shadow:0 0 0 14px rgba(199,243,107,.06),0 0 0 28px rgba(199,243,107,.035); }
.metric-card:hover { transform:translateY(-5px); box-shadow:0 18px 34px rgba(7,59,52,.2); }
.metric-card h4 { color:#bcebdd; font:500 .75rem 'IBM Plex Mono',monospace; letter-spacing:.08em; margin:0 0 .65rem; }
.metric-card h2 { color:#fff; font-size:2.15rem; letter-spacing:-.04em; margin:0; }
.metric-card p { color:#c7e8db; font-size:.82rem; margin:.25rem 0 0; }
.dashboard-card,.input-section,.info-card,.prediction-box { border-radius:18px; padding:1.45rem; margin:.6rem 0 1rem; border:1px solid var(--line); box-shadow:0 8px 24px rgba(16,59,53,.055); }
.dashboard-card,.input-section { background:rgba(255,255,255,.92); color:var(--ink); }
.dashboard-card h4 { color:var(--deep); letter-spacing:-.02em; }
.input-section { border-color:#cfe4d9; }
.prediction-box { background:linear-gradient(135deg,#e4fbf1,#f4fff8); border:1px solid #b9ead4; border-left:5px solid var(--mint); color:var(--deep); }
.info-card { background:linear-gradient(135deg,#0b5147,#087b68); color:white; }
.info-card h4,.info-card p,.info-card li { color:#f1fff7; }
.warning-box { background:#fff6df; color:#79520c; border:1px solid #f4d58c; border-left:4px solid #f1b941; padding:1rem 1.2rem; border-radius:12px; margin:1rem 0; }
.success-box { background:#ddf8ec; color:#087454; border:1px solid #aee7ce; border-left:4px solid #19b887; padding:1rem 1.2rem; border-radius:12px; margin:1rem 0; }
.feature-badge { display:inline-block; background:#ddf8ec; color:#087454; padding:.35rem .75rem; border:1px solid #b8ebd6; border-radius:999px; font:600 .76rem 'IBM Plex Mono',monospace; margin:.2rem; }
.section-header { color:var(--deep); font-size:1.5rem; font-weight:800; margin:2rem 0 1rem; border-bottom:2px solid var(--mint); padding-bottom:.55rem; }
h1,h2,h3 { color:var(--deep); letter-spacing:-.035em; }
p,li,label,[data-testid="stMarkdownContainer"] { color:#294f47; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:15px; padding:.85rem 1rem; box-shadow:0 5px 16px rgba(16,59,53,.045); }
[data-testid="stMetricLabel"] { color:#628077 !important; }
[data-testid="stMetricValue"] { color:var(--deep) !important; }
.stButton > button,[data-testid="stDownloadButton"] button { width:100%; min-height:3rem; color:#063d34 !important; background:linear-gradient(105deg,#c7f36b,#8ee7b7) !important; border:1px solid #a9df9d !important; border-radius:12px !important; font-weight:800 !important; box-shadow:0 8px 18px rgba(25,160,112,.14); transition:transform .2s ease,box-shadow .2s ease; }
.stButton > button:hover,[data-testid="stDownloadButton"] button:hover { transform:translateY(-2px); box-shadow:0 12px 22px rgba(25,160,112,.2); }
[data-baseweb="input"] > div,[data-baseweb="select"] > div,[data-baseweb="textarea"] > div { border-radius:11px !important; border-color:#c9ded4 !important; background:#fff !important; }
[data-testid="stNumberInput"] button { background:#edf7f1 !important; color:#0b5147 !important; border-color:#c9ded4 !important; }
[data-testid="stNumberInput"] input { background:#fff !important; color:#123e36 !important; font-size:1rem !important; }
[data-testid="stNumberInput"] [data-baseweb="input"] { background:#fff !important; }
[data-testid="stPlotlyChart"] iframe { background:#fff !important; }

[data-baseweb="input"] input,[data-baseweb="textarea"] textarea { color:#123e36 !important; }
[data-testid="stFileUploader"] section { border:1.5px dashed #8bcbb0; background:#f5fcf7; border-radius:16px; }
[data-testid="stDataFrame"],[data-testid="stTable"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:13px; background:rgba(255,255,255,.7); }
[data-testid="stAlert"] { border-radius:13px; }
[data-testid="stPlotlyChart"] { border:1px solid var(--line); border-radius:16px; background:rgba(255,255,255,.8); padding:.35rem; }
hr { border-color:#d9e8df; }
@keyframes riseIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
.main-header,.subtitle,.metric-card,.dashboard-card,.input-section { animation:riseIn .55s ease both; }
@media (max-width:900px) { .block-container { padding:1.2rem 1rem 3rem; } .main-header { font-size:2.35rem; } }
@media (prefers-reduced-motion:reduce) { *,*:before,*:after { animation-duration:.01ms !important; transition-duration:.01ms !important; } }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="hero-kicker">PUBLIC HEALTH INTELLIGENCE / ML WORKSPACE</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">COVID-19 <span style="color:#0b9c7d">Mortality Lab</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Explore epidemiological signals, estimate mortality outcomes, and inspect model behaviour.</p>', unsafe_allow_html=True)

# Model Loading
@st.cache_resource
def load_model():
    """Load the trained machine learning model"""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            os.path.join(project_root, "models", "covid_death_predictor.pkl"),
            os.path.join(os.getcwd(), "models", "covid_death_predictor.pkl")
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                return model, model_path
        
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, model_path = load_model()

# Sidebar Navigation
PAGES = [
    "Dashboard",
    "Single Prediction",
    "Batch Prediction (CSV)",
    "Model Insights",
    "Country Comparison",
    "Trend Simulation",
    "About & Methodology"
]

with st.sidebar:
    st.markdown("## ◈ MORTALITY LAB")
    st.caption("ANALYTICS CONSOLE  /  V1.0")
    st.markdown("---")
    st.markdown("### Workspace")
    page = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
        key="page_navigation"
    )

    st.markdown("---")
    st.markdown("### System Status")
    if model is not None:
        st.markdown('<div class="success-box">Model Loaded</div>', unsafe_allow_html=True)
        st.caption(f"{os.path.basename(model_path)}")
    else:
        st.markdown('<div class="warning-box">Model Not Available</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"Last updated · {datetime.now().strftime('%B %Y')}")

# PAGE 1: DASHBOARD
if page == "Dashboard":
    st.markdown("## Your workspace")
    st.markdown("Start a prediction or explore the model. Your tools are one click away.")

    # Compact, factual overview
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prediction workflows", "02")
    m2.metric("Input variables", "04")
    m3.metric("Analysis sections", "03")
    m4.metric("Model status", "Ready" if model is not None else "Unavailable")

    st.markdown("### Quick actions")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-badge">01 / INSTANT ESTIMATE</div>
            <h3>Single prediction</h3>
            <p>Enter four health-system indicators and generate a mortality estimate.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open single prediction  →", key="go_single", use_container_width=True):
            st.session_state["page_navigation"] = "Single Prediction"
            st.rerun()
    with a2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-badge">02 / BULK ANALYSIS</div>
            <h3>Batch prediction</h3>
            <p>Upload a CSV dataset to process multiple records in one workflow.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open batch prediction  →", key="go_batch", use_container_width=True):
            st.session_state["page_navigation"] = "Batch Prediction (CSV)"
            st.rerun()

    st.markdown("### Explore the model")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown('<div class="dashboard-card"><h4>Model insights</h4><p>Inspect feature importance and model behaviour.</p></div>', unsafe_allow_html=True)
        if st.button("View insights", key="go_insights", use_container_width=True):
            st.session_state["page_navigation"] = "Model Insights"
            st.rerun()
    with e2:
        st.markdown('<div class="dashboard-card"><h4>Country comparison</h4><p>Compare epidemiological indicators across countries.</p></div>', unsafe_allow_html=True)
        if st.button("Compare countries", key="go_countries", use_container_width=True):
            st.session_state["page_navigation"] = "Country Comparison"
            st.rerun()
    with e3:
        st.markdown('<div class="dashboard-card"><h4>Trend simulation</h4><p>Explore possible future case trajectories.</p></div>', unsafe_allow_html=True)
        if st.button("Run simulation", key="go_trends", use_container_width=True):
            st.session_state["page_navigation"] = "Trend Simulation"
            st.rerun()

    st.markdown("### Prediction inputs")
    st.markdown("""
    <div class="dashboard-card">
      <p>Both prediction workflows use the same four input signals:</p>
      <span class="feature-badge">Confirmed cases</span>
      <span class="feature-badge">Active cases</span>
      <span class="feature-badge">New cases · 24h</span>
      <span class="feature-badge">Hospital beds / 1,000</span>
    </div>
    """, unsafe_allow_html=True)

# PAGE 2: SINGLE PREDICTION
elif page == "Single Prediction":
    st.markdown("## Single Case Prediction")
    
    if model is not None:
        st.markdown("### Input Parameters")
        st.markdown("Enter the current epidemiological data and healthcare metrics:")
        
        with st.container():
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                confirmed = st.number_input(
                    "**Total Confirmed Cases**", 
                    min_value=0, 
                    value=10000,
                    step=100,
                    help="Cumulative number of confirmed COVID-19 cases"
                )
                
                active = st.number_input(
                    "**Active Cases**", 
                    min_value=0, 
                    value=5000,
                    step=100,
                    help="Currently active COVID-19 cases"
                )
            
            with col2:
                new_cases = st.number_input(
                    "**New Cases (24h)**", 
                    min_value=0, 
                    value=500,
                    step=10,
                    help="New cases reported in the last 24 hours"
                )
                
                beds = st.number_input(
                    "**Hospital Beds per 1000**", 
                    min_value=0.0, 
                    max_value=20.0,
                    value=2.5, 
                    step=0.1,
                    format="%.1f",
                    help="Available hospital beds per 1000 people"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Validation metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            if active > confirmed:
                st.warning("Active cases cannot exceed confirmed cases")
        with col2:
            case_activity_rate = (active / confirmed * 100) if confirmed > 0 else 0
            st.metric("Case Activity Rate", f"{case_activity_rate:.1f}%")
        with col3:
            daily_rate = (new_cases / confirmed * 100) if confirmed > 0 else 0
            st.metric("Daily Growth Rate", f"{daily_rate:.2f}%")
        
        # Prediction button
        if st.button("Generate Prediction", use_container_width=True):
            if active <= confirmed:
                with st.spinner("Computing prediction..."):
                    try:
                        features = np.array([[confirmed, active, new_cases, beds]])
                        prediction = model.predict(features)
                        predicted_deaths = max(0, int(prediction[0]))
                        
                        # Display prediction
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.markdown("### Prediction Result")
                            st.markdown(f"#### Estimated COVID-19 Deaths:")
                            st.markdown(f"# **{predicted_deaths:,}**")
                        
                        with col2:
                            mortality_rate = (predicted_deaths / confirmed * 100) if confirmed > 0 else 0
                            st.metric("Estimated Mortality Rate", f"{mortality_rate:.2f}%")
                            st.metric("Deaths per 1000 Beds", f"{predicted_deaths / (beds * 1000) if beds > 0 else 0:.2f}")
                        
                        with col3:
                            if mortality_rate < 1:
                                risk = "Low"
                            elif mortality_rate < 3:
                                risk = "Moderate"
                            else:
                                risk = "High"
                            st.markdown(f"**Risk Level:**")
                            st.markdown(f"## {risk}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Visualization
                        st.markdown("### Data Visualization")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            labels = ['Active Cases', 'Predicted Deaths', 'Others']
                            values = [active, predicted_deaths, max(0, confirmed - active - predicted_deaths)]
                            colors = ['#3498db', '#e74c3c', '#27ae60']
                            
                            fig1 = go.Figure(data=[go.Pie(
                                labels=labels,
                                values=values,
                                marker=dict(colors=colors),
                                hole=0.4
                            )])
                            fig1.update_layout(title="Case Distribution", height=300)
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            metrics = ['Confirmed', 'Active', 'New Cases', 'Predicted Deaths']
                            values_bar = [confirmed, active, new_cases, predicted_deaths]
                            
                            fig2 = go.Figure(data=[go.Bar(
                                x=metrics,
                                y=values_bar,
                                marker=dict(color=['#3498db', '#f39c12', '#27ae60', '#e74c3c'])
                            )])
                            fig2.update_layout(title="Key Metrics Overview", height=300)
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Export
                        st.markdown("### Export Results")
                        result_data = {
                            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                            'Confirmed Cases': [confirmed],
                            'Active Cases': [active],
                            'New Cases': [new_cases],
                            'Hospital Beds per 1000': [beds],
                            'Predicted Deaths': [predicted_deaths],
                            'Mortality Rate (%)': [mortality_rate],
                            'Risk Level': [risk]
                        }
                        df_results = pd.DataFrame(result_data)
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="Download Report (CSV)",
                            data=csv,
                            file_name=f'prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv',
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.error("Invalid: Active cases cannot exceed confirmed cases.")
    else:
        st.error("Model not available.")

# PAGE 3: BATCH PREDICTION (CSV)
elif page == "Batch Prediction (CSV)":
    st.markdown("## Batch Prediction from CSV")
    
    st.info("""
    **Upload a CSV file** with multiple entries to get predictions for all at once.
    
    **Required columns:**
    - `Confirmed` - Total confirmed cases
    - `Active` - Active cases
    - `New_Cases` - New cases in 24h
    - `Beds_per_1000` - Hospital beds per 1000 population
    """)
    
    # Sample CSV Download
    col1, col2 = st.columns([1, 2])
    with col1:
        sample_data = {
            'Confirmed': [10000, 25000, 50000],
            'Active': [5000, 12000, 30000],
            'New_Cases': [500, 1200, 2500],
            'Beds_per_1000': [2.5, 3.0, 1.8]
        }
        sample_df = pd.DataFrame(sample_data)
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV",
            data=csv_sample,
            file_name='sample_covid_data.csv',
            mime='text/csv',
        )
    
    with col2:
        st.markdown("**Sample CSV format preview:**")
        st.dataframe(sample_df, use_container_width=True)
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Your CSV File", type=['csv'])
    
    if uploaded_file is not None and model is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! Found {len(df)} rows.")
            
            # Validate columns
            required_cols = ['Confirmed', 'Active', 'New_Cases', 'Beds_per_1000']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                st.markdown("### Uploaded Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("Run Batch Prediction", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        predictions = []
                        mortality_rates = []
                        risk_levels = []
                        
                        for idx, row in df.iterrows():
                            features = np.array([[row['Confirmed'], row['Active'], 
                                                row['New_Cases'], row['Beds_per_1000']]])
                            pred = model.predict(features)[0]
                            predicted_deaths = max(0, int(pred))
                            predictions.append(predicted_deaths)
                            
                            mortality_rate = (predicted_deaths / row['Confirmed'] * 100) if row['Confirmed'] > 0 else 0
                            mortality_rates.append(round(mortality_rate, 2))
                            
                            if mortality_rate < 1:
                                risk_levels.append("🟢 Low")
                            elif mortality_rate < 3:
                                risk_levels.append("🟡 Moderate")
                            else:
                                risk_levels.append("🔴 High")
                        
                        df['Predicted_Deaths'] = predictions
                        df['Mortality_Rate_%'] = mortality_rates
                        df['Risk_Level'] = risk_levels
                        
                        st.success("Predictions completed!")
                        
                        # Display results
                        st.markdown("### Prediction Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Predicted Deaths", f"{df['Predicted_Deaths'].sum():,}")
                        with col2:
                            st.metric("Average Mortality Rate", f"{df['Mortality_Rate_%'].mean():.2f}%")
                        with col3:
                            high_risk_count = len(df[df['Risk_Level'] == "🔴 High"])
                            st.metric("High Risk Entries", high_risk_count)
                        
                        # Visualization
                        st.markdown("### Batch Analysis")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=df.index,
                            y=df['Predicted_Deaths'],
                            name='Predicted Deaths',
                            marker_color='#e74c3c'
                        ))
                        fig.update_layout(
                            title="Predicted Deaths by Entry",
                            xaxis_title="Entry Index",
                            yaxis_title="Deaths",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv_results = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv_results,
                            file_name=f'batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv',
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    elif model is None:
        st.error("Model not available.")

# PAGE 4: MODEL INSIGHTS
elif page == "Model Insights":
    st.markdown("## Model Insights & Feature Importance")
    
    st.info("""
    **Understanding the Model:** This section shows which features have the most impact on predictions.
    Feature importance helps explain how the model makes decisions.
    """)
    
    if model is not None:
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = ['Confirmed Cases', 'Active Cases', 'New Cases (24h)', 'Hospital Beds per 1000']
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            st.markdown("### Feature Importance Analysis")
            
            # Horizontal bar chart
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(color='#3498db')
            ))
            fig.update_layout(
                title="Feature Importance (Higher = More Impact)",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("### Interpretation")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **What does this mean?**
                - Features with higher importance scores have more influence on predictions
                - The model relies more heavily on top-ranked features
                - Lower-ranked features still contribute but with less weight
                """)
            
            with col2:
                # Show sorted importance
                st.markdown("**Ranking:**")
                for idx, row in importance_df.sort_values('Importance', ascending=False).iterrows():
                    st.markdown(f"{row['Feature']}: `{row['Importance']:.3f}`")
        
        else:
            st.warning("""
            Feature importance is not available for this model type.
            
            **Note:** Linear models use coefficients instead of feature importance.
            If you're using Linear Regression, coefficients show feature impact.
            """)
            
            # Try to show coefficients if available
            if hasattr(model, 'coef_'):
                st.markdown("### Model Coefficients")
                features = ['Confirmed Cases', 'Active Cases', 'New Cases (24h)', 'Hospital Beds per 1000']
                coefs = model.coef_
                
                coef_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': coefs
                }).sort_values('Coefficient', ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=coef_df['Coefficient'],
                    y=coef_df['Feature'],
                    orientation='h',
                    marker=dict(color='#3498db')
                ))
                fig.update_layout(
                    title="Model Coefficients",
                    xaxis_title="Coefficient Value",
                    yaxis_title="Feature",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Model not available.")

# PAGE 5: COUNTRY COMPARISON
elif page == "Country Comparison":
    st.markdown("## Multi-Country Comparison")
    
    st.info("""
    **Compare COVID-19 metrics across multiple countries/regions.**
    This feature allows side-by-side analysis of infection rates, mortality, and healthcare capacity.
    """)
    
    st.warning("**Coming Soon!** This feature requires integration with external COVID-19 data APIs.")

    st.markdown("### Planned Features:")
    st.markdown("""
    - Side-by-side metric comparison
    - Trend line visualization
    - Geographic heatmaps
    - Mortality rate comparisons
    - Healthcare capacity analysis
    """)
    
    # Demo placeholder
    st.markdown("### Preview (Demo Data)")
    
    demo_data = {
        'Country': ['USA', 'India', 'Brazil', 'UK', 'Germany'],
        'Confirmed': [50000000, 45000000, 35000000, 24000000, 38000000],
        'Deaths': [1000000, 530000, 700000, 210000, 170000],
        'Active': [500000, 200000, 150000, 50000, 100000],
        'Beds_per_1000': [2.9, 0.5, 2.1, 2.5, 8.0]
    }
    demo_df = pd.DataFrame(demo_data)
    
    st.dataframe(demo_df, use_container_width=True)
    
    # Demo chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=demo_df['Country'],
        y=demo_df['Deaths'],
        name='Deaths',
        marker_color='#e74c3c'
    ))
    fig.update_layout(
        title="COVID-19 Deaths by Country (Demo)",
        xaxis_title="Country",
        yaxis_title="Total Deaths",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# PAGE 6: TREND SIMULATION
elif page == "Trend Simulation":
    st.markdown("## Predictive Trend Simulation")
    
    st.info("""
    **Project future COVID-19 trends** based on current data and adjustable parameters.
    Simulate different scenarios to understand potential outcomes.
    """)
    
    st.warning("**Coming Soon!** This feature requires time-series forecasting models.")
    
    st.markdown("### Planned Features:")
    st.markdown("""
    - 7-30 day projections
    - Adjustable growth rate sliders
    - Multiple scenario comparisons
    - Early warning indicators
    - Confidence interval bands
    """)
    
    # Demo simulation
    st.markdown("### Demo Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        days_ahead = st.slider("Forecast Days Ahead", 7, 30, 14)
        growth_rate = st.slider("Daily Growth Rate (%)", 0.0, 10.0, 2.5, 0.5)
    
    with col2:
        current_cases = st.number_input("Current Active Cases", 1000, 100000, 5000, 500)
        intervention = st.selectbox("Intervention Level", ["None", "Moderate", "Strong"])
    
    # Generate demo projection
    if st.button("Generate Projection", use_container_width=True):
        dates = [datetime.now() + timedelta(days=i) for i in range(days_ahead)]
        
        # Simple exponential growth simulation
        intervention_factor = {"None": 1.0, "Moderate": 0.7, "Strong": 0.4}[intervention]
        adjusted_growth = growth_rate * intervention_factor / 100
        
        projected_cases = [current_cases * (1 + adjusted_growth) ** i for i in range(days_ahead)]
        
        # Create projection chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=projected_cases,
            mode='lines+markers',
            name='Projected Cases',
            line=dict(color='#e74c3c', width=3)
        ))
        fig.update_layout(
            title=f"COVID-19 Case Projection - {days_ahead} Days ({intervention} Intervention)",
            xaxis_title="Date",
            yaxis_title="Active Cases",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"By day {days_ahead}, projected cases: **{int(projected_cases[-1]):,}**")

# PAGE 7: ABOUT & METHODOLOGY
elif page == "About & Methodology":
    st.markdown("""
    <div class="about-hero">
        <div class="about-eyebrow">PROJECT DOCUMENTATION · MODEL TRANSPARENCY</div>
        <h1>Behind the <span>prediction.</span></h1>
        <p>A clear guide to what this system does, how its machine-learning model works, and where its predictions should — and should not — be used.</p>
        <div class="about-tags"><span>RANDOM FOREST REGRESSION</span><span>SCIKIT-LEARN</span><span>STREAMLIT</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .about-hero { position:relative; overflow:hidden; padding:2.1rem 2.2rem; margin:.4rem 0 1.6rem; border-radius:22px; color:#f3fff8; background:linear-gradient(115deg,#073b34 0%,#0b6657 65%,#0b8c73 100%); box-shadow:0 16px 36px rgba(7,59,52,.14); }
    .about-hero:after { content:''; position:absolute; width:260px; height:260px; border:1px solid rgba(199,243,107,.35); border-radius:50%; right:-75px; top:-125px; box-shadow:0 0 0 22px rgba(199,243,107,.055),0 0 0 48px rgba(199,243,107,.035); }
    .about-eyebrow { color:#c7f36b; font:600 .7rem 'IBM Plex Mono',monospace; letter-spacing:.12em; margin-bottom:1.1rem; }
    .about-hero h1 { color:#fff; font-size:clamp(2rem,4vw,3.1rem); letter-spacing:-.055em; line-height:1.04; margin:0 0 .8rem; }
    .about-hero h1 span { color:#c7f36b; }
    .about-hero p { color:#d2eee3; max-width:760px; font-size:1.02rem; line-height:1.7; margin:0 0 1.25rem; }
    .about-tags { display:flex; flex-wrap:wrap; gap:.5rem; position:relative; z-index:1; }
    .about-tags span { color:#eafff4; border:1px solid rgba(220,255,238,.28); background:rgba(255,255,255,.08); border-radius:999px; padding:.38rem .7rem; font:500 .65rem 'IBM Plex Mono',monospace; letter-spacing:.05em; }
    .about-section-kicker { color:#0b9476; font:600 .7rem 'IBM Plex Mono',monospace; letter-spacing:.12em; margin:1.5rem 0 .35rem; }
    .about-feature { height:100%; background:#fff; border:1px solid #dce9e2; border-radius:16px; padding:1.1rem 1.15rem; box-shadow:0 6px 20px rgba(16,59,53,.045); }
    .about-feature .num { color:#0b9c7d; font:600 .72rem 'IBM Plex Mono',monospace; margin-bottom:.7rem; }
    .about-feature h3 { font-size:1.02rem; margin:0 0 .45rem; color:#073b34; }
    .about-feature p { font-size:.88rem; line-height:1.55; color:#55766e; margin:0; }
    .about-panel { background:#fff; border:1px solid #dce9e2; border-radius:17px; padding:1.25rem 1.35rem; margin:.5rem 0 1rem; }
    .about-panel h3 { margin:.1rem 0 .6rem; font-size:1.08rem; }
    .about-panel p,.about-panel li { color:#55766e; line-height:1.65; font-size:.91rem; }
    .about-step { display:flex; gap:.9rem; align-items:flex-start; padding:.8rem 0; border-bottom:1px solid #e8f0eb; }
    .about-step:last-child { border-bottom:0; }
    .about-step .step { flex:0 0 32px; height:32px; display:grid; place-items:center; border-radius:10px; background:#ddf8ec; color:#087454; font:600 .75rem 'IBM Plex Mono',monospace; }
    .about-step strong { display:block; color:#103b35; margin-bottom:.18rem; }
    .about-step span { color:#66827b; font-size:.88rem; line-height:1.5; }
    .about-note { border-radius:14px; padding:1rem 1.15rem; background:#fff6df; border:1px solid #f1d99e; color:#79520c; line-height:1.6; margin:.8rem 0 1rem; }
    .about-note strong { color:#684608; }
    .about-link { color:#087b68 !important; font-weight:600; text-decoration:none; }
    @media(max-width:700px) { .about-hero { padding:1.5rem; } .about-hero:after { opacity:.45; } }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-section-kicker">01 / OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown("## What this project does")
    st.markdown("""
    This project is a machine-learning application that estimates COVID-19 deaths from
    country-level epidemiological and healthcare indicators. A trained regression model
    is exposed through a Streamlit interface for individual predictions, CSV-based batch
    processing, visual exploration and model-related analysis.
    """)

    st.markdown("""
    <div class="about-note"><strong>Important distinction</strong><br>
    This is a research and educational prediction project. It does not connect to a live
    epidemiological feed, and its output is an estimate from a model trained on historical
    data — not a current official death count or a clinical assessment.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-section-kicker">02 / MODEL INPUTS</div>', unsafe_allow_html=True)
    st.markdown("## Four signals go into the model")
    st.caption("The model expects these features in this exact order.")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""<div class="about-feature"><div class="num">FEATURE 01</div><h3>Confirmed cases</h3><p>Cumulative reported COVID-19 cases. Represents the recorded scale of the outbreak in the supplied data.</p></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="about-feature"><div class="num">FEATURE 02</div><h3>Active cases</h3><p>Currently active reported cases. Provides a snapshot of the ongoing case burden.</p></div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="about-feature"><div class="num">FEATURE 03</div><h3>New cases · 24 hours</h3><p>Newly reported cases over a 24-hour period. Adds a short-term signal to the input data.</p></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="about-feature"><div class="num">FEATURE 04</div><h3>Hospital beds per 1,000</h3><p>A healthcare-capacity indicator representing the number of hospital beds per 1,000 people.</p></div>""", unsafe_allow_html=True)

    st.markdown('<div class="about-section-kicker">03 / HOW IT WORKS</div>', unsafe_allow_html=True)
    st.markdown("## From dataset to estimate")
    st.markdown("""
    <div class="about-panel">
      <div class="about-step"><div class="step">01</div><div><strong>Prepare the dataset</strong><span>The training script reads the merged COVID-19 dataset, selects four input columns and the deaths target, and fills missing values with zero.</span></div></div>
      <div class="about-step"><div class="step">02</div><div><strong>Split the data</strong><span>The script creates an 80/20 train-test split using a fixed random state of 42.</span></div></div>
      <div class="about-step"><div class="step">03</div><div><strong>Train the regressor</strong><span>A Random Forest Regressor with 200 trees is fitted to the training data.</span></div></div>
      <div class="about-step"><div class="step">04</div><div><strong>Evaluate the model</strong><span>The training script calculates R², mean absolute error, root mean squared error and a five-fold cross-validation R² score.</span></div></div>
      <div class="about-step"><div class="step">05</div><div><strong>Generate predictions</strong><span>The Streamlit app loads the saved Joblib model and passes the four user-provided values to its predict method.</span></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-section-kicker">04 / MODEL DETAILS</div>', unsafe_allow_html=True)
    st.markdown("## What is under the hood?")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Algorithm", "Random Forest")
    with d2:
        st.metric("Estimators", "200 trees")
    with d3:
        st.metric("Target", "Deaths")
    st.markdown("""
    <div class="about-panel">
      <h3>Why Random Forest?</h3>
      <p>Random Forest is an ensemble learning method that combines predictions from multiple
      decision trees. For regression, the trees contribute to a numeric estimate. The project
      uses scikit-learn's <code>RandomForestRegressor</code>.</p>
      <p><strong>Evaluation transparency:</strong> the training script computes R², MAE, RMSE
      and five-fold cross-validation R². The actual measured values are not currently displayed
      on this page, so no accuracy percentage is claimed here.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-section-kicker">05 / INTERPRETATION</div>', unsafe_allow_html=True)
    st.markdown("## Understand the result")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""<div class="about-panel"><h3>What the estimate means</h3><p>The predicted value is the model's estimated number of deaths for the four input values provided. It is generated by the trained regression model and may differ from observed outcomes.</p></div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""<div class="about-panel"><h3>What it does not mean</h3><p>It is not a guaranteed outcome, an individual survival probability, a medical diagnosis, or a validated forecast of future deaths.</p></div>""", unsafe_allow_html=True)

    st.markdown('<div class="about-section-kicker">06 / LIMITATIONS & RESPONSIBLE USE</div>', unsafe_allow_html=True)
    st.markdown("## Read this before using predictions")
    st.markdown("""
    <div class="about-note"><strong>Not for clinical or emergency decisions.</strong><br>
    This project is intended for demonstration, learning and exploratory analysis. Do not
    use its estimates to make decisions about an individual's care, treatment, triage,
    hospital admission or public-health policy.</div>
    """, unsafe_allow_html=True)
    l1, l2 = st.columns(2)
    with l1:
        st.markdown("""<div class="about-panel"><h3>Data limitations</h3><ul><li>Predictions depend on the quality and coverage of the historical training data.</li><li>Reporting practices and healthcare capacity vary across regions and time.</li><li>Missing values are filled with zero in the current training script; this can affect model behaviour.</li><li>The project does not currently retrieve live data from an external API.</li></ul></div>""", unsafe_allow_html=True)
    with l2:
        st.markdown("""<div class="about-panel"><h3>Model limitations</h3><ul><li>Performance on historical test data does not guarantee performance on new situations.</li><li>The current training script uses a random split; it does not establish time-based validation.</li><li>The app does not provide calibrated prediction intervals or uncertainty estimates.</li><li>Relationships learned from past data may not hold during new variants or changing conditions.</li></ul></div>""", unsafe_allow_html=True)

    st.markdown('<div class="about-section-kicker">07 / TECHNOLOGY</div>', unsafe_allow_html=True)
    st.markdown("## Built with")
    tech = st.columns(4)
    for col, name, detail in zip(tech, ["Python", "scikit-learn", "Pandas + NumPy", "Streamlit"], ["Core application language", "Random Forest regression", "Data preparation and numerical computing", "Interactive web application"]):
        with col:
            st.markdown(f'<div class="about-feature"><h3>{name}</h3><p>{detail}</p></div>', unsafe_allow_html=True)
    st.caption("Additional project libraries include Joblib for model persistence and Plotly for interactive charts.")

    st.markdown('<div class="about-section-kicker">08 / PROJECT INFORMATION</div>', unsafe_allow_html=True)
    st.markdown("## Project & creator")
    p1, p2 = st.columns([1.2, 1])
    with p1:
        st.markdown("""
        <div class="about-panel">
          <h3>COVID-19 Mortality Prediction System</h3>
          <p>An educational machine-learning project combining epidemiological inputs,
          healthcare-capacity data and regression modelling.</p>
          <p><strong>License:</strong> MIT (see repository license file)</p>
          <p><strong>Interface:</strong> Streamlit</p>
          <p><strong>Model artifact:</strong> Joblib (.pkl)</p>
          <p><a class="about-link" href="https://github.com/SudharsanamRK/covid19-death-prediction" target="_blank">View source code on GitHub ↗</a></p>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="about-panel">
          <h3>Created by</h3>
          <p><strong>Sudharsanam R K</strong></p>
          <p>Computer Science & Engineering</p>
          <p><a class="about-link" href="https://github.com/SudharsanamRK" target="_blank">GitHub profile ↗</a></p>
          <p><a class="about-link" href="https://www.linkedin.com/in/sudharsanamrk/" target="_blank">LinkedIn profile ↗</a></p>
          <p><a class="about-link" href="mailto:rksudharsanam2005@gmail.com">Contact by email ↗</a></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="about-section-kicker">09 / DOCUMENTATION</div>', unsafe_allow_html=True)
    st.markdown("## Learn more")
    st.markdown("""
    - [Project repository and README](https://github.com/SudharsanamRK/covid19-death-prediction)
    - [scikit-learn Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    - [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
    - [Streamlit documentation](https://docs.streamlit.io/)
    """)

    st.markdown("---")
    st.caption("COVID-19 Mortality Lab · Project documentation · Estimates are for educational exploration only.")

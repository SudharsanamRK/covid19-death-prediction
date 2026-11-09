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

# Visual styling only; prediction logic remains below.
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
/* Sidebar sizing: collapse fully without leaving an empty rail, while keeping the reopen control on screen. */
[data-testid="stSidebar"] { flex-shrink:0 !important; transition:width .24s ease, min-width .24s ease, flex-basis .24s ease; }
[data-testid="stSidebar"][aria-expanded="true"] { width:280px !important; min-width:280px !important; max-width:280px !important; flex-basis:280px !important; }
[data-testid="stSidebar"][aria-expanded="false"] { width:0 !important; min-width:0 !important; max-width:0 !important; flex-basis:0 !important; margin:0 !important; padding:0 !important; overflow:visible !important; }
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] { visibility:hidden !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:1.1rem; }
[data-testid="stSidebarCollapsedControl"] { position:fixed !important; left:0 !important; top:1rem !important; display:flex !important; visibility:visible !important; opacity:1 !important; z-index:1000000 !important; }
[data-testid="stSidebarCollapsedControl"] button { display:flex !important; visibility:visible !important; opacity:1 !important; min-width:42px !important; min-height:44px !important; background:#073b34 !important; color:#fff !important; border:1px solid rgba(255,255,255,.25) !important; border-radius:0 12px 12px 0 !important; box-shadow:0 4px 16px rgba(7,59,52,.22) !important; }
[data-testid="stSidebarCollapsedControl"] button svg { fill:#fff !important; color:#fff !important; }
@media (max-width:900px) {
  [data-testid="stSidebar"][aria-expanded="true"] { width:min(86vw,300px) !important; min-width:min(86vw,300px) !important; max-width:min(86vw,300px) !important; flex-basis:min(86vw,300px) !important; }
  [data-testid="stSidebar"][aria-expanded="false"] { width:0 !important; min-width:0 !important; max-width:0 !important; flex-basis:0 !important; }
}
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

# Load the repository's static country-wise snapshot for the comparison page
@st.cache_data
def load_country_dataset():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "country_wise_latest.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path), data_path
    return None, None

country_data, country_data_path = load_country_dataset()

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
    st.markdown("""
    <style>
    .dash-hero {position:relative;overflow:hidden;padding:2.35rem 2.5rem;margin:.25rem 0 1.5rem;border-radius:25px;background:linear-gradient(115deg,#073b34 0%,#0b6657 62%,#0b9478 100%);box-shadow:0 18px 42px rgba(7,59,52,.16);animation:dashEnter .65s cubic-bezier(.2,.75,.25,1) both}
    .dash-hero:before {content:'';position:absolute;width:330px;height:330px;right:-95px;top:-190px;border:1px solid rgba(199,243,107,.45);border-radius:50%;box-shadow:0 0 0 25px rgba(199,243,107,.07),0 0 0 55px rgba(199,243,107,.045),0 0 0 90px rgba(199,243,107,.025);animation:orbit 18s linear infinite}
    .dash-hero:after {content:'';position:absolute;width:150px;height:150px;right:17%;bottom:-115px;border:1px solid rgba(255,255,255,.16);border-radius:50%;box-shadow:0 0 0 18px rgba(255,255,255,.035)}
    .dash-kicker {display:inline-flex;align-items:center;gap:.55rem;color:#c7f36b;font:600 .7rem 'IBM Plex Mono',monospace;letter-spacing:.13em;margin-bottom:1.1rem}
    .dash-kicker:before {content:'';width:8px;height:8px;border-radius:50%;background:#c7f36b;box-shadow:0 0 0 5px rgba(199,243,107,.13);animation:pulseDot 2s ease-in-out infinite}
    .dash-hero h1 {position:relative;z-index:1;color:#fff;font-size:clamp(2.25rem,5vw,3.8rem);line-height:1.02;letter-spacing:-.06em;margin:0 0 .85rem;max-width:800px}
    .dash-hero h1 span {color:#c7f36b}
    .dash-hero p {position:relative;z-index:1;color:#d2eee3;max-width:730px;font-size:1.05rem;line-height:1.7;margin:0 0 1.4rem}
    .dash-hero-foot {position:relative;z-index:1;display:flex;gap:.55rem;flex-wrap:wrap}
    .dash-hero-foot span {color:#effff6;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.2);padding:.42rem .75rem;border-radius:999px;font:500 .66rem 'IBM Plex Mono',monospace}
    .dash-section {margin:1.6rem 0 .8rem;color:#103b35;font-size:1.45rem;font-weight:800;letter-spacing:-.04em}
    .dash-sub {color:#66827b;margin:-.35rem 0 1rem}
    .dash-step {height:100%;min-height:170px;background:#fff;border:1px solid #dce9e2;border-radius:18px;padding:1.15rem 1.2rem;box-shadow:0 8px 24px rgba(16,59,53,.045);transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease;animation:dashEnter .6s ease both}
    .dash-step:hover {transform:translateY(-6px);box-shadow:0 16px 30px rgba(16,59,53,.11);border-color:#9bd9bd}
    .dash-step .num {color:#0b9476;font:600 .7rem 'IBM Plex Mono',monospace;letter-spacing:.12em}
    .dash-step h3 {color:#103b35;font-size:1.12rem;margin:.65rem 0 .4rem}
    .dash-step p {color:#66827b;font-size:.9rem;line-height:1.55;margin:0}
    .dash-action {height:100%;min-height:185px;background:linear-gradient(145deg,#fff,#f3faf5);border:1px solid #dce9e2;border-radius:19px;padding:1.25rem;box-shadow:0 8px 24px rgba(16,59,53,.045);transition:transform .25s ease,box-shadow .25s ease}
    .dash-action:hover {transform:translateY(-5px);box-shadow:0 16px 32px rgba(16,59,53,.1)}
    .dash-action .tag {color:#087a65;font:600 .68rem 'IBM Plex Mono',monospace;letter-spacing:.1em}
    .dash-action h3 {color:#103b35;font-size:1.3rem;margin:.65rem 0 .45rem}
    .dash-action p {color:#66827b;line-height:1.55;font-size:.9rem;min-height:44px}
    .dash-explain {background:#fff;border:1px solid #dce9e2;border-radius:18px;padding:1.25rem 1.4rem;margin:.5rem 0 1rem}
    .dash-explain h3 {color:#103b35;margin:.1rem 0 .55rem}
    .dash-explain p,.dash-explain li {color:#55766e;line-height:1.65}
    .dash-flow {display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;margin:.8rem 0}
    .dash-flow span {background:#e7f8ee;border:1px solid #c7ead7;color:#087454;padding:.55rem .75rem;border-radius:10px;font-size:.82rem;font-weight:700}
    .dash-flow b {color:#0b9476;font-size:1rem}
    @keyframes dashEnter {from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
    @keyframes orbit {to{transform:rotate(360deg)}}
    @keyframes pulseDot {50%{box-shadow:0 0 0 8px rgba(199,243,107,.05);opacity:.7}}
    @media(max-width:900px){.dash-hero{padding:1.6rem 1.35rem}.dash-hero h1{font-size:2.45rem}.dash-step{min-height:145px}}
    @media(prefers-reduced-motion:reduce){.dash-hero,.dash-step,.dash-action,.dash-kicker:before,.dash-hero:before{animation:none!important;transition:none!important}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <section class="dash-hero">
      <div class="dash-kicker">PUBLIC HEALTH DATA WORKSPACE</div>
      <h1>See the signals.<br><span>Understand the estimate.</span></h1>
      <p>A practical workspace for exploring COVID-19 indicators, estimating mortality outcomes with a trained regression model, and comparing historical country data.</p>
      <div class="dash-hero-foot"><span>MODEL STATUS · """ + ("READY" if model is not None else "UNAVAILABLE") + """</span><span>4 INPUT VARIABLES</span><span>2 PREDICTION WORKFLOWS</span></div>
    </section>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dash-section">Workspace overview</div><p class="dash-sub">Choose a task to get started. Each tool is designed for a specific part of the analysis.</p>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Prediction methods","02","Single and batch")
    m2.metric("Input variables","04","Shared model inputs")
    m3.metric("Analysis tools","03","Insights, countries, trends")
    m4.metric("Model status","Ready" if model is not None else "Unavailable")
    
    st.markdown('<div class="dash-section">Quick actions</div><p class="dash-sub">Open a workflow directly without searching through menus.</p>',unsafe_allow_html=True)
    a1,a2=st.columns(2,gap="large")
    with a1:
        st.markdown('<div class="dash-action"><div class="tag">01 / INDIVIDUAL RECORD</div><h3>Single prediction</h3><p>Enter four indicators, run the model and review the estimated number of deaths.</p></div>',unsafe_allow_html=True)
        if st.button("Start single prediction  →",key="go_single",use_container_width=True):
            st.session_state["page_navigation"]="Single Prediction"
            st.rerun()
    with a2:
        st.markdown('<div class="dash-action"><div class="tag">02 / MULTIPLE RECORDS</div><h3>Batch prediction</h3><p>Upload a CSV, check data quality, process valid rows and export the results.</p></div>',unsafe_allow_html=True)
        if st.button("Open batch workspace  →",key="go_batch",use_container_width=True):
            st.session_state["page_navigation"]="Batch Prediction (CSV)"
            st.rerun()

    st.markdown('<div class="dash-section">Explore the data</div>',unsafe_allow_html=True)
    e1,e2,e3=st.columns(3,gap="medium")
    with e1:
        st.markdown('<div class="dash-step"><div class="num">TOOL 01</div><h3>Model insights</h3><p>See how the fitted forest model uses its input variables and understand its evaluation limits.</p></div>',unsafe_allow_html=True)
        if st.button("Explore model  →",key="go_insights",use_container_width=True):
            st.session_state["page_navigation"]="Model Insights"
            st.rerun()
    with e2:
        st.markdown('<div class="dash-step"><div class="num">TOOL 02</div><h3>Country comparison</h3><p>Compare reported cases, deaths, recoveries and regional patterns in the saved dataset.</p></div>',unsafe_allow_html=True)
        if st.button("Compare countries  →",key="go_countries",use_container_width=True):
            st.session_state["page_navigation"]="Country Comparison"
            st.rerun()
    with e3:
        st.markdown('<div class="dash-step"><div class="num">TOOL 03</div><h3>Trend simulation</h3><p>Change the assumed daily growth rate and compare three illustrative scenarios.</p></div>',unsafe_allow_html=True)
        if st.button("Open simulation  →",key="go_trends",use_container_width=True):
            st.session_state["page_navigation"]="Trend Simulation"
            st.rerun()

    st.markdown('<div class="dash-section">How this application works</div><p class="dash-sub">A simple walkthrough from data entry to interpretation.</p>',unsafe_allow_html=True)
    st.markdown("""
    <div class="dash-explain">
      <h3>The prediction process</h3>
      <p>The application uses a previously trained Random Forest regression model. You provide four numeric indicators. The model uses patterns learned during training to return an estimated death count.</p>
      <div class="dash-flow"><span>01 · Enter indicators</span><b>→</b><span>02 · Validate inputs</span><b>→</b><span>03 · Run model</span><b>→</b><span>04 · Review estimate</span></div>
      <p>Single Prediction handles one record. Batch Prediction applies the same model to each valid row in an uploaded CSV.</p>
    </div>
    """,unsafe_allow_html=True)
    st.markdown('<div class="dash-section">What do the four inputs mean?</div>',unsafe_allow_html=True)
    v1,v2=st.columns(2,gap="large")
    with v1:
        st.markdown('<div class="dash-explain"><h3>Confirmed cases</h3><p>Total reported confirmed COVID-19 cases in the observation.</p><h3>Active cases</h3><p>Reported cases currently classified as active in the supplied data.</p></div>',unsafe_allow_html=True)
    with v2:
        st.markdown('<div class="dash-explain"><h3>New cases in 24 hours</h3><p>Recently reported cases, used as a short-term indicator.</p><h3>Hospital beds per 1,000</h3><p>A healthcare capacity indicator representing available beds per 1,000 people.</p></div>',unsafe_allow_html=True)
    st.markdown("""
    <div class="dash-explain">
      <h3>How to interpret results responsibly</h3>
      <p>The output is a model estimate, not a medical diagnosis, an individual risk score or a guaranteed future outcome. Results depend on the training data, the quality of the values entered and the limits of the fitted model. Country data and simulations are separate analysis tools and should not be treated as live surveillance.</p>
    </div>
    """,unsafe_allow_html=True)

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
    st.markdown("""
    <div class="batch-hero">
      <div class="batch-kicker">BULK WORKSPACE · CSV PROCESSING</div>
      <h1>One file. <span>Many predictions.</span></h1>
      <p>Validate your dataset, run the mortality model across every valid row, inspect the output and export a processed report.</p>
      <div class="batch-tags"><span>CSV UPLOAD</span><span>ROW VALIDATION</span><span>BULK INFERENCE</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .batch-hero,.insight-hero,.country-hero {position:relative;overflow:hidden;padding:2rem 2.1rem;margin:.35rem 0 1.35rem;border-radius:21px;background:linear-gradient(115deg,#073b34,#0b6657 66%,#0b8c73);box-shadow:0 14px 32px rgba(7,59,52,.13)}
    .batch-hero:after,.insight-hero:after,.country-hero:after {content:'';position:absolute;width:240px;height:240px;right:-75px;top:-125px;border:1px solid rgba(199,243,107,.4);border-radius:50%;box-shadow:0 0 0 22px rgba(199,243,107,.06),0 0 0 46px rgba(199,243,107,.035)}
    .batch-kicker,.insight-kicker,.country-kicker {color:#c7f36b;font:600 .7rem 'IBM Plex Mono',monospace;letter-spacing:.12em;margin-bottom:1rem}
    .batch-hero h1,.insight-hero h1,.country-hero h1 {color:#fff;font-size:clamp(2rem,4vw,3rem);line-height:1.05;letter-spacing:-.05em;margin:0 0 .7rem}
    .batch-hero h1 span,.insight-hero h1 span,.country-hero h1 span {color:#c7f36b}
    .batch-hero p,.insight-hero p,.country-hero p {color:#d2eee3;max-width:760px;font-size:1rem;line-height:1.65;margin:0 0 1.1rem}
    .batch-tags,.insight-tags,.country-tags {display:flex;flex-wrap:wrap;gap:.45rem;position:relative;z-index:1}
    .batch-tags span,.insight-tags span,.country-tags span {color:#effff6;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.25);padding:.35rem .65rem;border-radius:999px;font:500 .64rem 'IBM Plex Mono',monospace}
    .work-panel,.insight-panel,.country-panel {background:#fff;border:1px solid #dce9e2;border-radius:17px;padding:1.15rem 1.25rem;margin:.4rem 0 1rem}
    .work-note,.insight-note,.country-note {padding:1rem 1.15rem;border:1px solid #d4e9dd;border-left:4px solid #19b887;background:#effaf3;color:#31584f;border-radius:13px;line-height:1.6;margin:.8rem 0 1rem}
    .work-kicker,.insight-kicker-small,.country-kicker-small {color:#0b9476;font:600 .68rem 'IBM Plex Mono',monospace;letter-spacing:.1em;margin-bottom:.45rem}
    .work-panel h3,.insight-panel h3,.country-panel h3 {color:#103b35;margin:.1rem 0 .55rem}
    .work-panel p,.insight-panel p,.country-panel p {color:#55766e;line-height:1.6}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### Start with a template")
    st.caption("Download the sample structure, add one observation per row, then upload your file.")
    template_df = pd.DataFrame({"Confirmed":[10000,25000,50000],"Active":[5000,12000,30000],"New cases":[500,1200,2500],"Hospital_beds_per_1000":[2.5,3.0,1.8]})
    c1,c2 = st.columns([1,1.5],gap="large")
    with c1:
        st.download_button("Download CSV template  ↓",template_df.to_csv(index=False).encode("utf-8"),"mortality_lab_template.csv","text/csv",use_container_width=True)
        st.markdown('<div class="work-note"><strong>Required inputs</strong><br>Confirmed · Active · New cases · Hospital beds per 1,000</div>',unsafe_allow_html=True)
    with c2:
        st.dataframe(template_df,use_container_width=True,hide_index=True)

    st.markdown("### Upload your dataset")
    uploaded_file = st.file_uploader("Choose a CSV file",type=["csv"],key="batch_csv_upload")
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file)
            if raw_df.empty:
                st.error("The CSV contains no data rows.")
            elif len(raw_df)>50000:
                st.error("This workspace supports up to 50,000 rows per upload.")
            else:
                aliases = {"confirmed":"Confirmed","confirmed cases":"Confirmed","active":"Active","active cases":"Active","new cases":"New cases","new_cases":"New cases","new cases (24h)":"New cases","hospital_beds_per_1000":"Hospital_beds_per_1000","beds_per_1000":"Hospital_beds_per_1000","hospital beds per 1000":"Hospital_beds_per_1000"}
                rename_map = {c:aliases[str(c).strip().lower()] for c in raw_df.columns if str(c).strip().lower() in aliases}
                input_df = raw_df.rename(columns=rename_map).copy()
                required = ["Confirmed","Active","New cases","Hospital_beds_per_1000"]
                missing = [c for c in required if c not in input_df.columns]
                if missing:
                    st.error("Missing required columns: "+", ".join(missing))
                    st.caption("Detected columns: "+", ".join(map(str,raw_df.columns)))
                    st.info("Use the template above. Common variants such as New_Cases and Beds_per_1000 are accepted.")
                else:
                    numeric = input_df[required].apply(pd.to_numeric,errors="coerce")
                    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
                    nonnegative = (numeric>=0).all(axis=1).to_numpy()
                    active_ok = (numeric["Active"]<=numeric["Confirmed"]).to_numpy()
                    valid = finite & nonnegative & active_ok
                    st.markdown("### File quality check")
                    q1,q2,q3,q4 = st.columns(4)
                    q1.metric("Rows received",f"{len(raw_df):,}")
                    q2.metric("Valid rows",f"{int(valid.sum()):,}")
                    q3.metric("Rows to review",f"{int((~valid).sum()):,}")
                    q4.metric("Inputs mapped","4 / 4")
                    st.dataframe(raw_df.head(8),use_container_width=True,hide_index=True)
                    if (~valid).any():
                        issues = pd.DataFrame({"CSV row":np.where(~valid)[0]+2,"Issue":["Missing or non-numeric value" if not finite[i] else "Negative input" if not nonnegative[i] else "Active exceeds confirmed" for i in np.where(~valid)[0]]})
                        with st.expander("Review invalid rows"):
                            st.dataframe(issues,use_container_width=True,hide_index=True)
                    if model is None:
                        st.error("The model is unavailable; predictions cannot be generated.")
                    elif not valid.any():
                        st.warning("There are no valid rows to process. Correct the input data and upload it again.")
                    elif st.button(f"Generate predictions for {int(valid.sum()):,} valid rows",type="primary",use_container_width=True):
                        with st.spinner("Running model inference..."):
                            features = numeric.loc[valid,required].copy()
                            if hasattr(model,"feature_names_in_"):
                                features = features[list(model.feature_names_in_)]
                            predictions = np.maximum(0,model.predict(features))
                            result = raw_df.copy()
                            result["Prediction status"] = np.where(valid,"Processed","Needs review")
                            result["Predicted deaths"] = np.nan
                            result["Estimated deaths / 100 confirmed (%)"] = np.nan
                            result.loc[valid,"Predicted deaths"] = np.rint(predictions).astype(int)
                            confirmed = numeric.loc[valid,"Confirmed"].to_numpy()
                            rates = np.divide(predictions,confirmed,out=np.zeros_like(predictions,dtype=float),where=confirmed>0)*100
                            result.loc[valid,"Estimated deaths / 100 confirmed (%)"] = np.round(rates,2)
                            st.session_state["batch_prediction_result"] = result
                            st.session_state["batch_prediction_source"] = uploaded_file.name
                    result = st.session_state.get("batch_prediction_result")
                    if result is not None and len(result)==len(raw_df) and st.session_state.get("batch_prediction_source")==uploaded_file.name:
                        done = result[result["Prediction status"]=="Processed"]
                        st.markdown("### Prediction results")
                        a,b,c = st.columns(3)
                        a.metric("Processed rows",f"{len(done):,}")
                        b.metric("Estimated deaths · total",f"{int(done['Predicted deaths'].sum()):,}")
                        c.metric("Needs review",f"{len(result)-len(done):,}")
                        st.dataframe(result,use_container_width=True,hide_index=True)
                        if not done.empty:
                            plot_df=done.head(30).copy()
                            plot_df["CSV row"]=plot_df.index+2
                            fig=px.bar(plot_df,x="CSV row",y="Predicted deaths",title="Predicted deaths · first 30 processed rows",color_discrete_sequence=["#0b8f78"])
                            fig.update_layout(template="plotly_white",height=360,margin=dict(l=15,r=15,t=55,b=15),xaxis_title="CSV row",yaxis_title="Predicted deaths")
                            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
                        st.download_button("Download processed results  ↓",result.to_csv(index=False).encode("utf-8"),f"mortality_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv","text/csv",use_container_width=True)
        except Exception as exc:
            st.error(f"Unable to read or process this CSV: {exc}")
    else:
        st.markdown('<div class="work-panel"><div class="work-kicker">WORKSPACE READY</div><h3>Your batch analysis starts here</h3><p>Upload a CSV to preview its structure, validate rows, run predictions and export a result report. Your original file is never modified.</p></div>',unsafe_allow_html=True)

# PAGE 4: MODEL INSIGHTS
elif page == "Model Insights":
    st.markdown("""
    <div class="insight-hero"><div class="insight-kicker">MODEL TRANSPARENCY · RANDOM FOREST</div><h1>Understand the <span>model behind the estimate.</span></h1><p>Inspect the fitted estimator, its learned feature-importance scores and the limits of what those scores can explain.</p><div class="insight-tags"><span>MODEL PROFILE</span><span>FEATURE IMPORTANCE</span><span>INTERPRETABILITY</span></div></div>
    """,unsafe_allow_html=True)
    if model is None:
        st.error("The trained model could not be loaded.")
    else:
        names=list(getattr(model,"feature_names_in_",["Confirmed","Active","New cases","Hospital_beds_per_1000"]))
        cols=st.columns(4)
        cols[0].metric("Estimator",type(model).__name__)
        cols[1].metric("Trees",f"{getattr(model,'n_estimators',0):,}" if hasattr(model,"n_estimators") else "N/A")
        cols[2].metric("Max depth",str(getattr(model,"max_depth",None) or "Unrestricted"))
        cols[3].metric("Input features",str(len(names)))
        left,right=st.columns(2,gap="large")
        with left:
            st.markdown('<div class="insight-panel"><div class="insight-kicker-small">ESTIMATOR PROFILE</div><h3>Random Forest regression</h3><p>An ensemble of decision trees. Each tree produces a numeric estimate, and the ensemble aggregates those estimates into the model output.</p><p><strong>Target:</strong> COVID-19 deaths<br><strong>Model artifact:</strong> '+os.path.basename(model_path or "Not found")+'</p></div>',unsafe_allow_html=True)
        with right:
            st.markdown('<div class="insight-panel"><div class="insight-kicker-small">INPUT CONTRACT</div><h3>Four model features</h3>'+''.join([f'<p><strong>{i+1:02d}</strong> · {n}</p>' for i,n in enumerate(names)])+'</div>',unsafe_allow_html=True)
        if hasattr(model,"feature_importances_"):
            imp=np.asarray(model.feature_importances_,dtype=float)
            if len(imp)==len(names):
                importance=pd.DataFrame({"Feature":names,"Importance":imp}).sort_values("Importance",ascending=True)
                st.markdown("### What the model relies on")
                st.caption("Built-in impurity-based importance from the fitted forest. Scores sum to 1.0; they are not causal effects or percentage contributions to an individual prediction.")
                fig=go.Figure(go.Bar(x=importance["Importance"],y=importance["Feature"],orientation="h",marker=dict(color=importance["Importance"],colorscale=[[0,"#b8e8d2"],[1,"#087b68"]]),text=[f"{v:.3f}" for v in importance["Importance"]],textposition="outside"))
                fig.update_layout(template="plotly_white",height=390,xaxis_title="Relative importance score",yaxis_title="",margin=dict(l=15,r=55,t=25,b=20),xaxis=dict(gridcolor="#e7f0ea"))
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
                ranking=importance.sort_values("Importance",ascending=False).reset_index(drop=True)
                x,y=st.columns(2,gap="large")
                with x:
                    st.markdown("### Feature ranking")
                    st.dataframe(ranking,use_container_width=True,hide_index=True)
                with y:
                    st.markdown('<div class="insight-panel"><h3>How to interpret</h3><p>Higher scores indicate a feature contributed more to reducing split impurity across the fitted trees. Importance does not show whether a feature raises or lowers a prediction, does not establish causation, and may be affected by correlated inputs.</p></div>',unsafe_allow_html=True)
            else:
                st.warning("The model's importance vector does not match its feature-name list. The chart is withheld to avoid incorrect labels.")
        else:
            st.info("This estimator does not expose built-in feature importance.")
        st.markdown("### Evaluation status")
        st.markdown('<div class="insight-note"><strong>Measured evaluation values are not stored with this model artifact.</strong><br>The training script calculates R², mean absolute error, root mean squared error and cross-validation R², but does not save those results. This page therefore does not display guessed scores.</div>',unsafe_allow_html=True)
        with st.expander("What should a full evaluation include?"):
            st.markdown("- R²: variance explained on held-out data.\\n- MAE: average absolute prediction error.\\n- RMSE: error measure that penalizes large misses.\\n- Cross-validation R²: performance across validation folds.\\n- Temporal validation: important before claiming future forecasting performance.")
        if st.button("Open Single Prediction  →",use_container_width=True):
            st.session_state["page_navigation"]="Single Prediction"
            st.rerun()

# PAGE 5: COUNTRY COMPARISON
elif page == "Country Comparison":
    st.markdown("""
    <div class="country-hero"><div class="country-kicker">GLOBAL DATA EXPLORER · HISTORICAL SNAPSHOT</div><h1>Compare the <span>country-level picture.</span></h1><p>Explore the country-wise COVID-19 snapshot stored in this repository, compare reported indicators and inspect differences across regions.</p><div class="country-tags"><span>COUNTRY COMPARISON</span><span>WHO REGION</span><span>HISTORICAL DATA</span></div></div>
    """,unsafe_allow_html=True)
    if country_data is None:
        st.error("The country-wise dataset could not be found in the repository data folder.")
    else:
        countries=country_data.copy()
        countries.columns=[str(c).strip() for c in countries.columns]
        numeric_cols=["Confirmed","Deaths","Recovered","Active","New cases","New deaths","Confirmed last week","1 week change","Deaths / 100 Cases","Recovered / 100 Cases","1 week % increase"]
        for col in numeric_cols:
            if col in countries.columns:
                countries[col]=pd.to_numeric(countries[col],errors="coerce")
        countries=countries.dropna(subset=["Country/Region"]).drop_duplicates(subset=["Country/Region"])
        st.markdown("### Dataset at a glance")
        a,b,c,d=st.columns(4)
        a.metric("Countries / regions",f"{len(countries):,}")
        b.metric("Reported confirmed",f"{int(countries['Confirmed'].fillna(0).sum()):,}")
        c.metric("Reported deaths",f"{int(countries['Deaths'].fillna(0).sum()):,}")
        d.metric("WHO regions",str(countries["WHO Region"].nunique()) if "WHO Region" in countries else "N/A")
        st.markdown('<div class="country-note"><strong>Data context:</strong> This is the static country-wise dataset committed to the repository, not a live feed. Values represent a historical reporting snapshot. Hospital-bed availability is omitted because the merged repository data currently contains zero-filled bed values, which are not reliable measurements.</div>',unsafe_allow_html=True)
        st.markdown("### Build a comparison")
        fcol,scol=st.columns([.8,1.2],gap="large")
        with fcol:
            regions=sorted(countries["WHO Region"].dropna().unique().tolist()) if "WHO Region" in countries else []
            region=st.selectbox("Filter by WHO region",["All regions"]+regions)
            metric_map={"Confirmed cases":"Confirmed","Reported deaths":"Deaths","Active cases":"Active","New cases (24h)":"New cases","Recovered cases":"Recovered","Deaths per 100 confirmed (%)":"Deaths / 100 Cases","Weekly case increase (%)":"1 week % increase"}
            metrics={k:v for k,v in metric_map.items() if v in countries.columns}
            metric_label=st.selectbox("Metric to compare",list(metrics.keys()))
            top_n=st.slider("Countries in ranking",5,min(25,max(5,len(countries))),min(10,max(5,len(countries))))
        filtered=countries.copy()
        if region!="All regions" and "WHO Region" in filtered:
            filtered=filtered[filtered["WHO Region"]==region]
        options=sorted(filtered["Country/Region"].astype(str).tolist())
        defaults=[v for v in ["India","US","Brazil"] if v in options] or options[:min(3,len(options))]
        with scol:
            selected=st.multiselect("Choose up to 6 countries / regions",options,default=defaults,max_selections=6)
        metric=metrics[metric_label]
        if selected:
            chosen=filtered[filtered["Country/Region"].isin(selected)].copy()
            st.markdown("### Side-by-side comparison")
            chart=px.bar(chosen.sort_values(metric,ascending=False),x="Country/Region",y=metric,color="Country/Region",title=metric_label,color_discrete_sequence=px.colors.sequential.Tealgrn)
            chart.update_layout(template="plotly_white",showlegend=False,height=390,xaxis_title="",yaxis_title=metric_label,margin=dict(l=15,r=15,t=55,b=15),yaxis=dict(gridcolor="#e7f0ea"))
            st.plotly_chart(chart,use_container_width=True,config={"displayModeBar":False})
            display=[v for v in ["Country/Region","WHO Region","Confirmed","Deaths","Recovered","Active","New cases","Deaths / 100 Cases","1 week % increase"] if v in chosen.columns]
            st.dataframe(chosen[display].sort_values(metric,ascending=False),use_container_width=True,hide_index=True)
            st.download_button("Download selected comparison  ↓",chosen[display].to_csv(index=False).encode("utf-8"),"country_comparison.csv","text/csv",use_container_width=True)
        else:
            st.info("Choose at least one country or region to populate the comparison.")
        st.markdown("### Global ranking")
        ranking=filtered[["Country/Region",metric]].dropna().sort_values(metric,ascending=False).head(top_n)
        rank_fig=px.bar(ranking.sort_values(metric),x=metric,y="Country/Region",orientation="h",title=f"Top {len(ranking)} · {metric_label}",color_discrete_sequence=["#0b8f78"])
        rank_fig.update_layout(template="plotly_white",height=max(360,len(ranking)*27),margin=dict(l=15,r=20,t=55,b=15),xaxis_title=metric_label,yaxis_title="",yaxis=dict(autorange="reversed"),xaxis=dict(gridcolor="#e7f0ea"))
        st.plotly_chart(rank_fig,use_container_width=True,config={"displayModeBar":False})
        st.caption("Source file: "+os.path.basename(country_data_path or "country_wise_latest.csv")+" · Repository snapshot; not updated live.")

# PAGE 6: TREND SIMULATION
elif page == "Trend Simulation":
    st.markdown("""
    <div class="trend-hero">
      <div class="trend-kicker">SCENARIO PLANNER · WHAT-IF ANALYSIS</div>
      <h1>Explore possible <span>case trajectories.</span></h1>
      <p>Adjust the starting point and daily change rate to compare illustrative scenarios. This is a simple mathematical simulation , not a trained forecasting model.</p>
      <div class="trend-tags"><span>INTERACTIVE</span><span>SCENARIO COMPARISON</span><span>NOT A CLINICAL FORECAST</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .trend-hero { position:relative; overflow:hidden; padding:2rem 2.1rem; margin:.35rem 0 1.3rem; border-radius:21px; background:linear-gradient(115deg,#073b34,#0b6657 66%,#0b8c73); box-shadow:0 14px 32px rgba(7,59,52,.13); }
    .trend-hero:after { content:''; position:absolute; width:240px; height:240px; right:-75px; top:-125px; border:1px solid rgba(199,243,107,.4); border-radius:50%; box-shadow:0 0 0 22px rgba(199,243,107,.06),0 0 0 46px rgba(199,243,107,.035); }
    .trend-kicker { color:#c7f36b; font:600 .7rem 'IBM Plex Mono',monospace; letter-spacing:.12em; margin-bottom:1rem; }
    .trend-hero h1 { color:#fff; font-size:clamp(2rem,4vw,3rem); line-height:1.05; letter-spacing:-.05em; margin:0 0 .7rem; }
    .trend-hero h1 span { color:#c7f36b; }
    .trend-hero p { color:#d2eee3; max-width:760px; font-size:1rem; line-height:1.65; margin:0 0 1.1rem; }
    .trend-tags { display:flex; flex-wrap:wrap; gap:.45rem; position:relative; z-index:1; }
    .trend-tags span { color:#effff6; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.25); padding:.35rem .65rem; border-radius:999px; font:500 .64rem 'IBM Plex Mono',monospace; }
    .trend-panel { background:#fff; border:1px solid #dce9e2; border-radius:17px; padding:1.15rem 1.25rem; margin:.4rem 0 1rem; }
    .trend-note { padding:1rem 1.15rem; border:1px solid #f1d99e; border-left:4px solid #e5b34e; background:#fff8e8; color:#79520c; border-radius:13px; line-height:1.6; margin:.8rem 0 1rem; }
    .trend-note strong { color:#684608; }
    .trend-kpi { background:#fff; border:1px solid #dce9e2; border-radius:15px; padding:1rem 1.1rem; height:100%; }
    .trend-kpi span { color:#66827b; font:600 .68rem 'IBM Plex Mono',monospace; letter-spacing:.06em; }
    .trend-kpi strong { display:block; color:#073b34; font-size:1.65rem; letter-spacing:-.04em; margin-top:.35rem; }
    .trend-kpi small { color:#66827b; font-size:.78rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### Configure your scenario")
    st.caption("Use a starting active-case count and an assumed daily percentage change.")
    control_col, preview_col = st.columns([1.05, .95], gap="large")
    with control_col:
        with st.container(border=True):
            current_cases = st.number_input(
                "Current active cases",
                min_value=0,
                max_value=1000000000,
                value=5000,
                step=500,
                help="Starting case count for this illustrative scenario."
            )
            daily_change = st.slider(
                "Assumed daily change (%)",
                min_value=-10.0,
                max_value=15.0,
                value=2.5,
                step=0.5,
                help="Positive values increase the simulated count; negative values decrease it."
            )
            days_ahead = st.slider(
                "Simulation period",
                min_value=7,
                max_value=30,
                value=14,
                step=1,
                help="Number of days to simulate."
            )
            st.markdown("**Scenario assumptions**")
            mild_reduction = st.slider(
                "Moderate scenario: growth-rate reduction (%)",
                min_value=0,
                max_value=90,
                value=25,
                step=5
            )
            strong_reduction = st.slider(
                "Stronger scenario: growth-rate reduction (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=5
            )
            run_simulation = st.button("Run scenario comparison  →", use_container_width=True, type="primary")
    with preview_col:
        st.markdown("""
        <div class="trend-panel">
          <div class="trend-kicker">HOW TO READ THIS TOOL</div>
          <h3 style="color:#073b34;margin:.2rem 0 .6rem">Three possible paths</h3>
          <p style="color:#55766e;line-height:1.6">The baseline keeps the assumed daily change unchanged. The moderate and stronger scenarios reduce that assumed rate by the percentages you choose.</p>
          <hr style="border-color:#e5eee8">
          <p style="color:#55766e;font-size:.88rem"><strong style="color:#103b35">Baseline</strong> · No change to the assumed rate</p>
          <p style="color:#55766e;font-size:.88rem"><strong style="color:#103b35">Moderate</strong> · Your selected reduction</p>
          <p style="color:#55766e;font-size:.88rem"><strong style="color:#103b35">Stronger</strong> · Your selected reduction</p>
        </div>
        """, unsafe_allow_html=True)

    if mild_reduction > strong_reduction:
        st.warning("Your moderate scenario currently has a larger growth-rate reduction than the stronger scenario. Adjust the sliders if you want the stronger scenario to show a greater reduction.")

    st.markdown("""
    <div class="trend-note"><strong>Interpretation notice:</strong> The scenario reductions are user-defined assumptions, not measured intervention effects. The simulation uses compound daily growth and does not account for changing behaviour, demographics, reporting delays, immunity, variants, healthcare capacity or uncertainty.</div>
    """, unsafe_allow_html=True)

    if run_simulation:
        if current_cases < 0:
            st.error("Starting active cases cannot be negative.")
        else:
            scenario_specs = [
                ("Baseline", 0, "#0b8f78"),
                ("Moderate", mild_reduction, "#e0a12b"),
                ("Stronger", strong_reduction, "#398fce")
            ]
            day_numbers = np.arange(days_ahead + 1)
            start_date = datetime.now().date()
            dates = [start_date + timedelta(days=int(day)) for day in day_numbers]
            fig = go.Figure()
            final_values = {}
            for scenario_name, reduction, line_color in scenario_specs:
                scenario_rate = (daily_change / 100.0) * (1 - reduction / 100.0)
                values = [float(current_cases) * ((1 + scenario_rate) ** int(day)) for day in day_numbers]
                values = [max(0.0, value) for value in values]
                final_values[scenario_name] = values[-1]
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode="lines",
                    name=f"{scenario_name} · {reduction}% reduction",
                    line=dict(color=line_color, width=3),
                    hovertemplate="%{x|%d %b %Y}<br>Simulated active cases: %{y:,.0f}<extra></extra>"
                ))
            fig.add_vline(x=dates[0], line_dash="dot", line_color="#9ab5a8")
            fig.update_layout(
                template="plotly_white",
                title=dict(text=f"Illustrative active-case scenarios · {days_ahead} days", font=dict(size=18, color="#103b35")),
                xaxis_title="Simulation date",
                yaxis_title="Simulated active cases",
                hovermode="x unified",
                height=440,
                margin=dict(l=20, r=20, t=65, b=25),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                font=dict(color="#31584f")
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor="#e7f0ea", rangemode="tozero", tickformat=",")
            st.markdown("### Scenario results")
            st.caption("Starting value is day zero. Each line uses the same starting count.")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

            k1, k2, k3 = st.columns(3)
            for col, (scenario_name, reduction, line_color) in zip([k1, k2, k3], scenario_specs):
                with col:
                    change = final_values[scenario_name] - float(current_cases)
                    pct = (change / float(current_cases) * 100) if current_cases else 0
                    col.metric(
                        f"{scenario_name} · day {days_ahead}",
                        f"{final_values[scenario_name]:,.0f}",
                        f"{pct:+.1f}% from start"
                    )
            result_table = pd.DataFrame([
                {"Scenario": name, "Growth-rate reduction": f"{reduction}%", f"Simulated cases · day {days_ahead}": round(final_values[name])}
                for name, reduction, _ in scenario_specs
            ])
            st.markdown("### Compare outcomes")
            st.dataframe(result_table, use_container_width=True, hide_index=True)
            if current_cases == 0:
                st.info("The starting count is zero, so all three simulated paths remain at zero under this proportional-growth calculation.")
            st.download_button(
                "Download scenario data (CSV)",
                data=pd.DataFrame({
                    "Date": dates,
                    "Day": day_numbers,
                    **{
                        f"{name} ({reduction}% rate reduction)": [
                            float(current_cases) * ((1 + (daily_change / 100.0) * (1 - reduction / 100.0)) ** int(day))
                            for day in day_numbers
                        ]
                        for name, reduction, _ in scenario_specs
                    }
                }).to_csv(index=False).encode("utf-8"),
                file_name="covid_trend_scenarios.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.markdown("""
        <div class="trend-panel">
          <div class="trend-kicker">READY WHEN YOU ARE</div>
          <h3 style="color:#073b34;margin:.2rem 0 .5rem">Your scenario comparison will appear here</h3>
          <p style="color:#55766e;line-height:1.6;margin:0">Set your assumptions on the left, then select <strong>Run scenario comparison</strong> to generate the chart, end-of-period comparisons and a downloadable CSV.</p>
        </div>
        """, unsafe_allow_html=True)

# PAGE 7: ABOUT & METHODOLOGY
elif page == "About & Methodology":
    st.markdown("""
    <div class="about-hero">
        <div class="about-eyebrow">PROJECT DOCUMENTATION · MODEL TRANSPARENCY</div>
        <h1>Behind the <span>prediction.</span></h1>
        <p>A clear guide to what this system does, how its machine-learning model works, and where its predictions should , and should not , be used.</p>
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
    data , not a current official death count or a clinical assessment.</div>
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

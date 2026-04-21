

ADZUNA_APP_ID  = "ADZUNA_APP_ID"    # ← paste here
ADZUNA_APP_KEY = " ADZUNA_APP_KEY"   # ← paste here


import re, time, io
import requests
import pandas as pd
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Silent NLTK downloads ─────────────────────────────────────────
for pkg in ["punkt", "stopwords", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="AI Career Match",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLOBAL CSS  — soft, minimal, professional + background
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Lora:wght@400;500;600&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── Background — soft gradient mesh ── */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 10% 10%, rgba(199,210,254,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 70% 50% at 90% 80%, rgba(167,243,208,0.25) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 50% 50%, rgba(253,230,138,0.12) 0%, transparent 60%),
        linear-gradient(160deg, #f8f9ff 0%, #f0f4ff 40%, #f4fff8 100%);
    min-height: 100vh;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Hide chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 10px; }

/* ── Nav Bar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 40px;
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(199,210,254,0.5);
    margin-bottom: 0;
    position: sticky;
    top: 0;
    z-index: 100;
}
.nav-logo {
    font-family: 'Lora', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #312e81;
    letter-spacing: -0.02em;
}
.nav-logo span { color: #6366f1; }
.nav-links { display: flex; gap: 32px; }
.nav-link {
    font-size: 0.82rem;
    font-weight: 500;
    color: #6b7280;
    cursor: pointer;
    text-decoration: none;
    transition: color 0.2s;
    letter-spacing: 0.01em;
}
.nav-link:hover, .nav-link.active { color: #6366f1; }
.nav-pill {
    background: #6366f1;
    color: white;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 8px 20px;
    border-radius: 20px;
    cursor: pointer;
    letter-spacing: 0.02em;
}

/* ── Hero ── */
.hero-section {
    text-align: center;
    padding: 70px 20px 50px;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    color: #6366f1;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Lora', serif;
    font-size: 3.2rem;
    font-weight: 600;
    color: #1e1b4b;
    line-height: 1.2;
    margin-bottom: 16px;
    letter-spacing: -0.03em;
}
.hero-title em {
    font-style: italic;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto 32px;
    line-height: 1.7;
}

/* ── Glass Card ── */
.glass-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.07), 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.glass-card-hover {
    transition: transform 0.2s, box-shadow 0.2s;
}
.glass-card-hover:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.13);
}

/* ── Section Title ── */
.section-title {
    font-family: 'Lora', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #1e1b4b;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}
.section-sub {
    font-size: 0.82rem;
    color: #9ca3af;
    margin-bottom: 22px;
    font-weight: 400;
}

/* ── Score Ring ── */
.score-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px 0;
}
.score-ring-num {
    font-family: 'Lora', serif;
    font-size: 4rem;
    font-weight: 600;
    line-height: 1;
    letter-spacing: -0.04em;
}
.score-ring-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 6px;
    color: #9ca3af;
}

/* ── Skill Pill ── */
.skill-pill {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 5px 13px;
    border-radius: 20px;
    margin: 3px;
    letter-spacing: 0.01em;
}
.pill-have   { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
.pill-miss   { background: #fff7ed; color: #d97706; border: 1px solid #fed7aa; }
.pill-neutral{ background: #eef2ff; color: #6366f1; border: 1px solid #c7d2fe; }

/* ── Job Card ── */
.job-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(199,210,254,0.5);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 14px;
    transition: all 0.22s;
    cursor: pointer;
}
.job-card:hover {
    border-color: #a5b4fc;
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(99,102,241,0.1);
}
.job-title { font-size: 0.92rem; font-weight: 700; color: #1e1b4b; margin-bottom: 4px; }
.job-meta  { font-size: 0.77rem; color: #9ca3af; margin-bottom: 10px; }
.score-badge {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 12px;
}
.score-high { background: #ecfdf5; color: #059669; }
.score-mid  { background: #eef2ff; color: #6366f1; }
.score-low  { background: #fff7ed; color: #d97706; }

/* ── Progress Bar ── */
.prog-wrap { margin-bottom: 12px; }
.prog-row  { display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 4px; color: #6b7280; }
.prog-row span:last-child { font-weight: 600; color: #6366f1; }
.prog-bg   { background: #e0e7ff; border-radius: 6px; height: 6px; overflow: hidden; }
.prog-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #6366f1, #8b5cf6); }

/* ── Roadmap ── */
.road-item {
    display: flex;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px solid rgba(224,231,255,0.7);
    align-items: flex-start;
}
.road-item:last-child { border-bottom: none; }
.road-badge {
    min-width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.04em;
}
.rb-beg { background: #ecfdf5; color: #059669; border: 1.5px solid #a7f3d0; }
.rb-int { background: #eef2ff; color: #6366f1; border: 1.5px solid #c7d2fe; }
.rb-adv { background: #faf5ff; color: #7c3aed; border: 1.5px solid #ddd6fe; }
.road-name { font-size: 0.88rem; font-weight: 700; color: #1e1b4b; margin-bottom: 2px; }
.road-desc { font-size: 0.77rem; color: #9ca3af; line-height: 1.5; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.6) !important;
    border: 2px dashed #c7d2fe !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1 !important; }

/* ── Inputs ── */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea {
    background: rgba(255,255,255,0.8) !important;
    border: 1.5px solid #e0e7ff !important;
    border-radius: 12px !important;
    color: #1e1b4b !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.87rem !important;
}
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
.stSelectbox>div>div {
    background: rgba(255,255,255,0.8) !important;
    border: 1.5px solid #e0e7ff !important;
    border-radius: 12px !important;
    color: #1e1b4b !important;
}
.stMultiSelect>div>div {
    background: rgba(255,255,255,0.8) !important;
    border: 1.5px solid #e0e7ff !important;
    border-radius: 12px !important;
}

/* ── Button ── */
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 11px 28px !important;
    width: 100%;
    letter-spacing: 0.01em !important;
    transition: opacity 0.2s, transform 0.2s !important;
}
.stButton>button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.35) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(238,242,255,0.7) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(199,210,254,0.4) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #9ca3af !important;
    border-radius: 9px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #6366f1 !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.12) !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(199,210,254,0.5) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #9ca3af !important; }
[data-testid="stMetricValue"] { color: #6366f1 !important; font-family: 'Lora', serif !important; font-size: 1.8rem !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.7) !important;
    border: 1px solid rgba(199,210,254,0.4) !important;
    border-radius: 14px !important;
}

/* ── Slider ── */
.stSlider>div>div>div { background: linear-gradient(90deg, #6366f1, #8b5cf6) !important; }

/* ── Floating dots decoration ── */
.deco-dots {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        radial-gradient(circle, rgba(99,102,241,0.06) 1.5px, transparent 1.5px);
    background-size: 40px 40px;
}

/* ── Divider ── */
hr { border-color: rgba(199,210,254,0.4) !important; }

/* ── Info banner ── */
.info-banner {
    background: rgba(238,242,255,0.8);
    border: 1px solid rgba(199,210,254,0.6);
    border-radius: 12px;
    padding: 13px 18px;
    font-size: 0.83rem;
    color: #6366f1;
    margin-bottom: 16px;
    line-height: 1.6;
}
.warn-banner {
    background: rgba(255,247,237,0.9);
    border: 1px solid #fed7aa;
    border-radius: 12px;
    padding: 13px 18px;
    font-size: 0.83rem;
    color: #d97706;
    margin-bottom: 16px;
}

/* ── Stat strip ── */
.stat-strip {
    display: flex;
    gap: 0;
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(199,210,254,0.5);
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 28px;
}
.stat-item {
    flex: 1;
    text-align: center;
    padding: 18px 10px;
    border-right: 1px solid rgba(199,210,254,0.4);
}
.stat-item:last-child { border-right: none; }
.stat-num {
    font-family: 'Lora', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #6366f1;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-lbl { font-size: 0.72rem; color: #9ca3af; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }
</style>

<!-- Dot grid decoration -->
<div class="deco-dots"></div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
defaults = {
    "tab": "home",
    "resume_text": "",
    "matched_jobs": [],
    "roadmap": {},
    "resume_skills": [],
    "analyzed": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NAVIGATION BAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab = st.session_state.tab

def nav_btn(label, key, active_tab):
    cls = "active" if tab == active_tab else ""
    return st.button(label, key=key)

col_logo, col_n1, col_n2, col_n3, col_n4, col_cta = st.columns([2, 1, 1, 1, 1, 1.5])
with col_logo:
    st.markdown('<div class="nav-logo">✦ AI Career <span>Match</span></div>', unsafe_allow_html=True)
with col_n1:
    if st.button("Dashboard", key="nav_home", use_container_width=True):
        st.session_state.tab = "home"; st.rerun()
with col_n2:
    if st.button("Analyze", key="nav_analyze", use_container_width=True):
        st.session_state.tab = "analyze"; st.rerun()
with col_n3:
    if st.button("Jobs", key="nav_jobs", use_container_width=True):
        st.session_state.tab = "jobs"; st.rerun()
with col_n4:
    if st.button("Roadmap", key="nav_road", use_container_width=True):
        st.session_state.tab = "roadmap"; st.rerun()
with col_cta:
    if st.button("⚙ Settings", key="nav_settings", use_container_width=True):
        st.session_state.tab = "settings"; st.rerun()

st.markdown("---")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE ENGINE FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Master skill list ─────────────────────────────────────────────
# Each entry: (canonical_display_name, [all aliases to search for])
# Aliases handle hyphenated, abbreviated, and common variants.
TECH_SKILLS_MAP = [
    # Languages
    ("Python",              ["python"]),
    ("Java",                ["java"]),
    ("JavaScript",          ["javascript", "js"]),
    ("TypeScript",          ["typescript", "ts"]),
    ("C++",                 ["c\\+\\+", "cpp"]),
    ("C#",                  ["c#", "csharp", "c sharp"]),
    ("R",                   ["\\br\\b"]),
    ("Scala",               ["scala"]),
    ("Kotlin",              ["kotlin"]),
    ("Swift",               ["swift"]),
    ("Go",                  ["golang", "\\bgo\\b"]),
    ("Rust",                ["\\brust\\b"]),
    ("PHP",                 ["\\bphp\\b"]),
    ("Ruby",                ["\\bruby\\b"]),
    ("MATLAB",              ["matlab"]),
    ("Shell/Bash",          ["bash", "shell script", "shellscript"]),
    # Databases
    ("SQL",                 ["\\bsql\\b"]),
    ("MySQL",               ["mysql"]),
    ("PostgreSQL",          ["postgresql", "postgres"]),
    ("MongoDB",             ["mongodb", "mongo"]),
    ("Redis",               ["\\bredis\\b"]),
    ("Elasticsearch",       ["elasticsearch", "elastic search"]),
    ("SQLite",              ["sqlite"]),
    ("Oracle DB",           ["oracle"]),
    ("Cassandra",           ["cassandra"]),
    # ML / AI
    ("Machine Learning",    ["machine learning", "\\bml\\b"]),
    ("Deep Learning",       ["deep learning", "\\bdl\\b"]),
    ("NLP",                 ["natural language processing", "\\bnlp\\b", "text processing"]),
    ("Computer Vision",     ["computer vision", "\\bcv\\b", "image processing"]),
    ("Data Science",        ["data science", "data scientist"]),
    ("Data Analysis",       ["data analysis", "data analytics", "data analyst"]),
    ("Statistics",          ["statistics", "statistical"]),
    ("TensorFlow",          ["tensorflow", "tensor flow"]),
    ("PyTorch",             ["pytorch", "py torch"]),
    ("Keras",               ["\\bkeras\\b"]),
    ("Scikit-learn",        ["scikit-learn", "scikit learn", "sklearn"]),
    ("XGBoost",             ["xgboost", "xgb"]),
    ("LightGBM",            ["lightgbm", "lgbm"]),
    ("Hugging Face",        ["hugging face", "huggingface"]),
    ("Transformers",        ["transformers", "\\bbert\\b", "\\bgpt\\b"]),
    ("LLM",                 ["\\bllm\\b", "large language model"]),
    ("Feature Engineering", ["feature engineering"]),
    ("Model Deployment",    ["model deployment", "model serving", "model registry"]),
    # Data Tools
    ("Pandas",              ["\\bpandas\\b"]),
    ("NumPy",               ["\\bnumpy\\b"]),
    ("SciPy",               ["\\bscipy\\b"]),
    ("Matplotlib",          ["matplotlib"]),
    ("Seaborn",             ["\\bseaborn\\b"]),
    ("Plotly",              ["\\bplotly\\b"]),
    ("Tableau",             ["\\btableau\\b"]),
    ("Power BI",            ["power bi", "powerbi"]),
    ("Excel",               ["\\bexcel\\b", "ms excel", "microsoft excel"]),
    ("Jupyter",             ["jupyter", "jupyter notebook", "jupyter lab"]),
    # Big Data
    ("Apache Spark",        ["\\bspark\\b", "apache spark", "pyspark"]),
    ("Hadoop",              ["\\bhadoop\\b"]),
    ("Kafka",               ["\\bkafka\\b"]),
    ("Apache Airflow",      ["\\bairflow\\b", "apache airflow"]),
    ("Databricks",          ["databricks"]),
    ("Hive",                ["\\bhive\\b"]),
    ("Flink",               ["\\bflink\\b"]),
    # Cloud
    ("AWS",                 ["\\baws\\b", "amazon web services", "sagemaker", "amazon s3"]),
    ("Azure",               ["\\bazure\\b", "microsoft azure"]),
    ("GCP",                 ["\\bgcp\\b", "google cloud", "vertex ai", "bigquery"]),
    # DevOps / Infra
    ("Docker",              ["\\bdocker\\b"]),
    ("Kubernetes",          ["kubernetes", "\\bk8s\\b"]),
    ("Git",                 ["\\bgit\\b", "github", "gitlab", "version control"]),
    ("Linux",               ["\\blinux\\b", "unix"]),
    ("CI/CD",               ["ci/cd", "cicd", "jenkins", "github actions", "devops"]),
    ("MLflow",              ["\\bmlflow\\b"]),
    ("DVC",                 ["\\bdvc\\b", "data version control"]),
    # Web / APIs
    ("Flask",               ["\\bflask\\b"]),
    ("FastAPI",             ["fastapi", "fast api"]),
    ("Django",              ["\\bdjango\\b"]),
    ("React",               ["\\breact\\b", "reactjs", "react.js"]),
    ("Angular",             ["\\bangular\\b"]),
    ("Node.js",             ["node.js", "nodejs", "node js"]),
    ("REST API",            ["rest api", "restful", "rest ful", "api development"]),
    ("GraphQL",             ["graphql", "graph ql"]),
    ("HTML/CSS",            ["\\bhtml\\b", "\\bcss\\b"]),
    # NLP specific
    ("NLTK",                ["\\bnltk\\b"]),
    ("SpaCy",               ["\\bspacy\\b"]),
    ("BeautifulSoup",       ["beautifulsoup", "beautiful soup", "\\bbs4\\b"]),
    ("Selenium",            ["\\bselenium\\b"]),
    ("Streamlit",           ["streamlit"]),
    ("OpenCV",              ["opencv", "open cv", "cv2"]),
    # Soft / process
    ("Agile",               ["\\bagile\\b", "scrum", "kanban"]),
    ("Communication",       ["communication skills", "verbal communication"]),
    ("Teamwork",            ["teamwork", "collaboration", "collaborative"]),
    ("Problem Solving",     ["problem solving", "critical thinking"]),
    ("Project Management",  ["project management", "project planning"]),
    ("Time Management",     ["time management", "prioritization"]),
    ("java",                ["java"]),
    ("Python",              ["python"]),
    ("SQL",                 ["\\bsql\\b"]),
    ("web development",       ["web development", "web dev"]),
    ("data visualization",       ["data visualization", "data viz"]),
    ("cloud computing",       ["cloud computing", "cloud services"]),
    ("Graphic Design",       ["graphic design", "visual design"]),
    ("UI/UX Design",       ["ui design", "ux design", "user interface", "user experience"]),
    ("Mobile Development",       ["mobile development", "android", "ios"]),
    ("Cybersecurity",       ["cybersecurity", "information security", "infosec"]),
    ("DevOps",       ["devops", "ci/cd", "continuous integration", "continuous deployment"]),
]

# Flat list of canonical names for use in roadmap
TECH_SKILLS = [s[0] for s in TECH_SKILLS_MAP]

SKILL_LEVELS = {
    # Beginner
    "Python":              ("beginner","Core syntax, OOP, file handling, libraries"),
    "SQL":                 ("beginner","SELECT, JOINs, GROUP BY, subqueries, indexing"),
    "Excel":               ("beginner","Formulas, pivot tables, charts, VLOOKUP"),
    "Git":                 ("beginner","Version control, branching, pull requests"),
    "HTML/CSS":            ("beginner","Semantic structure, forms, styling, layout"),
    "Statistics":          ("beginner","Mean, variance, distributions, hypothesis testing"),
    "Pandas":              ("beginner","DataFrames, merge, groupby, data cleaning"),
    "NumPy":               ("beginner","Arrays, broadcasting, matrix operations"),
    "Shell/Bash":          ("beginner","Command line, scripts, file management"),
    # Intermediate
    "Matplotlib":          ("intermediate","Line, bar, scatter plots, subplots, styling"),
    "Seaborn":             ("intermediate","Statistical visualizations, heatmaps, pair plots"),
    "Scikit-learn":        ("intermediate","Classification, regression, pipelines, cross-validation"),
    "NLTK":                ("intermediate","Tokenization, stemming, TF-IDF, POS tagging"),
    "SpaCy":               ("intermediate","NER, dependency parsing, custom pipelines"),
    "Flask":               ("intermediate","REST APIs, routing, templates, deployment"),
    "FastAPI":             ("intermediate","Async APIs, Pydantic models, OpenAPI docs"),
    "Docker":              ("intermediate","Images, containers, Dockerfiles, compose"),
    "Tableau":             ("intermediate","Dashboards, calculated fields, LOD expressions"),
    "Power BI":            ("intermediate","Reports, DAX, data modelling, dashboards"),
    "PostgreSQL":          ("intermediate","Advanced queries, indexing, stored procedures"),
    "MySQL":               ("intermediate","CRUD, joins, stored procedures, optimization"),
    "MongoDB":             ("intermediate","Documents, aggregation, indexing, Atlas"),
    "REST API":            ("intermediate","HTTP verbs, status codes, authentication, design"),
    "Feature Engineering": ("intermediate","Encoding, scaling, selection, creation"),
    "Jupyter":             ("intermediate","Notebooks, widgets, magic commands, sharing"),
    # Advanced
    "TensorFlow":          ("advanced","Neural networks, CNNs, RNNs, transfer learning"),
    "PyTorch":             ("advanced","Dynamic graphs, custom layers, research models"),
    "Keras":               ("advanced","High-level DL API, callbacks, custom layers"),
    "Transformers":        ("advanced","BERT, GPT fine-tuning, Hugging Face pipelines"),
    "Hugging Face":        ("advanced","Model hub, pipelines, fine-tuning, tokenizers"),
    "LLM":                 ("advanced","Prompt engineering, RAG, fine-tuning LLMs"),
    "AWS":                 ("advanced","SageMaker, Lambda, S3, EC2, IAM"),
    "Azure":               ("advanced","ML Studio, Blob storage, Functions, AKS"),
    "GCP":                 ("advanced","Vertex AI, BigQuery, Cloud Run, AutoML"),
    "Kubernetes":          ("advanced","Pods, services, Helm charts, scaling, ingress"),
    "Apache Spark":        ("advanced","Distributed DataFrames, MLlib, streaming"),
    "MLflow":              ("advanced","Experiment tracking, model registry, serving"),
    "Model Deployment":    ("advanced","Containerization, REST serving, monitoring"),
    "CI/CD":               ("advanced","GitHub Actions, Jenkins, pipelines, automation"),
    "NLP":                 ("advanced","Text preprocessing, embeddings, seq2seq models"),
    "Deep Learning":       ("advanced","Architectures, optimizers, regularization"),
    "Computer Vision":     ("advanced","CNNs, object detection, segmentation, OpenCV"),
    "Machine Learning":    ("intermediate","Supervised/unsupervised learning, model evaluation"),
    "Data Analysis":       ("beginner","Exploratory analysis, visualization, reporting"),
    "Data Science":        ("intermediate","End-to-end pipelines, modeling, storytelling"),
    "web development":       ("intermediate","Frontend/backend frameworks, databases, APIs"),
    "data visualization":       ("intermediate","Visualization libraries, dashboarding, storytelling"),
    "cloud computing":       ("intermediate","Cloud providers, services, deployment"),
    "graphic design":       ("beginner","Design principles, Adobe tools, typography"),
    "ui/ux design":       ("intermediate","User research, wireframing, prototyping, usability testing"),
    "mobile development":       ("intermediate","Android/iOS platforms, cross-platform tools, app deployment"),
    "cybersecurity":       ("intermediate","Threat modeling, network security, encryption, penetration testing"),
    "devops":       ("intermediate","Infrastructure as code, monitoring, automation, cloud services"),
    "adobe photoshop":       ("intermediate","Photo editing, compositing, retouching, filters"),
    "adobe illustrator":       ("intermediate","Vector graphics, logo design, typography"),
    "adobe indesign":       ("intermediate","Layout design, print/digital publishing, typography"),
    "sketch":       ("intermediate","UI design, prototyping, vector editing"),
    "figma":       ("intermediate","Collaborative design, prototyping, design systems"),
    "coreldraw":       ("intermediate","Vector graphics, page layout, typography"),
    "canva":       ("beginner","Templates, drag-and-drop design, social media graphics"),
    "gimp":       ("intermediate","Photo editing, compositing, open-source alternative to Photoshop"),
    "inkscape":       ("intermediate","Vector graphics, open-source alternative to Illustrator"),
    "affinity designer":       ("intermediate","Vector/raster design, alternative to Adobe Illustrator")
                                


}


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def extract_skills(text: str) -> list:
    text_l = text.lower()
    found = set()

    for canonical, aliases in TECH_SKILLS_MAP:
        for alias in aliases:
            try:
                if re.search(rf"\b{alias}\b", text_l):
                    found.add(canonical)
                    break
            except:
                if alias in text_l:
                    found.add(canonical)
                    break

    return list(found)

# Role → expected skills mapping
# Used to AUGMENT short Adzuna descriptions so skill extraction works
# even when the snippet only mentions 2–3 skills explicitly.
ROLE_SKILL_MAP = {
    "data scientist": [
        "Python","Machine Learning","Deep Learning","Statistics","SQL",
        "Pandas","NumPy","Scikit-learn","TensorFlow","Data Analysis",
        "Matplotlib","Jupyter","Feature Engineering","NLP",
    ],
    "machine learning engineer": [
        "Python","Machine Learning","Deep Learning","TensorFlow","PyTorch",
        "Scikit-learn","Docker","Kubernetes","MLflow","AWS","REST API",
        "Feature Engineering","Model Deployment","SQL","Git",
    ],
    "data analyst": [
        "SQL","Python","Excel","Tableau","Power BI","Pandas","NumPy",
        "Data Analysis","Statistics","Matplotlib","Seaborn","MySQL",
        "PostgreSQL",
    ],
    "nlp engineer": [
        "Python","NLP","SpaCy","NLTK","Transformers","Hugging Face",
        "BERT","Machine Learning","FastAPI","Flask","SQL","Git",
        "Deep Learning","TensorFlow","PyTorch",
    ],
    "ai engineer": [
        "Python","Machine Learning","Deep Learning","TensorFlow","PyTorch",
        "NLP","Computer Vision","Docker","AWS","REST API","Git","Linux",
        "Feature Engineering","Model Deployment",
    ],
    "python developer": [
        "Python","Flask","Django","FastAPI","REST API","SQL","PostgreSQL",
        "Docker","Git","Linux","HTML/CSS","JavaScript",
    ],
    "business analyst": [
        "SQL","Excel","Tableau","Power BI","Data Analysis","Statistics",
        "Python","Pandas","Communication","Agile",
    ],
    "data engineer": [
        "Python","SQL","Apache Spark","Hadoop","Kafka","Apache Airflow",
        "AWS","Azure","GCP","Docker","PostgreSQL","Git","Linux","Databricks",
    ],
    "software engineer": [
        "Python","Java","JavaScript","SQL","Git","Docker","REST API",
        "Linux","Agile","HTML/CSS","Node.js",
    ],
    "cloud engineer": [
        "AWS","Azure","GCP","Docker","Kubernetes","Linux","Python",
        "CI/CD","Terraform","Git","Bash/Shell",
    ],
    "full stack developer": [
        "JavaScript","TypeScript","React","Node.js","HTML/CSS","SQL",
        "MongoDB","REST API","Git","Docker","Python",
    ],
    "Graphic designer": [
        "Adobe Photoshop","Adobe Illustrator","Adobe InDesign","Sketch",
        "Figma","CorelDRAW","Canva","GIMP","Inkscape","Affinity Designer",
    ],
    "Digital marketer": [
        "SEO","Google Analytics","Content Marketing","Social Media Marketing"],
    "Product manager": [
        "Agile","Scrum","Kanban","Roadmapping","Stakeholder Management",
        "Communication","JIRA","Confluence","Product Analytics",
    ]     
}


def enrich_job_description(job: dict) -> str:
    """
    Adzuna descriptions are short (100–200 chars).
    This function builds a RICH text blob for a job by combining:
      1. The raw Adzuna description
      2. The job title keywords
      3. Role-based expected skills (from ROLE_SKILL_MAP)
    This ensures extract_skills() has enough text to work with.
    """
    base = job.get("description", "")
    title = job.get("title", "").lower()
    role_key = job.get("search_role", "").lower()

    # Find best matching role in map
    extra_skills = []
    for role, skills in ROLE_SKILL_MAP.items():
        if any(word in title for word in role.split()):
            extra_skills = skills
            break
    if not extra_skills:
        for role, skills in ROLE_SKILL_MAP.items():
            if any(word in role_key for word in role.split()):
                extra_skills = skills
                break

    # Combine everything into one rich text blob
    enriched = base + " " + title + " " + " ".join(s.lower() for s in extra_skills)
    return enriched


def preprocess(text: str) -> str:
    text = clean_text(text)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    sw = set(stopwords.words("english")) - {"r", "c", "go"}
    return " ".join(t for t in tokens if t not in sw and len(t) > 1)


def parse_resume_file(uploaded_file) -> str:
    """Parse PDF / DOCX / TXT resume file."""
    name = uploaded_file.name.lower()
    text = ""
    try:
        if name.endswith(".pdf"):
            try:
                import pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text += t + "\n"
            except ImportError:
                # fallback: read raw bytes as text
                raw = uploaded_file.read()
                text = raw.decode("latin-1", errors="ignore")
        elif name.endswith(".docx"):
            try:
                import docx as python_docx
                doc = python_docx.Document(uploaded_file)
                text = "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Could not read file: {e}")
    return text.strip()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_jobs_adzuna(query: str, num_results: int = 8) -> list:
    """
    Fetch live jobs from Adzuna API.
    Tries India (in) first, falls back to GB, then US.
    Returns list of job dicts.
    """
    if ADZUNA_APP_ID == "YOUR_APP_ID":
        return []  # API not configured

    countries_to_try = ["in", "gb", "us"]
    jobs = []

    for country in countries_to_try:
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
        params = {
            "app_id": "ADZUNA_APP_ID",
            "app_key": "ADZUNA_APP_KEY",
            "what": query,
            "results_per_page": num_results,
            "content-type": "application/json",
            "sort_by": "relevance",
        }
        try:
            resp = requests.get(url, params=params, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    for r in results:
                        jobs.append({
                            "title":       r.get("title", ""),
                            "company":     r.get("company", {}).get("display_name", "N/A"),
                            "location":    r.get("location", {}).get("display_name", "N/A"),
                            "description": clean_text(r.get("description", "")),
                            "salary_min":  r.get("salary_min"),
                            "salary_max":  r.get("salary_max"),
                            "url":         r.get("redirect_url", "#"),
                            "country":     country.upper(),
                        })
                    break  # got results, stop trying more countries
        except Exception:
            continue

    return jobs


def compute_scores(resume_text: str, jobs: list) -> list:
    """
    TF-IDF + Cosine Similarity matching.
    - Uses enrich_job_description() so job skill extraction works
      even on short Adzuna snippets.
    - Extracts skills from BOTH resume and enriched job description.
    - matched_skills  = skills in BOTH resume AND job
    - missing_skills  = skills in job but NOT in resume
    Returns jobs list sorted by match_score (0–100).
    """
    if not jobs or not resume_text.strip():
        return jobs

    resume_skills = extract_skills(resume_text)
    resume_proc   = preprocess(resume_text)

    # Build enriched descriptions for TF-IDF AND skill extraction
    enriched_texts = [enrich_job_description(j) for j in jobs]
    job_procs      = [preprocess(t) for t in enriched_texts]

    corpus = [resume_proc] + job_procs

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=1,
        sublinear_tf=True,
    )
    try:
        tfidf = vectorizer.fit_transform(corpus)
    except Exception:
        for j in jobs:
            j.update({"match_score": 0, "matched_skills": [], "missing_skills": [],
                       "job_skills": [], "resume_skills": resume_skills})
        return jobs

    resume_vec = tfidf[0]
    job_vecs   = tfidf[1:]
    raw_sims   = cosine_similarity(resume_vec, job_vecs)[0]

    # Scale raw cosine scores to meaningful 0–100 range
    max_sim = max(raw_sims) if max(raw_sims) > 0 else 1.0
    scores  = np.clip((raw_sims / max_sim) * 100, 0, 100).round(1)

    result = []
    for i, job in enumerate(jobs):
        # Extract skills from the enriched job text
        job_skills = extract_skills(enriched_texts[i])

        # ── THE CORE FIX ──────────────────────────────────────────
        # matched = resume ∩ job_required
        # missing = job_required − resume
        matched = [s for s in resume_skills if s in job_skills]
        missing = [s for s in job_skills    if s not in resume_skills]
        # ─────────────────────────────────────────────────────────

        result.append({
            **job,
            "match_score":    float(scores[i]),
            "matched_skills": matched,
            "missing_skills": missing[:10],   # top 10 missing
            "job_skills":     job_skills,     # all job-required skills
            "resume_skills":  resume_skills,  # all resume skills
        })

    return sorted(result, key=lambda x: x["match_score"], reverse=True)


def build_roadmap(missing_skills: list) -> dict:
    roadmap = {"beginner": [], "intermediate": [], "advanced": []}
    seen = set()
    for skill in missing_skills:
        if skill in seen:
            continue
        seen.add(skill)
        # Direct match on canonical name (keys are already canonical)
        if skill in SKILL_LEVELS:
            level, desc = SKILL_LEVELS[skill]
        else:
            level, desc = "intermediate", f"Learn {skill} fundamentals and build hands-on projects"
        roadmap[level].append({"skill": skill, "desc": desc})
    return roadmap


def score_color(score):
    if score >= 70: return "#059669", "score-high"
    if score >= 45: return "#6366f1", "score-mid"
    return "#d97706", "score-low"


def salary_str(mn, mx):
    if mn and mx: return f"₹{mn:,.0f}–₹{mx:,.0f}"
    if mn:        return f"₹{mn:,.0f}+"
    if mx:        return f"Up to ₹{mx:,.0f}"
    return "Not disclosed"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB: HOME / DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.session_state.tab == "home":

    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-tag">✦ AI-Powered Career Intelligence</div>
        <div class="hero-title">Match Your Resume to<br><em>Real Jobs, Right Now</em></div>
        <div class="hero-sub">
            Upload your resume. Our AI fetches live job listings, scores your match, identifies skill gaps, and builds your personal learning roadmap — all in seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat strip
    jobs_count = len(st.session_state.matched_jobs)
    skills_count = len(st.session_state.resume_skills)
    top_score = f"{st.session_state.matched_jobs[0]['match_score']:.0f}%" if st.session_state.matched_jobs else "—"
    st.markdown(f"""
    <div class="stat-strip">
        <div class="stat-item"><div class="stat-num">Adzuna</div><div class="stat-lbl">Live Job Source</div></div>
        <div class="stat-item"><div class="stat-num">{jobs_count if jobs_count else "—"}</div><div class="stat-lbl">Jobs Fetched</div></div>
        <div class="stat-item"><div class="stat-num">{skills_count if skills_count else "—"}</div><div class="stat-lbl">Skills Extracted</div></div>
        <div class="stat-item"><div class="stat-num">{top_score}</div><div class="stat-lbl">Top Match Score</div></div>
        <div class="stat-item"><div class="stat-num">TF-IDF</div><div class="stat-lbl">Algorithm</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Two column layout
    left, right = st.columns([1.1, 1])

    with left:
        # ── Upload & Configure ──────────────────────────────────
        st.markdown("""
        <div class="glass-card">
        <div class="section-title">Upload Your Resume</div>
        <div class="section-sub">PDF, DOCX, or TXT — parsed instantly</div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Resume file", type=["pdf","docx","txt"],
                                    label_visibility="collapsed")
        st.markdown("**or paste text below**")
        manual = st.text_area("", height=120,
                              placeholder="Paste your resume content here...",
                              label_visibility="collapsed")

        roles = st.multiselect(
            "Target Job Roles",
            ["Data Scientist","Machine Learning Engineer","Data Analyst",
             "NLP Engineer","AI Engineer","Python Developer",
             "Business Analyst","Data Engineer","Software Engineer",
             "Cloud Engineer","Full Stack Developer","Graphic Designer","Digital Marketer","Product Manager","Other"],
            default=["Data Scientist","Data Analyst"],
        )

        c1, c2 = st.columns(2)
        with c1:
            num_jobs = st.slider("Jobs per role", 3, 10, 5, label_visibility="visible")
        with c2:
            use_api = ADZUNA_APP_ID != "YOUR_APP_ID"
            st.markdown(f"""
            <div style="padding:10px 0;">
                <div style="font-size:.75rem;color:#9ca3af;margin-bottom:4px;">API STATUS</div>
                <div style="font-size:.85rem;font-weight:600;color:{'#059669' if use_api else '#d97706'};">
                    {'● Connected' if use_api else '● Key Needed'}
                </div>
            </div>""", unsafe_allow_html=True)

        if st.button("✦  Analyze & Match Jobs", key="main_run"):
            # Get resume text
            rtext = ""
            if uploaded:
                rtext = parse_resume_file(uploaded)
            elif manual.strip():
                rtext = manual.strip()
            else:
                st.error("Please upload or paste your resume first.")
                st.stop()

            if not roles:
                st.error("Select at least one job role.")
                st.stop()

            all_jobs = []
            prog = st.progress(0)

            if use_api:
                total = len(roles)
                for idx, role in enumerate(roles):
                    prog.progress(int((idx / total) * 55),
                                  text=f"🔄 Fetching '{role}' jobs from Adzuna...")
                    jobs = fetch_jobs_adzuna(role, num_jobs)
                    for j in jobs:
                        j["search_role"] = role
                    all_jobs.extend(jobs)
                    time.sleep(0.3)

                if not all_jobs:
                    prog.empty()
                    st.warning("⚠️ Adzuna returned no results. Check your API keys or try different roles.")
                    st.stop()
            else:
                # Demo mode — realistic mock data
                prog.progress(30, text="Demo mode: using sample job data...")
                time.sleep(0.4)
                all_jobs = (roles, rtext)

            prog.progress(65, text="🧠 Running TF-IDF vectorization + cosine similarity...")
            matched = compute_scores(rtext, all_jobs)
            prog.progress(85, text="📊 Computing skill gaps and roadmap...")

            # Build roadmap from top 5 jobs
            all_missing = []
            for j in matched[:5]:
                all_missing.extend(j.get("missing_skills", []))
            roadmap = build_roadmap(list(set(all_missing)))

            prog.progress(100, text="✅ Complete!")
            time.sleep(0.3)
            prog.empty()

            st.session_state.matched_jobs  = matched
            st.session_state.roadmap       = roadmap
            st.session_state.resume_text   = rtext
            st.session_state.resume_skills = extract_skills(rtext)
            st.session_state.analyzed      = True
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if st.session_state.analyzed and st.session_state.matched_jobs:
            top  = st.session_state.matched_jobs[0]
            sc   = top["match_score"]
            col, cls = score_color(sc)

            # Score display
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:.75rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:12px;">Best Match Score</div>
                <div style="font-family:'Lora',serif;font-size:4.5rem;font-weight:600;color:{col};line-height:1;letter-spacing:-0.04em;">{sc:.0f}%</div>
                <div style="font-size:.88rem;color:#6b7280;margin:10px 0 4px;font-weight:500;">{top['title'][:40]}</div>
                <div style="font-size:.78rem;color:#9ca3af;">{top['company']}</div>
                <div style="margin-top:16px;font-size:.78rem;color:{'#059669' if sc>=70 else '#6366f1' if sc>=45 else '#d97706'};font-weight:600;">
                    {'🟢 Strong Match' if sc>=70 else '🔵 Good Match' if sc>=45 else '🟡 Fair Match'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Skills found
            skills = st.session_state.resume_skills
            if skills:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="section-title" style="font-size:1rem;">Skills Extracted ({len(skills)})</div>', unsafe_allow_html=True)
                tags = "".join([f'<span class="skill-pill pill-neutral">{s}</span>' for s in skills])
                st.markdown(f'<div style="line-height:2.3;">{tags}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Quick preview of top 3 matches
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:1rem;">Top 3 Job Matches</div>', unsafe_allow_html=True)
            for j in st.session_state.matched_jobs[:3]:
                s   = j["match_score"]
                c,_ = score_color(s)
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:10px 14px;background:rgba(238,242,255,0.5);border-radius:10px;margin-bottom:8px;">
                    <div>
                        <div style="font-size:.85rem;font-weight:600;color:#1e1b4b;">{j['title'][:32]}</div>
                        <div style="font-size:.75rem;color:#9ca3af;">{j['company'][:28]}</div>
                    </div>
                    <div style="font-family:'Lora',serif;font-size:1.2rem;font-weight:600;color:{c};">{s:.0f}%</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Empty state
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:50px 30px;">
                <div style="font-size:3.5rem;margin-bottom:16px;opacity:0.4;">✦</div>
                <div style="font-family:'Lora',serif;font-size:1.1rem;color:#6b7280;margin-bottom:8px;">Ready to Match</div>
                <div style="font-size:.82rem;color:#9ca3af;line-height:1.6;">
                    Upload your resume on the left and click<br>
                    <b style="color:#6366f1;">Analyze & Match Jobs</b> to get started.
                </div>
            </div>
            <div class="glass-card glass-card-hover" style="text-align:center;padding:20px;">
                <div style="font-size:.8rem;color:#9ca3af;margin-bottom:8px;font-weight:500;">Live data from</div>
                <div style="font-family:'Lora',serif;font-size:1.3rem;color:#6366f1;font-weight:600;">Adzuna India</div>
                <div style="font-size:.75rem;color:#9ca3af;margin-top:4px;">developer.adzuna.com</div>
            </div>
            """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB: JOBS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.tab == "jobs":
    st.markdown('<div class="section-title">Live Job Matches</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Jobs fetched live · sorted by AI match score</div>', unsafe_allow_html=True)

    if not st.session_state.matched_jobs:
        st.markdown('<div class="info-banner">ℹ️ No jobs yet — go to <b>Dashboard</b>, upload your resume and click Analyze.</div>', unsafe_allow_html=True)
        st.stop()

    jobs = st.session_state.matched_jobs

    # Filter bar
    fc1, fc2, fc3 = st.columns(3)
    with fc1: min_s = st.slider("Min Match %", 0, 100, 0)
    with fc2:
        companies = ["All"] + sorted(list(set(j["company"] for j in jobs if j["company"] != "N/A")))
        comp = st.selectbox("Company", companies, label_visibility="visible")
    with fc3:
        sort = st.selectbox("Sort", ["Match ↓", "Match ↑"], label_visibility="visible")

    filtered = [j for j in jobs if j["match_score"] >= min_s]
    if comp != "All":
        filtered = [j for j in filtered if j["company"] == comp]
    if sort == "Match ↑":
        filtered = sorted(filtered, key=lambda x: x["match_score"])

    st.markdown(f'<div style="font-size:.78rem;color:#9ca3af;margin-bottom:16px;">Showing <b style="color:#6366f1;">{len(filtered)}</b> of {len(jobs)} jobs</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for i, job in enumerate(filtered):
        col = col1 if i % 2 == 0 else col2
        sc = job["match_score"]
        c, cls = score_color(sc)
        m_tags = "".join([f'<span class="skill-pill pill-have" style="font-size:.7rem;">{s}</span>' for s in job.get("matched_skills", [])[:3]])
        sal = salary_str(job.get("salary_min"), job.get("salary_max"))
        ctry = job.get("country", "")

        with col:
            st.markdown(f"""
            <div class="job-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
                    <div class="job-title">{job['title'][:42]}</div>
                    <span class="score-badge {cls}">{sc:.0f}%</span>
                </div>
                <div class="job-meta">🏢 {job['company'][:28]} &nbsp;·&nbsp; 📍 {job['location'][:22]} &nbsp;·&nbsp; 💰 {sal}
                    {"&nbsp;·&nbsp; 🌐 "+ctry if ctry else ""}
                </div>
                <div style="font-size:.77rem;color:#9ca3af;line-height:1.5;margin-bottom:10px;">{job['description'][:110]}…</div>
                <div>{m_tags}</div>
                <div style="margin-top:10px;">
                    <div class="prog-bg"><div class="prog-fill" style="width:{sc}%;"></div></div>
                </div>
                <div style="margin-top:10px;font-size:.75rem;">
                    <a href="{job['url']}" target="_blank" style="color:#6366f1;text-decoration:none;font-weight:600;">View Job →</a>
                </div>
            </div>""", unsafe_allow_html=True)

    # Chart
    st.markdown("<br>", unsafe_allow_html=True)
    chart_df = pd.DataFrame({
        "Job": [f"{j['title'][:18]}…" for j in filtered[:8]],
        "Match %": [round(j["match_score"], 1) for j in filtered[:8]]
    }).set_index("Job")
    st.bar_chart(chart_df, color="#6366f1")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB: ANALYZE (Skill Gap deep dive)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.tab == "analyze":
    st.markdown('<div class="section-title">Skill Gap Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Compare your skills against each job listing</div>', unsafe_allow_html=True)

    if not st.session_state.matched_jobs:
        st.markdown('<div class="info-banner">ℹ️ Run the analysis from the Dashboard first.</div>', unsafe_allow_html=True)
        st.stop()

    jobs = st.session_state.matched_jobs
    options = [f"{j['title'][:35]} @ {j['company'][:20]} — {j['match_score']:.0f}%" for j in jobs]
    sel_idx = st.selectbox("Select a job to analyze", range(len(options)),
                           format_func=lambda i: options[i])
    job = jobs[sel_idx]
    sc = job["match_score"]
    c, cls = score_color(sc)

    matched = job.get("matched_skills", [])
    missing = job.get("missing_skills", [])
    job_skills = job.get("job_skills", [])
    resume_skills = job.get("resume_skills", st.session_state.resume_skills)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Match Score",         f"{sc:.0f}%")
    m2.metric("Job Requires",        f"{len(job_skills)} skills")
    m3.metric("You Have",            f"{len(matched)} skills")
    m4.metric("Gap to Fill",         f"{len(missing)} skills")

    st.markdown("<br>", unsafe_allow_html=True)

    # All job required skills overview
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1rem;">📋 All Skills This Job Requires</div>', unsafe_allow_html=True)
    if job_skills:
        pills = ""
        for s in sorted(job_skills):
            cls2 = "pill-have" if s in matched else "pill-miss"
            pills += f'<span class="skill-pill {cls2}">{s}</span>'
        st.markdown(f'<div style="line-height:2.5;">{pills}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.75rem;color:#9ca3af;margin-top:10px;">🟢 Green = you have it &nbsp;&nbsp; 🟠 Orange = you need to learn it</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-banner">Job description too short to extract skills.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title" style="font-size:1rem;">✅ Your Matched Skills ({len(matched)})</div>', unsafe_allow_html=True)
        if matched:
            for s in matched:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:9px 14px;
                    background:rgba(236,253,245,0.7);border:1px solid #a7f3d0;
                    border-radius:10px;margin-bottom:7px;">
                    <span style="color:#059669;font-weight:700;">✓</span>
                    <span style="font-size:.85rem;color:#065f46;font-weight:500;">{s}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-banner">No direct skill matches found. Add more technical keywords to your resume.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title" style="font-size:1rem;">📌 Skills You Need to Learn ({len(missing)})</div>', unsafe_allow_html=True)
        if missing:
            for s in missing:
                level = SKILL_LEVELS.get(s, ("intermediate",""))[0]
                level_color = {"beginner":"#059669","intermediate":"#6366f1","advanced":"#7c3aed"}.get(level,"#6366f1")
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;padding:9px 14px;
                    background:rgba(255,247,237,0.7);border:1px solid #fed7aa;
                    border-radius:10px;margin-bottom:7px;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="color:#d97706;font-weight:700;">→</span>
                        <span style="font-size:.85rem;color:#92400e;font-weight:500;">{s}</span>
                    </div>
                    <span style="font-size:.7rem;font-weight:600;color:{level_color};
                        background:rgba(99,102,241,0.08);padding:2px 8px;border-radius:8px;">{level}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("🎉 No major skill gaps detected for this role!")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📄 Full Job Description"):
        st.markdown(f'<div style="font-size:.85rem;color:#6b7280;line-height:1.8;">{job["description"][:1200]}</div>', unsafe_allow_html=True)
        st.markdown(f"[🔗 Apply on Adzuna →]({job['url']})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB: ROADMAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.tab == "roadmap":
    st.markdown('<div class="section-title">Your Learning Roadmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Personalized path from your skill gaps · Beginner → Advanced</div>', unsafe_allow_html=True)

    if not st.session_state.roadmap:
        st.markdown('<div class="info-banner">ℹ️ Run the analysis from the Dashboard first.</div>', unsafe_allow_html=True)
        st.stop()

    roadmap = st.session_state.roadmap
    beg = roadmap.get("beginner", [])
    inter = roadmap.get("intermediate", [])
    adv = roadmap.get("advanced", [])

    # Summary strip
    st.markdown(f"""
    <div class="stat-strip" style="margin-bottom:22px;">
        <div class="stat-item"><div class="stat-num" style="color:#059669;">{len(beg)}</div><div class="stat-lbl">Beginner</div></div>
        <div class="stat-item"><div class="stat-num">{len(inter)}</div><div class="stat-lbl">Intermediate</div></div>
        <div class="stat-item"><div class="stat-num" style="color:#7c3aed;">{len(adv)}</div><div class="stat-lbl">Advanced</div></div>
        <div class="stat-item"><div class="stat-num">{len(beg)+len(inter)+len(adv)}</div><div class="stat-lbl">Total Skills</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["All Skills", "🟢 Beginner", "🔵 Intermediate · 🟣 Advanced"])

    def render_roadmap_items(items, badge_cls, badge_lbl, border_color):
        if not items:
            st.markdown('<div class="info-banner">Nothing at this level — you\'re all good here!</div>', unsafe_allow_html=True)
            return
        st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
        for item in items:
            st.markdown(f"""
            <div class="road-item">
                <div class="road-badge {badge_cls}">{badge_lbl}</div>
                <div>
                    <div class="road-name">{item['skill']}</div>
                    <div class="road-desc">{item['desc']}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab1:
        render_roadmap_items(beg, "rb-beg", "B", "#a7f3d0")
        render_roadmap_items(inter, "rb-int", "I", "#c7d2fe")
        render_roadmap_items(adv, "rb-adv", "A", "#ddd6fe")

    with tab2:
        render_roadmap_items(beg, "rb-beg", "B", "#a7f3d0")

    with tab3:
        render_roadmap_items(inter, "rb-int", "I", "#c7d2fe")
        render_roadmap_items(adv, "rb-adv", "A", "#ddd6fe")

    # Progress bars per skill
    if st.session_state.resume_skills:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1rem;">Skills You Already Have</div>', unsafe_allow_html=True)
        skills = st.session_state.resume_skills[:10]
        import random; random.seed(42)
        for skill in skills:
            pct = random.randint(55, 95)
            st.markdown(f"""
            <div class="prog-wrap">
                <div class="prog-row"><span>{skill}</span><span>{pct}%</span></div>
                <div class="prog-bg"><div class="prog-fill" style="width:{pct}%;"></div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB: SETTINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.tab == "settings":
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1rem;">🔑 Adzuna API Keys</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-banner">
            1. Go to <b>developer.adzuna.com</b><br>
            2. Register free → Create App<br>
            3. Copy APP_ID and APP_KEY<br>
            4. Paste below and save to <b>ai_career_match.py</b><br>
            <b>Free tier: 250 calls/day</b>
        </div>""", unsafe_allow_html=True)
        aid  = st.text_input("APP_ID",  placeholder="e.g. a1b2c3d4")
        akey = st.text_input("APP_KEY", placeholder="e.g. x9y8z7w6", type="password")
        if st.button("Show code to paste"):
            if aid and akey:
                st.code(f'ADZUNA_APP_ID  = "{aid}"\nADZUNA_APP_KEY = "{akey}"', language="python")
                st.info("Copy the code above and replace lines 24–25 in ai_career_match.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1rem;">👤 Profile</div>', unsafe_allow_html=True)
        st.text_input("Full Name", value="Namandeep Kaur")
        st.text_input("College",   value="GNDU Regional Campus, Jalandhar")
        st.selectbox("Degree",     ["B.Tech CSE","B.Tech IT","MCA","BCA"])
        st.selectbox("Year",       ["2026","2025","2027"])
        if st.button("Save Profile"):
            st.success("Saved!")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1rem;">📦 Install Libraries</div>', unsafe_allow_html=True)
    st.code("pip install streamlit pandas scikit-learn nltk pdfplumber python-docx requests", language="bash")
    st.code("streamlit run ai_career_match.py", language="bash")
    st.markdown('</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMO DATA (used when API key not configured)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _demo_jobs(roles, resume_text=""):
    """Realistic demo jobs with RICH descriptions for proper skill extraction."""
    base = [
        {"title":"Data Scientist","company":"Infosys BPM","location":"Bangalore, India",
         "search_role":"Data Scientist",
         "description":"We are looking for a Data Scientist with strong Python and machine learning skills. Required: python scikit-learn tensorflow pandas numpy statistics deep learning nlp data analysis sql tableau aws jupyter feature engineering classification regression model deployment",
         "salary_min":800000,"salary_max":1400000,"url":"#","country":"IN"},
        {"title":"Machine Learning Engineer","company":"TCS iON","location":"Hyderabad, India",
         "search_role":"Machine Learning Engineer",
         "description":"Seeking ML Engineer proficient in machine learning and deep learning. Must know: python tensorflow pytorch docker kubernetes mlflow model deployment aws gcp airflow feature engineering sql git linux ci/cd rest api",
         "salary_min":1000000,"salary_max":1800000,"url":"#","country":"IN"},
        {"title":"Data Analyst","company":"Wipro Analytics","location":"Pune, India",
         "search_role":"Data Analyst",
         "description":"Data Analyst role requiring strong analytical skills. Required: sql python pandas excel tableau power bi data analysis statistics matplotlib seaborn mysql postgresql business intelligence reporting",
         "salary_min":500000,"salary_max":900000,"url":"#","country":"IN"},
        {"title":"NLP Engineer","company":"Freshworks","location":"Chennai, India",
         "search_role":"NLP Engineer",
         "description":"NLP Engineer to build conversational AI systems. Required: python nlp spacy nltk transformers bert hugging face text classification named entity recognition sentiment analysis fastapi flask sql gcp deep learning pytorch",
         "salary_min":900000,"salary_max":1600000,"url":"#","country":"IN"},
        {"title":"AI Engineer","company":"Mphasis AI","location":"Bangalore, India",
         "search_role":"AI Engineer",
         "description":"AI Engineer for building intelligent systems. Required: python machine learning deep learning tensorflow pytorch nlp computer vision docker aws rest api fastapi git linux model deployment feature engineering",
         "salary_min":1100000,"salary_max":2000000,"url":"#","country":"IN"},
        {"title":"Python Developer","company":"HCL Technologies","location":"Noida, India",
         "search_role":"Python Developer",
         "description":"Python Backend Developer for enterprise systems. Required: python django flask fastapi rest api postgresql mysql git docker linux javascript html css agile ci/cd",
         "salary_min":600000,"salary_max":1200000,"url":"#","country":"IN"},
        {"title":"Business Analyst","company":"Accenture","location":"Mumbai, India",
         "search_role":"Business Analyst",
         "description":"Business Analyst with data skills. Required: sql excel power bi tableau data analysis statistics python pandas agile communication stakeholder management reporting visualization",
         "salary_min":700000,"salary_max":1300000,"url":"#","country":"IN"},
        {"title":"Data Engineer","company":"Cognizant","location":"Bangalore, India",
         "search_role":"Data Engineer",
         "description":"Data Engineer to build scalable pipelines. Required: python apache spark hadoop kafka apache airflow sql postgresql aws azure databricks etl pipeline git docker linux scala data warehouse",
         "salary_min":900000,"salary_max":1700000,"url":"#","country":"IN"},
        {"title":"Full Stack Developer","company":"Capgemini","location":"Pune, India",
         "search_role":"Full Stack Developer",
         "description":"Full Stack Developer for web applications. Required: javascript typescript react node.js html css mongodb rest api git docker python sql postgresql agile",
         "salary_min":700000,"salary_max":1300000,"url":"#","country":"IN"},
        {"title":"Cloud Engineer","company":"Wipro Cloud","location":"Bangalore, India",
         "search_role":"Cloud Engineer",
         "description":"Cloud Engineer for multi-cloud environments. Required: aws azure gcp docker kubernetes linux python ci/cd git terraform shell bash infrastructure automation",
         "salary_min":900000,"salary_max":1600000,"url":"#","country":"IN"},
    ]
    # Filter to match requested roles
    result = []
    for job in base:
        for role in roles:
            if any(word.lower() in job["title"].lower() for word in role.split()):
                result.append(job)
                break
    # If nothing matched, return all
    if not result:
        result = base[:5]
    return result
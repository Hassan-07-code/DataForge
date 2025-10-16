"""
DataForge - AutoClean + AutoML
Main controller app with sidebar navigation.
"""

import os
import streamlit as st

# UI imports
from ui.home import render_home
from ui.cleaning_ui import render_cleaning
from ui.visualization_ui import render_visualization
from ui.model_ui import render_model_training
from ui.export_ui import render_export

# Initialize folders
ROOT = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(ROOT, "reports")
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Page config
st.set_page_config(
    page_title="DataForge — AutoClean + AutoML",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "activity_log" not in st.session_state:   # ✅ FIX: structured activity_log
    st.session_state.activity_log = {
        "cleaning": [],
        "visualizations": [],
        "models": []
    }

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio(
    "Go to section",
    ["Home", "Data Cleaning", "Visualization", "Model Training", "Export"],
    index=0,
)

# Render based on section
if section == "Home":
    render_home()

elif section == "Data Cleaning":
    if st.session_state.df_raw is not None:
        render_cleaning(st.session_state.df_raw)
    else:
        st.warning("⚠️ Please upload a dataset first in Home.")

elif section == "Visualization":
    if st.session_state.df_clean is not None:
        render_visualization(st.session_state.df_clean)
    else:
        st.warning("⚠️ Please clean your dataset first in Data Cleaning.")

elif section == "Model Training":
    if st.session_state.df_clean is not None:
        render_model_training(st.session_state.df_clean, MODELS_DIR)
    else:
        st.warning("⚠️ Please clean your dataset first in Data Cleaning.")

elif section == "Export":
    render_export(REPORTS_DIR, MODELS_DIR)

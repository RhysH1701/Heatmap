import io
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Page / App Config
# ----------------------------
st.set_page_config(page_title="Scout Dashboard", layout="wide")

# Custom header styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #1E3D59;
        margin-bottom: 0.25em;
    }
    .subtitle {
        font-size: 16px;
        color: #4F6D7A;
        margin-bottom: 1em;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    .soft-divider {height: 1px; background: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.12), rgba(0,0,0,0)); margin: 1.2rem 0;}
    .badge {display:inline-block; font-size:12px; padding:4px 10px; border-radius:999px; border:1px solid rgba(0,0,0,0.12); background: rgba(0,0,0,0.03); margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True
)

col_hero_left, col_hero_right = st.columns([3,1])
with col_hero_left:
    st.markdown('<div class="main-title">Scout Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive scouting and efficiency analysis tool</div>', unsafe_allow_html=True)
    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<span class='badge'>Heatmap</span><span class='badge'>Table</span><span class='badge'>Breakdowns</span><span class='badge'>CSV/PNG Export</span>",
        unsafe_allow_html=True,
    )
with col_hero_right:
    st.markdown("#### ")
    st.markdown("**Version** 0.1.0  ")
    st.markdown("**Updated** today  ")
    st.markdown("**Theme** Light")

# ----------------------------
# Helper Utils
# ----------------------------
@st.cache_data(show_spinner=False)
def load_excel(file: bytes) -> pd.DataFrame:
    """Load Excel to DataFrame with engine fallback."""
    try:
        df = pd.read_excel(file, engine="openpyxl")
    except Exception:
        df = pd.read_excel(file)
    # Standardize column names (strip/upper) for robust access
    df.columns = [c.strip() for c in df.columns]
    return df


def has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


# Zone normalization based on PASS AREA text
ZONE_MAP_CANVAS = {
    'FLAT LEFT': (0, 0, 10, 15),
    'HOOK/CURL LEFT': (10, 0, 10, 15),
    'MIDDLE HOLE': (20, 0, 13.3, 15),
    'HOOK/CURL RIGHT': (33.3, 0, 10, 15),
    'FLAT RIGHT': (43.3, 0, 10, 15),
    'DEEP LEFT': (0, 15, 17.7, 15),
    'DEEP MIDDLE': (17.7, 15, 17.7, 15),
    'DEEP RIGHT': (35.4, 15, 17.7, 15)
}


def normalize_zone(val: str) -> str:
    if not isinstance(val, str):
        return "UNKNOWN"
    zone = val.upper()
    if "FLAT" in zone:
        return "FLAT RIGHT" if "RIGHT" in zone else "FLAT LEFT"
    if "HOOK/CURL" in zone:
        return "HOOK/CURL RIGHT" if "RIGHT" in zone else "HOOK/CURL LEFT"
    if "DEEP" in zone:
        if "LEFT" in zone:
            return "DEEP LEFT"
        if "RIGHT" in zone:
            return "DEEP RIGHT"
        return "DEEP MIDDLE"
    if "MIDDLE" in zone and "DEEP" not in zone:
        return "MIDDLE HOLE"
    return zone


# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Upload & Filters")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"]) 

example_note = st.sidebar.expander("Data schema (minimum)", expanded=False)
with example_note:
    st.markdown(
        """
        **Required columns:**  
        • `PLAY TYPE` (Pass/Run)  
        • `PASS AREA` (zone text)  
        • `RESULT` (Complete/Incomplete/INT)  
        
        **Optional columns:** `YARDS`, `AIR YARDS`, `YAC`, `EPA`, `QUARTER`, `DOWN`, `DISTANCE`, `HASH`, `FORMATION`, `PERSONNEL`, `QB`, `TARGET`, `OPPONENT`, `DATE`
        """
    )

if uploaded is None:
    st.info("Upload a dataset to begin. Or toggle demo mode below.")
    if st.checkbox("Use demo data"):
        demo = pd.DataFrame({
            "PLAY TYPE": ["Pass", "Pass", "Pass", "Run", "Pass", "Pass", "Pass"],
            "PASS AREA": ["Flat Left", "Hook/Curl Right", "Middle", "N/A", "Deep Right", "Flat Right", "Hook/Curl Left"],
            "RESULT": ["Complete", "Incomplete", "Complete", "N/A", "Complete", "Incomplete", "Complete"],
            "YARDS": [5, 0, 12, 4, 28, 0, 9],
            "QUARTER": [1,1,2,2,3,4,4],
            "DOWN": [1,3,2,2,1,3,2],
            "DISTANCE": [10,6,8,3,10,7,9],
            "OPPONENT": ["Hawks","Hawks","Hawks","Hawks","Bears","Bears","Bears"],
            "QB": ["QB1","QB1","QB1","QB1","QB1","QB1","QB1"],
            "TARGET": ["WR1","TE1","WR2","RB1","WR1","RB1","WR3"],
        })
        df_raw = demo
    else:
        st.stop()
else:
    df_raw = load_excel(uploaded)

# ----------------------------
# Defensive normalization (rest of the app remains unchanged)
# ----------------------------

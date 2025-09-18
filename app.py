import io
import zipfile
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

# ----------------------------
# Global Styles
# ----------------------------
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
    .soft-divider {
        height: 1px;
        background: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.12), rgba(0,0,0,0));
        margin: 1.2rem 0;
    }
    .kpi-card {
        border: 1px solid rgba(0,0,0,0.08);
        padding: 1rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.65);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Header
# ----------------------------
col_hero_left, col_hero_right = st.columns([3, 1])
with col_hero_left:
    st.markdown('<div class="main-title">Scout Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive scouting and efficiency analysis tool</div>', unsafe_allow_html=True)
with col_hero_right:
    st.markdown("#### ")
    st.markdown("**Version** 0.1.0  ")
    st.markdown(f"**Updated** {datetime.now().date()}  ")
    st.markdown("**Theme** Light")

st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Upload & Filters")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (.xlsx, .xls, .csv, or .zip)",
    type=["xlsx", "xls", "csv", "zip"],
    accept_multiple_files=True,
)
st.sidebar.caption("Tip: Hold Ctrl (Windows) / Cmd (Mac) to pick multiple files, or drag them all at once. On mobile, multi-select may not be supported — upload a .zip instead.")

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

st.sidebar.subheader("Display")
heatmap_metric = st.sidebar.selectbox("Heatmap color by", ["Completions", "Attempts", "Success Rate"], index=0)

# ----------------------------
# Data Load
# ----------------------------
@st.cache_data(show_spinner=False)
def load_any(file_bytes: bytes, filename: str) -> list[pd.DataFrame]:
    """Load one uploaded item (xlsx/xls/csv/zip) and return list of DataFrames.
    Adds a __DATASET__ column derived from filename/member name.
    """
    name_lower = filename.lower()
    out: list[pd.DataFrame] = []

    def _standardize(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        df["__DATASET__"] = tag
        return df

    bio = io.BytesIO(file_bytes)

    if name_lower.endswith(".csv"):
        df = pd.read_csv(bio)
        out.append(_standardize(df, filename))
    elif name_lower.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(bio, engine="openpyxl")  # for .xlsx
        except Exception:
            bio.seek(0)
            df = pd.read_excel(bio)  # fallback (may work for .xls if engine available)
        out.append(_standardize(df, filename))
    elif name_lower.endswith(".zip"):
        with zipfile.ZipFile(bio) as zf:
            for member in zf.namelist():
                if member.lower().endswith(".csv"):
                    with zf.open(member) as f:
                        df = pd.read_csv(f)
                        out.append(_standardize(df, f"{filename}:{member}"))
                elif member.lower().endswith((".xlsx", ".xls")):
                    with zf.open(member) as f:
                        data = f.read()
                        mbio = io.BytesIO(data)
                        try:
                            df = pd.read_excel(mbio, engine="openpyxl")
                        except Exception:
                            mbio.seek(0)
                            df = pd.read_excel(mbio)
                        out.append(_standardize(df, f"{filename}:{member}"))
    else:
        # Unknown extension — try CSV then Excel fallback
        try:
            bio.seek(0)
            df = pd.read_csv(bio)
            out.append(_standardize(df, filename))
        except Exception:
            bio.seek(0)
            df = pd.read_excel(bio)
            out.append(_standardize(df, filename))

    return out

# Load one or more files
if not uploaded_files:
    st.info("Upload one or more datasets to begin. Or toggle demo mode below.")
    if st.checkbox("Use demo data"):
        demo = pd.DataFrame({
            "PLAY TYPE": ["Pass", "Pass", "Pass", "Run", "Pass", "Pass", "Pass"],
            "PASS AREA": ["Flat Left", "Hook/Curl Right", "Middle", "N/A", "Deep Right", "Flat Right", "Hook/Curl Left"],
            "RESULT": ["Complete", "Incomplete", "Complete", "N/A", "Complete", "Incomplete", "Complete"],
            "YARDS": [5, 0, 12, 4, 28, 0, 9],
            "QUARTER": [1, 1, 2, 2, 3, 4, 4],
            "DOWN": [1, 3, 2, 2, 1, 3, 2],
            "DISTANCE": [10, 6, 8, 3, 10, 7, 9],
            "OPPONENT": ["Hawks", "Hawks", "Hawks", "Hawks", "Bears", "Bears", "Bears"],
            "QB": ["QB1", "QB1", "QB1", "QB1", "QB1", "QB1", "QB1"],
            "TARGET": ["WR1", "TE1", "WR2", "RB1", "WR1", "RB1", "WR3"],
        })
        demo["__DATASET__"] = "DEMO"
        df_raw = demo
    else:
        st.stop()
else:
    frames = []
    for i, f in enumerate(uploaded_files):
        try:
            bytes_data = f.read()
        finally:
            f.seek(0)
        loaded_frames = load_any(bytes_data, getattr(f, "name", f"Dataset {i+1}"))
        frames.extend(loaded_frames)
    if len(frames) == 0:
        st.error("No readable files found. Ensure they are .xlsx, .xls, .csv or a .zip containing those.")
        st.stop()
    df_raw = pd.concat(frames, ignore_index=True)

# ----------------------------
# Dataset filter (choose which uploaded datasets to include)
# ----------------------------
if "__DATASET__" in df_raw.columns:
    ds_all = sorted(df_raw["__DATASET__"].dropna().unique().tolist())
    sel_ds = st.sidebar.multiselect("Datasets", ds_all, default=ds_all)
    if sel_ds:
        df_raw = df_raw[df_raw["__DATASET__"].isin(sel_ds)]
    else:
        st.warning("No datasets selected.")
        st.stop()

# --- Tag datasets (perspective + teams) ---
if "__DATASET__" in df_raw.columns:
    if "dataset_tags" not in st.session_state:
        st.session_state["dataset_tags"] = {}  # { dataset_name: {"perspective":..., "us":..., "them":...} }

    with st.sidebar.expander("Tag datasets"):
        for ds in sorted(df_raw["__DATASET__"].dropna().unique()):
            current = st.session_state["dataset_tags"].get(ds, {"perspective": "Unlabeled", "us": "", "them": ""})
            st.markdown(f"**{ds}**")
            perspective = st.selectbox(
                f"Perspective – {ds}",
                ["Unlabeled", "Our Offense vs Them", "Their Offense vs Our Defense"],
                index=["Unlabeled", "Our Offense vs Them", "Their Offense vs Our Defense"].index(current["perspective"]),
                key=f"persp_{ds}",
            )
            us_team = st.text_input(f"Our team – {ds}", value=current["us"], key=f"us_{ds}")
            them_team = st.text_input(f"Opponent – {ds}", value=current["them"], key=f"them_{ds}")

            # persist in session
            st.session_state["dataset_tags"][ds] = {"perspective": perspective, "us": us_team, "them": them_team}

    # Apply tags to the rows of df_raw (auto each run)
    tag_map = st.session_state.get("dataset_tags", {})
    def _get(ds, key, default):
        return tag_map.get(ds, {}).get(key, default)

    df_raw["__PERSPECTIVE__"] = df_raw["__DATASET__"].map(lambda d: _get(d, "perspective", "Unlabeled"))
    df_raw["__US_TEAM__"]     = df_raw["__DATASET__"].map(lambda d: _get(d, "us", ""))
    df_raw["__OPP_TEAM__"]    = df_raw["__DATASET__"].map(lambda d: _get(d, "them", ""))


# ----------------------------
# Defensive normalization
# ----------------------------
cols_upper = {c.upper(): c for c in df_raw.columns}

COL_PLAY_TYPE = cols_upper.get("PLAY TYPE", "PLAY TYPE")
COL_PASS_AREA = cols_upper.get("PASS AREA", "PASS AREA")
COL_RESULT = cols_upper.get("RESULT", "RESULT")

if not all(c in df_raw.columns for c in [COL_PLAY_TYPE, COL_PASS_AREA, COL_RESULT]):
    st.error("Dataset must include PLAY TYPE, PASS AREA, and RESULT columns.")
    st.stop()

# Zone map for canvas layout
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

# Pass-only subset and normalized zone
pass_df = df_raw[df_raw[COL_PLAY_TYPE].astype(str).str.upper() == "PASS"].copy()
pass_df["ZONE"] = pass_df[COL_PASS_AREA].apply(normalize_zone)

# Optional metadata fields
opt_cols = {name: cols_upper.get(name, None) for name in [
    "YARDS", "AIR YARDS", "YAC", "EPA", "QUARTER", "DOWN", "DISTANCE", "HASH", "FORMATION", "PERSONNEL", "QB", "TARGET", "OPPONENT", "DATE"
]}

# ----------------------------
# Filters
# ----------------------------
filtered = pass_df

st.sidebar.subheader("Filters")
if opt_cols["OPPONENT"] and filtered[opt_cols["OPPONENT"]].notna().any():
    opps = ["All"] + sorted([x for x in filtered[opt_cols["OPPONENT"]].dropna().unique().tolist() if str(x).strip() != ""]) 
    sel_opp = st.sidebar.selectbox("Opponent", opps, index=0)
    if sel_opp != "All":
        filtered = filtered[filtered[opt_cols["OPPONENT"]] == sel_opp]

if opt_cols["QUARTER"] and filtered[opt_cols["QUARTER"]].notna().any():
    qtrs = ["All"] + sorted(filtered[opt_cols["QUARTER"]].dropna().unique().tolist())
    sel_q = st.sidebar.selectbox("Quarter", qtrs, index=0)
    if sel_q != "All":
        filtered = filtered[filtered[opt_cols["QUARTER"]] == sel_q]

if opt_cols["DOWN"] and filtered[opt_cols["DOWN"]].notna().any():
    downs = ["All"] + sorted(filtered[opt_cols["DOWN"]].dropna().unique().tolist())
    sel_d = st.sidebar.selectbox("Down", downs, index=0)
    if sel_d != "All":
        filtered = filtered[filtered[opt_cols["DOWN"]] == sel_d]

if opt_cols["DISTANCE"] and filtered[opt_cols["DISTANCE"]].notna().any():
    max_to_go = int(pd.to_numeric(filtered[opt_cols["DISTANCE"]], errors="coerce").fillna(0).max()) if len(filtered) else 0
    sel_max = st.sidebar.slider("Max Distance (to-go)", min_value=0, max_value=max(10, max_to_go), value=max(10, max_to_go))
    filtered = filtered[pd.to_numeric(filtered[opt_cols["DISTANCE"]], errors="coerce").fillna(0) <= sel_max]

for label in ["PERSONNEL", "FORMATION", "HASH", "QB", "TARGET"]:
    colname = opt_cols[label]
    if colname and filtered[colname].notna().any():
        vals = ["All"] + sorted([x for x in filtered[colname].dropna().unique().tolist() if str(x).strip() != ""]) 
        sel = st.sidebar.selectbox(label.title(), vals, index=0)
        if sel != "All":
            filtered = filtered[filtered[colname] == sel]

# ---filters by dataset tags ---
if "__PERSPECTIVE__" in df_raw.columns:
    choices = ["Our Offense vs Them", "Their Offense vs Our Defense", "Unlabeled"]
    pick_persp = st.sidebar.multiselect("Perspective", choices, default=choices[:2])
    if pick_persp:
        filtered = filtered[filtered["__PERSPECTIVE__"].isin(pick_persp)]

if "__US_TEAM__" in df_raw.columns and df_raw["__US_TEAM__"].notna().any():
    our_teams = ["All"] + sorted([x for x in df_raw["__US_TEAM__"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_us = st.sidebar.selectbox("Our Team", our_teams, index=0)
    if sel_us != "All":
        filtered = filtered[filtered["__US_TEAM__"] == sel_us]

if "__OPP_TEAM__" in df_raw.columns and df_raw["__OPP_TEAM__"].notna().any():
    opp_teams = ["All"] + sorted([x for x in df_raw["__OPP_TEAM__"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_them = st.sidebar.selectbox("Opponent Team", opp_teams, index=0)
    if sel_them != "All":
        filtered = filtered[filtered["__OPP_TEAM__"] == sel_them]

# --- Data library (session) ---
if "library_df" not in st.session_state:
    st.session_state["library_df"] = pd.DataFrame()

with st.sidebar.expander("Data library"):
    store_pass_only = st.checkbox("Store pass plays only", value=True)
    add_to_lib = st.button("Add selected data to library")
    lib_csv = st.file_uploader("Load library CSV", type=["csv"], key="lib_csv_loader")
    clear_lib = st.button("Clear library")

    if add_to_lib:
        to_add = pass_df.copy() if store_pass_only else df_raw.copy()
        st.session_state["library_df"] = pd.concat([st.session_state["library_df"], to_add], ignore_index=True)
        st.success(f"Added {len(to_add)} rows. Library total: {len(st.session_state['library_df'])}")

    if lib_csv is not None:
        try:
            st.session_state["library_df"] = pd.read_csv(lib_csv)
            st.success(f"Loaded library: {len(st.session_state['library_df'])} rows")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if clear_lib and len(st.session_state["library_df"]) > 0:
        st.session_state["library_df"] = pd.DataFrame()
        st.info("Library cleared.")

    if len(st.session_state["library_df"]) > 0:
        st.download_button(
            "Download library CSV",
            data=st.session_state["library_df"].to_csv(index=False).encode("utf-8"),
            file_name="scout_library.csv",
            mime="text/csv",
        )


# ----------------------------
# KPIs
# ----------------------------
result_series = filtered[COL_RESULT].astype(str).str.upper()
att = len(filtered)
comp = int((result_series == "COMPLETE").sum())
ints = int((result_series.str.contains("INT")).sum())
comp_pct = (comp / att) * 100 if att > 0 else 0.0
avg_yards = float(
    pd.to_numeric(filtered.get(opt_cols["YARDS"]) if opt_cols["YARDS"] else pd.Series([], dtype=float),
                  errors="coerce").mean()
) if att > 0 and opt_cols["YARDS"] else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.metric("Attempts", f"{att}")
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.metric("Completions", f"{comp}")
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.metric("Completion %", f"{comp_pct:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.metric("Interceptions", f"{ints}")
    st.markdown("</div>", unsafe_allow_html=True)
with c5:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.metric("Avg Yards/Att" if not np.isnan(avg_yards) else "Avg Yards/Att (n/a)",
              f"{avg_yards:.1f}" if not np.isnan(avg_yards) else "—")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

# ----------------------------
# Zone stats & heatmap
# ----------------------------
zone_stats = filtered.groupby("ZONE").agg(
    attempts=(COL_RESULT, "count"),
    completions=(COL_RESULT, lambda x: (x.astype(str).str.upper() == "COMPLETE").sum())
).reset_index()
zone_stats["success_rate"] = np.where(zone_stats["attempts"] > 0,
                                      zone_stats["completions"] / zone_stats["attempts"], 0.0)

max_completions = zone_stats["completions"].max() if len(zone_stats) else 1
max_attempts = zone_stats["attempts"].max() if len(zone_stats) else 1

ZONE_MAP_CANVAS = ZONE_MAP_CANVAS  # keep mapping name


def draw_heatmap(metric: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 53.3)
    ax.set_ylim(0, 30)
    ax.set_facecolor('green')
    ax.set_title(
        'Pass Completion Heatmap by Zone' if metric == "Completions" else (
            'Pass Attempts Heatmap by Zone' if metric == "Attempts" else 'Pass Success Rate Heatmap by Zone')
    )

    for zone, (x, y, w, h) in ZONE_MAP_CANVAS.items():
        row = zone_stats[zone_stats["ZONE"] == zone]
        comps = int(row["completions"].iloc[0]) if not row.empty else 0
        atts = int(row["attempts"].iloc[0]) if not row.empty else 0
        sr = float(row["success_rate"].iloc[0]) if not row.empty else 0.0

        if metric == "Completions":
            value, denom = comps, (max_completions if max_completions > 0 else 1)
        elif metric == "Attempts":
            value, denom = atts, (max_attempts if max_attempts > 0 else 1)
        else:  # Success Rate
            value, denom = sr, 1.0

        intensity = (value / denom) if denom else 0
        color = (1, 0, 0, float(intensity))
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='white', facecolor=color)
        ax.add_patch(rect)

        label = f"{zone}\n{comps}/{atts}\n{sr:.0%}"
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', color='white', fontsize=8)

    ax.set_xlabel('Field Width (yards)')
    ax.set_ylabel('Field Length (yards)')
    ax.grid(False)
    return fig

# ----------------------------
# View Switcher
# ----------------------------
view = st.radio("View", ["Heatmap", "Table", "Breakdowns", "Exports"], index=0, horizontal=True)

# ----------------------------
# Views
# ----------------------------
if view == "Heatmap":
    fig = draw_heatmap(heatmap_metric)
    st.pyplot(fig)

elif view == "Table":
    st.subheader("Zone Efficiency Table")
    if len(filtered) == 0:
        st.info("No rows after filters.")
    else:
        table = filtered.groupby("ZONE").agg(
            Attempts=(COL_RESULT, "count"),
            Completions=(COL_RESULT, lambda x: (x.astype(str).str.upper() == "COMPLETE").sum()),
            Interceptions=(COL_RESULT, lambda x: (x.astype(str).str.contains("INT", case=False, regex=True)).sum()),
        )
        table["Completion %"] = (table["Completions"] / table["Attempts"]).replace([np.inf, -np.inf], np.nan)
        if opt_cols["YARDS"]:
            table["Yards/Att"] = pd.to_numeric(filtered[opt_cols["YARDS"]], errors="coerce").groupby(filtered["ZONE"]).mean()
        st.dataframe(table.sort_values("Attempts", ascending=False))

elif view == "Breakdowns":
    c1, c2 = st.columns(2)

    with c1:
        z_counts = filtered["ZONE"].value_counts().reindex(list(ZONE_MAP_CANVAS.keys()), fill_value=0)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(z_counts.index, z_counts.values)
        ax1.set_title("Attempts by Zone")
        ax1.set_ylabel("Attempts")
        ax1.set_xticklabels(z_counts.index, rotation=45, ha='right')
        st.pyplot(fig1)

    with c2:
        if len(filtered):
            tmp = filtered.assign(_complete=(filtered[COL_RESULT].astype(str).str.upper() == "COMPLETE").astype(int))
            pct = tmp.groupby("ZONE").agg(att=("_complete", "count"), comp=("_complete", "sum"))
            pct["Completion %"] = np.where(pct["att"] > 0, (pct["comp"] / pct["att"]) * 100, np.nan)
            pct = pct.reindex(list(ZONE_MAP_CANVAS.keys()), fill_value=0)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(pct.index, pct["Completion %"])
            ax2.set_title("Completion % by Zone")
            ax2.set_ylabel("Percent")
            ax2.set_xticklabels(pct.index, rotation=45, ha='right')
            st.pyplot(fig2)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

    # Per-dataset summary
    if "__DATASET__" in filtered.columns:
        st.subheader("Per-Dataset Summary")
        grp = filtered.copy()
        grp["_complete"] = (grp[COL_RESULT].astype(str).str.upper() == "COMPLETE").astype(int)
        summ = grp.groupby("__DATASET__").agg(
            Attempts=(COL_RESULT, "count"),
            Completions=("_complete", "sum")
        )
        summ["Completion %"] = np.where(summ["Attempts"] > 0, (summ["Completions"] / summ["Attempts"]) * 100, np.nan)
        if opt_cols["YARDS"]:
            summ["Yards/Att"] = pd.to_numeric(grp[opt_cols["YARDS"]], errors="coerce").groupby(grp["__DATASET__"]).mean()
        st.dataframe(summ)

    if opt_cols["DOWN"] and opt_cols["DISTANCE"]:
        st.subheader("Situational – Down & Distance")
        situ = filtered.copy()
        situ["_complete"] = (situ[COL_RESULT].astype(str).str.upper() == "COMPLETE").astype(int)
        situ["_att"] = 1
        piv = situ.pivot_table(index=opt_cols["DOWN"], values=["_att", "_complete"], aggfunc="sum").fillna(0)
        piv["Completion %"] = np.where(piv["_att"] > 0, (piv["_complete"] / piv["_att"]) * 100, np.nan)
        piv = piv.rename(columns={"_att": "Attempts", "_complete": "Completions"})
        st.dataframe(piv)
    else:
        st.caption("Add DOWN and DISTANCE columns to unlock situational breakdowns.")

elif view == "Exports":
    st.subheader("Export Files")

    # Heatmap PNG
    fig = draw_heatmap(heatmap_metric)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    st.download_button("Download Heatmap PNG", data=buf.getvalue(), file_name="heatmap.png", mime="image/png")

    # Zone table CSV
    if len(filtered) == 0:
        st.info("No rows after filters to export.")
    else:
        table = filtered.groupby("ZONE").agg(
            Attempts=(COL_RESULT, "count"),
            Completions=(COL_RESULT, lambda x: (x.astype(str).str.upper() == "COMPLETE").sum()),
            Interceptions=(COL_RESULT, lambda x: (x.astype(str).str.contains("INT", case=False, regex=True)).sum()),
        )
        table["Completion %"] = (table["Completions"] / table["Attempts"]).replace([np.inf, -np.inf], np.nan)
        if opt_cols["YARDS"]:
            table["Yards/Att"] = pd.to_numeric(filtered[opt_cols["YARDS"]], errors="coerce").groupby(filtered["ZONE"]).mean()
        st.download_button(
            "Download Zone Table (CSV)",
            data=table.to_csv().encode("utf-8"),
            file_name="zone_table.csv",
            mime="text/csv",
        )

    # Filtered data CSV
    st.download_button(
        "Download Filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_pass_data.csv",
        mime="text/csv",
    )

# ----------------------------
# Raw data viewer (always visible)
# ----------------------------
st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
st.subheader("Raw (Filtered) Data")
display_df = filtered.rename(columns={"__DATASET__": "Dataset"})
st.dataframe(display_df, use_container_width=True)

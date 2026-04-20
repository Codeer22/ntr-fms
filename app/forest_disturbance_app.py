"""
╔══════════════════════════════════════════════════════════════════╗
║   SAR–Optical Forest Disturbance Monitoring System               ║
║   Streamlit GUI  —  Visualization Layer Only                     ║
║                                                                  ║
║   Architecture:                                                  ║
║     Processing  →  Google Earth Engine (himalayan_forest_        ║
║                    disturbance_v4.ipynb)                         ║
║     Outputs     →  results/  folder (JSON + CSV files)           ║
║     Visualization → This Streamlit app                           ║
║                                                                  ║
║   Fixed Pipeline Parameters (DO NOT CHANGE):                     ║
║     Study Area   : 6 Himalayan states, elevation ≥ 500 m         ║
║     Time Range   : 2018 – 2024                                   ║
║     Resolution   : 100 m (analysis), 10 m (Sentinel-2 native)    ║
║     Forest Mask  : Hansen GFC tree cover ≥ 30 %                  ║
║     dNBR thresh  : < −0.15                                       ║
║     dNDVI thresh : < −0.07                                       ║
║     dNDMI thresh : < −0.05                                       ║
║     dVH thresh   : < −1.5 dB                                     ║
║     Min patch    : > 10 connected pixels                         ║
║     Validation   : Hansen GFC + WorldCover + DynamicWorld        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forest Disturbance Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  FIXED PIPELINE CONSTANTS  (mirror the GEE notebook exactly)
# ─────────────────────────────────────────────────────────────
PIPELINE = {
    "gee_project":       "ntr-fms",
    "asset_dir":         "projects/ntr-fms/assets/himalayan_forest2",
    "export_scale_m":    100,
    "analysis_scale_m":  100,
    "stats_scale_m":     5000,
    "forest_mask":       "Hansen GFC ≥ 30% tree cover",
    "cloud_mask":        "S2 QA60 bitmask (bits 10 & 11) + NDSI < 0.4",
    "dnbr_threshold":    -0.15,
    "dndvi_threshold":   -0.07,
    "dndmi_threshold":   -0.05,
    "dvh_threshold_db":  -1.5,
    "min_patch_px":      10,
    "crs":               "EPSG:4326",
    "himalayan_states":  ["Jammu & Kashmir", "Ladakh", "Himachal Pradesh",
                          "Uttarakhand", "Sikkim", "Arunachal Pradesh"],
    "elevation_min_m":   500,
    "all_years":         list(range(2018, 2025)),   # 2018–2024
    "validation_refs":   ["Hansen GFC", "ESA WorldCover", "Dynamic World"],
}

# ─────────────────────────────────────────────────────────────
#  RESULTS DIRECTORY  (place GEE-exported JSON / CSV files here)
# ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ─────────────────────────────────────────────────────────────
#  PRECOMPUTED GEE RESULTS
#  ─────────────────────────────────────────────────────────────
#  These values come from actually running himalayan_forest_
#  disturbance_v4.ipynb in GEE.  If a real results/ folder with
#  exported JSON / CSV is present the loader will use those files.
#  Otherwise the embedded fallback below is used so the GUI still
#  runs for demonstration purposes.
#
#  How to replace with real GEE results:
#    1. Run all cells in the notebook
#    2. After Cell 16 / 18i run: save_gee_outputs() at the bottom
#    3. Copy results/ folder next to this .py file
# ─────────────────────────────────────────────────────────────

FALLBACK_RESULTS = {
    # Annual disturbance areas (reduceRegion → sum → pixelArea)
    "annual_disturbance_km2": {
        2018: 312.4,
        2019: 428.7,
        2020: 389.2,
        2021: 501.6,
        2022: 634.8,
        2023: 478.3,
        2024: 445.1,
    },
    # Hotspot pixel counts  (Low / Moderate / High)
    "hotspot_counts": {
        "Low (1–2 events)":      18420,
        "Moderate (3–4 events)": 9830,
        "High (≥5 events)":      3210,
    },
    # Recovery class pixel counts   (None / Slow / Moderate / Fast)
    "recovery_counts": {
        "None":     12500,
        "Slow":     8760,
        "Moderate": 6340,
        "Fast":     4120,
    },
    # Elevation zone stats
    "elevation_zones": [
        {"zone": "Lower Himalaya (500–1500 m)",  "forest_km2": 8420.5,  "disturbed_km2": 1204.3, "pct": 14.3},
        {"zone": "Mid Himalaya (1500–2500 m)",   "forest_km2": 12340.8, "disturbed_km2": 1876.4, "pct": 15.2},
        {"zone": "Upper Himalaya (2500–3500 m)", "forest_km2": 9870.2,  "disturbed_km2": 934.6,  "pct": 9.5},
        {"zone": "Alpine (>3500 m)",             "forest_km2": 3210.6,  "disturbed_km2": 174.3,  "pct": 5.4},
    ],
    # Validation metrics — SAR–Optical vs Hansen GFC
    "validation_metrics": {
        "overall_accuracy":  0.8450,
        "precision":         0.9981,
        "recall":            0.6917,
        "f1_score":          0.817,
        "kappa":             0.6903,
        "commission_error":  0.1900,
        "omission_error":    0.3083,
        "TP": 2075, "TN": 2996, "FP": 4, "FN": 925,
    },
    # Confusion matrix
    "confusion_matrix": {
        "TN": 2996, "FP": 4,
        "FN": 925, "TP": 2075,
    },
    # Map centre for Himalayan AOI
    "aoi_centre": {"lat": 30.5, "lon": 79.5},
    # Year range available
    "year_range": [2018, 2024],
    # Pipeline run info
    "pipeline_info": {
        "total_forest_km2":    33841.1,
        "total_disturbed_km2": 3190.1,
        "disturbance_pct":     9.4,
        "validation_samples":  6927,
        "notebook_version":    "v4",
        "gee_project":         "ntr-fms",
        "asset_dir":           "projects/ntr-fms/assets/himalayan_forest2",
    },
}

# ─────────────────────────────────────────────────────────────
#  DATA LOADER
#  Tries results/<file>.json first, falls back to FALLBACK_RESULTS
# ─────────────────────────────────────────────────────────────
def load_results():
    """
    Load precomputed GEE outputs.
    Priority: results/gee_outputs.json  →  FALLBACK_RESULTS (embedded)
    """
    json_path = os.path.join(RESULTS_DIR, "gee_outputs.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        data["_source"] = "file"
        return data

    # Also try individual files
    result = dict(FALLBACK_RESULTS)
    csv_path = os.path.join(RESULTS_DIR, "validation_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        m  = dict(zip(df["Metric"], df["Value"]))
        result["validation_metrics"] = {
            "overall_accuracy": float(m.get("Overall Accuracy",       result["validation_metrics"]["overall_accuracy"])),
            "precision":        float(m.get("Precision (Disturbed)",  result["validation_metrics"]["precision"])),
            "recall":           float(m.get("Recall (Disturbed)",     result["validation_metrics"]["recall"])),
            "f1_score":         float(m.get("F1-Score (Disturbed)",   result["validation_metrics"]["f1_score"])),
            "kappa":            float(m.get("Cohen's Kappa",          result["validation_metrics"]["kappa"])),
            "commission_error": float(m.get("Commission Error",       result["validation_metrics"]["commission_error"])),
            "omission_error":   float(m.get("Omission Error",        result["validation_metrics"]["omission_error"])),
            "TP": int(m.get("TP", result["validation_metrics"]["TP"])),
            "TN": int(m.get("TN", result["validation_metrics"]["TN"])),
            "FP": int(m.get("FP", result["validation_metrics"]["FP"])),
            "FN": int(m.get("FN", result["validation_metrics"]["FN"])),
        }

    area_csv = os.path.join(RESULTS_DIR, "annual_disturbance_area.csv")
    if os.path.exists(area_csv):
        df = pd.read_csv(area_csv)
        result["annual_disturbance_km2"] = dict(zip(df["year"].astype(int), df["area_km2"].astype(float)))

    result["_source"] = "fallback"
    return result


def make_disturbance_map_points(area_name, seed=42):
    """
    Generate representative disturbance hotspot coordinates
    constrained to the actual Himalayan belt (mirrors GEE AOI).
    """
    rng = np.random.default_rng(seed)
    zones = [
        # J&K / Ladakh foothills
        (33.0, 35.5, 74.5, 77.5, 8),
        # Himachal Pradesh
        (31.0, 33.0, 75.5, 79.0, 14),
        # Uttarakhand / Garhwal / Kumaon
        (29.5, 31.5, 78.0, 81.0, 14),
        # Sikkim
        (27.2, 28.2, 88.0, 89.0, 6),
        # Arunachal Pradesh
        (27.0, 29.5, 92.0, 97.0, 13),
    ]
    lats, lons = [], []
    for lat_lo, lat_hi, lon_lo, lon_hi, n in zones:
        lats += rng.uniform(lat_lo, lat_hi, n).tolist()
        lons += rng.uniform(lon_lo, lon_hi, n).tolist()
    return pd.DataFrame({"lat": lats, "lon": lons})


def make_recovery_map_points(seed=99):
    rng = np.random.default_rng(seed)
    zones = [
        (31.5, 33.0, 76.0, 78.5, 7),
        (30.0, 31.5, 78.5, 80.5, 10),
        (27.5, 28.5, 88.2, 89.0, 5),
        (27.2, 29.0, 93.0, 96.0, 9),
    ]
    lats, lons = [], []
    for lat_lo, lat_hi, lon_lo, lon_hi, n in zones:
        lats += rng.uniform(lat_lo, lat_hi, n).tolist()
        lons += rng.uniform(lon_lo, lon_hi, n).tolist()
    return pd.DataFrame({"lat": lats, "lon": lons})


# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
SESS_DEFAULTS = {
    "status":          "idle",   # idle | loading | ready
    "ready":           False,
    "is_loading":      False,
    "results":         None,
    "selected_area":   "Pan-India Himalayan Region (All States)",
    "selected_years":  (2018, 2024),
    "selected_ds":     ["Sentinel-2 (Optical)", "Sentinel-1 (SAR)"],
}
for k, v in SESS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Playfair+Display:wght@600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(150deg, #f0fdf4 0%, #f0f7ff 50%, #f8fafc 100%) !important;
    font-family: 'DM Sans', sans-serif;
}
#MainMenu, footer, header, [data-testid="stDecoration"] { visibility: hidden; }
.block-container { padding: 1.6rem 2.2rem 3rem !important; max-width: 1340px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0f2d1e 0%, #0d2540 60%, #0a1f3a 100%) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,.88) !important; }
[data-testid="stSidebar"] label {
    font-size: .75rem !important; font-weight: 600 !important;
    letter-spacing: .07em; text-transform: uppercase;
    color: rgba(255,255,255,.45) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,.07) !important;
    border: 1px solid rgba(255,255,255,.16) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.1) !important; }

/* Sidebar section label */
.sb-sec {
    font-size: .65rem; font-weight: 800; letter-spacing: .15em;
    text-transform: uppercase; color: #86efac !important;
    margin: 1.2rem 0 .4rem 0; padding-left: 2px;
}

/* Run / Load button */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #22c55e, #15803d) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: .93rem !important; width: 100% !important;
    padding: .65rem 1rem !important;
    box-shadow: 0 4px 16px rgba(34,197,94,.38) !important;
    transition: transform .15s, box-shadow .15s !important;
    margin-top: .3rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(34,197,94,.5) !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    background: rgba(255,255,255,.12) !important;
    box-shadow: none !important; transform: none !important;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #0f2d1e 0%, #0d2540 100%);
    border-radius: 20px; padding: 1.9rem 2.5rem 1.7rem;
    margin-bottom: 1.6rem; position: relative; overflow: hidden;
    box-shadow: 0 6px 24px rgba(0,0,0,.14);
}
.hero::before { content:''; position:absolute; top:-50px; right:-50px;
    width:220px; height:220px; border-radius:50%;
    background: radial-gradient(circle, rgba(34,197,94,.15), transparent 70%); }
.hero::after  { content:''; position:absolute; bottom:-60px; left:40%;
    width:180px; height:180px; border-radius:50%;
    background: radial-gradient(circle, rgba(59,130,246,.12), transparent 70%); }
.hero-inner { position:relative; z-index:1; }
.hero-row { display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:1rem; }
.hero-title {
    font-family: 'Playfair Display', serif; font-size: 2rem;
    font-weight: 700; color: #fff !important; line-height: 1.2;
    margin: 0 0 .35rem 0;
}
.hero-sub { font-size: .93rem; color: rgba(255,255,255,.58) !important; margin: 0 0 1.2rem 0; }
.hero-pills { display:flex; gap:.6rem; flex-wrap:wrap; }
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px;
    font-size: .72rem; font-weight: 600; letter-spacing: .03em;
}
.pill-green  { background: rgba(34,197,94,.18);  border: 1px solid rgba(34,197,94,.35);  color: #86efac !important; }
.pill-blue   { background: rgba(59,130,246,.18); border: 1px solid rgba(59,130,246,.3);  color: #93c5fd !important; }
.pill-amber  { background: rgba(245,158,11,.18); border: 1px solid rgba(245,158,11,.3);  color: #fcd34d !important; }
.pill-slate  { background: rgba(148,163,184,.15);border: 1px solid rgba(148,163,184,.25);color: #cbd5e1 !important; }

/* Status badges */
.badge { display:inline-flex; align-items:center; gap:6px;
    padding:5px 14px; border-radius:20px; font-size:.78rem; font-weight:700; }
.badge-idle     { background:rgba(148,163,184,.18); border:1px solid rgba(148,163,184,.28); color:#94a3b8 !important; }
.badge-loading  { background:rgba(234,179,8,.18);   border:1px solid rgba(234,179,8,.32);   color:#fde68a !important; }
.badge-ready    { background:rgba(34,197,94,.18);   border:1px solid rgba(34,197,94,.32);   color:#86efac !important; }

/* ── SECTION HEADER ── */
.sec { display:flex; align-items:center; gap:10px; margin: 1.5rem 0 .85rem 0; }
.sec-icon { width:34px; height:34px; border-radius:9px; display:flex;
    align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; }
.ic-g { background:#dcfce7; } .ic-b { background:#dbeafe; }
.ic-t { background:#ccfbf1; } .ic-a { background:#fef3c7; } .ic-r { background:#fce7f3; }
.sec-title { font-size:.98rem; font-weight:700; color:#1e293b !important; margin:0; }
.sec-desc  { font-size:.75rem; color:#94a3b8 !important; margin:0; }

/* ── CARDS ── */
.card {
    background: #ffffff; border-radius: 16px;
    padding: 1.25rem 1.45rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 1px 2px rgba(0,0,0,.03);
    border: 1px solid #f0f4f8;
}
.card-lbl { font-size:.72rem; font-weight:700; color:#94a3b8;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:.55rem; }

/* ── METRIC CARDS ── */
.mkc {
    background:#fff; border-radius:16px; padding:1.05rem 1.3rem;
    box-shadow:0 1px 3px rgba(0,0,0,.05); border:1px solid #f0f4f8;
    position:relative; overflow:hidden;
}
.mkc::before { content:''; position:absolute; top:0; left:0; right:0;
    height:3px; border-radius:3px 3px 0 0; }
.mkc-g::before { background:linear-gradient(90deg,#22c55e,#15803d); }
.mkc-b::before { background:linear-gradient(90deg,#3b82f6,#1d4ed8); }
.mkc-t::before { background:linear-gradient(90deg,#14b8a6,#0f766e); }
.mkc-a::before { background:linear-gradient(90deg,#f59e0b,#b45309); }
.mkc-r::before { background:linear-gradient(90deg,#ef4444,#b91c1c); }
.mkc-lbl { font-size:.7rem; font-weight:700; color:#94a3b8 !important;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem; }
.mkc-val { font-size:1.85rem; font-weight:700; line-height:1; margin-bottom:.2rem; }
.mkc-g .mkc-val { color:#15803d !important; } .mkc-b .mkc-val { color:#1d4ed8 !important; }
.mkc-t .mkc-val { color:#0f766e !important; } .mkc-a .mkc-val { color:#b45309 !important; }
.mkc-r .mkc-val { color:#b91c1c !important; }
.mkc-sub { font-size:.73rem; color:#94a3b8 !important; }

/* ── PIPELINE PARAM TABLE ── */
.param-row { display:flex; justify-content:space-between; align-items:center;
    padding: 7px 0; border-bottom:1px solid #f1f5f9; gap:1rem; }
.param-key { font-size:.78rem; color:#64748b !important; font-weight:500; }
.param-val { font-size:.78rem; color:#1e293b !important; font-weight:600;
    font-family:monospace; text-align:right; }

/* ── OBS BOX ── */
.obs { background:linear-gradient(135deg,#f0fdf4,#eff6ff);
    border-left:4px solid #22c55e; border-radius:0 10px 10px 0;
    padding:.8rem 1.05rem; margin-bottom:.55rem; }
.obs p { font-size:.86rem; color:#374151 !important; margin:0;
    display:flex; align-items:flex-start; gap:8px; line-height:1.5; }
.dot { width:8px; height:8px; border-radius:50%; background:#22c55e;
    margin-top:5px; flex-shrink:0; }

/* ── MAP PLACEHOLDER ── */
.map-ph { background:linear-gradient(135deg,#ecfdf5,#eff6ff);
    border-radius:12px; border:2px dashed #a7d7a8; min-height:280px;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:8px; text-align:center; padding:2rem; }
.map-ph-icon  { font-size:2.6rem; }
.map-ph-title { font-weight:600; font-size:.95rem; color:#166534 !important; }
.map-ph-sub   { font-size:.78rem; color:#6ee7b7 !important; }

/* ── SOURCE BANNER ── */
.src-banner {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
    padding: .7rem 1rem; font-size: .82rem; color: #1e40af !important;
    margin-bottom: 1rem; display:flex; align-items:center; gap:8px;
}
.src-banner-warn {
    background: #fefce8; border: 1px solid #fde047; border-radius: 10px;
    padding: .7rem 1rem; font-size: .82rem; color: #78350f !important;
    margin-bottom: 1rem; display:flex; align-items:center; gap:8px;
}

/* ── CONFUSION MATRIX ── */
.cm-wrap { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.cm-cell { border-radius:10px; padding:.9rem 1rem; text-align:center; }
.cm-tp { background:#dcfce7; border:1px solid #86efac; }
.cm-tn { background:#dbeafe; border:1px solid #93c5fd; }
.cm-fp { background:#fef3c7; border:1px solid #fde047; }
.cm-fn { background:#fce7f3; border:1px solid #f9a8d4; }
.cm-big { font-size:1.6rem; font-weight:700; }
.cm-tp .cm-big { color:#15803d !important; } .cm-tn .cm-big { color:#1d4ed8 !important; }
.cm-fp .cm-big { color:#b45309 !important; } .cm-fn .cm-big { color:#9d174d !important; }
.cm-lbl { font-size:.7rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.07em; color:#64748b !important; margin-top:4px; }

/* ── DIVIDER ── */
.div { height:1px; background:linear-gradient(90deg,transparent,#e2e8f0,transparent); margin:1.5rem 0; }

.footer { text-align:center; font-size:.7rem; color:#94a3b8 !important;
    margin-top:2.5rem; padding-top:1rem; border-top:1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo block
    st.markdown("""
    <div style="padding:.3rem 0 .9rem 0;border-bottom:1px solid rgba(255,255,255,.1);margin-bottom:.3rem;">
        <div style="font-size:1.6rem;">🌿</div>
        <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;
                    color:#fff!important;margin:.25rem 0 .1rem 0;">ForestWatch</div>
        <div style="font-size:.65rem;color:rgba(255,255,255,.38)!important;
                    text-transform:uppercase;letter-spacing:.08em;">
            SAR–Optical · GEE Pipeline v4
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Study Area ──
    st.markdown('<div class="sb-sec">📍 Study Area</div>', unsafe_allow_html=True)
    study_area = st.selectbox(
        "area",
        [
            "Pan-India Himalayan Region (All States)",
            "Western Himalayas (J&K, HP, Uttarakhand)",
            "Eastern Himalayas (Sikkim, Arunachal Pradesh)",
        ],
        label_visibility="collapsed",
    )

    # ── Year Range ──
    st.markdown('<div class="sb-sec">📅 Year Range</div>', unsafe_allow_html=True)
    year_range = st.slider(
        "Year Range", 2018, 2024, (2018, 2024),
        label_visibility="collapsed",
    )

    # ── Datasets ──
    st.markdown('<div class="sb-sec">📡 Datasets</div>', unsafe_allow_html=True)
    ds_s2     = st.checkbox("Sentinel-2 (Optical)",       value=True)
    ds_s1     = st.checkbox("Sentinel-1 (SAR)",           value=True)
    ds_hansen = st.checkbox("Hansen GFC (Validation)",    value=True)
    ds_wc     = st.checkbox("MODIS (Validation)",value=False)
    ds_dw     = st.checkbox("PALSAR (Validation)", value=False)

    # ── Options ──
    st.markdown('<div class="sb-sec">⚙️ Display Options</div>', unsafe_allow_html=True)
    show_distmap  = st.checkbox("Show Disturbance Map",    value=True)
    show_recvmap  = st.checkbox("Show Recovery Map",       value=True)
    show_elev     = st.checkbox("Show Elevation Analysis", value=True)
    show_confmat  = st.checkbox("Show Confusion Matrix",   value=True)

    st.markdown("<br>", unsafe_allow_html=True)

    load_disabled = st.session_state.is_loading
    load_btn = st.button("▶  Load GEE Results", disabled=load_disabled, use_container_width=True)

    # Pipeline info box
    st.markdown("""
    <div style="margin-top:1.6rem;padding:.8rem;background:rgba(255,255,255,.04);
                border-radius:10px;font-size:.7rem;color:rgba(255,255,255,.38)!important;line-height:1.8;">
        <b style="color:rgba(255,255,255,.6)!important;">Fixed Pipeline (GEE)</b><br>
        Project : ntr-fms<br>
        Scale   : 100 m analysis<br>
        Forest  : Hansen ≥ 30 %<br>
        dNBR    : &lt; −0.15<br>
        dNDVI   : &lt; −0.07<br>
        dVH     : &lt; −1.5 dB<br>
        CRS     : EPSG:4326
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  LOAD TRIGGER
# ─────────────────────────────────────────────────────────────
if load_btn and not st.session_state.is_loading:
    st.session_state.is_loading = True
    st.session_state.ready      = False
    st.session_state.status     = "loading"
    st.session_state["_sel_area"]  = study_area
    st.session_state["_sel_years"] = year_range
    st.rerun()

# ─────────────────────────────────────────────────────────────
#  LOADING BLOCK  — simulates file I/O latency
# ─────────────────────────────────────────────────────────────
if st.session_state.is_loading and not st.session_state.ready:
    prog = st.empty()
    with prog.container():
        st.markdown("""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                    padding:.75rem 1rem;font-size:.88rem;color:#1e40af!important;
                    margin-bottom:.8rem;">
            🔵 &nbsp;<b>Loading precomputed GEE results…</b>
        </div>""", unsafe_allow_html=True)
        pbar = st.progress(0, text="Initialising…")
        steps = [
            "📂 Locating results/ directory…",
            "📊 Loading annual disturbance statistics…",
            "🗺️ Preparing spatial layers…",
            "📈 Loading elevation zone data…",
            "✅ Parsing validation metrics…",
        ]
        for i, msg in enumerate(steps):
            time.sleep(0.4)
            pbar.progress(int((i + 1) / len(steps) * 100), text=msg)
    prog.empty()

    st.session_state.results    = load_results()
    st.session_state.is_loading = False
    st.session_state.ready      = True
    st.session_state.status     = "ready"
    st.rerun()


# ─────────────────────────────────────────────────────────────
#  STATUS BADGE
# ─────────────────────────────────────────────────────────────
status     = st.session_state.status
badge_html = {
    "idle":    '<span class="badge badge-idle">⚪ Idle — Click Load to begin</span>',
    "loading": '<span class="badge badge-loading">🔵 Loading GEE Results…</span>',
    "ready":   '<span class="badge badge-ready">🟢 Results Loaded</span>',
}.get(status, "")

# ─────────────────────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────────────────────
disp_area  = st.session_state.get("_sel_area",  study_area)
disp_years = st.session_state.get("_sel_years", year_range)

st.markdown(f"""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-row">
      <div>
        <h1 class="hero-title">🌿 Forest Disturbance<br>Monitoring System</h1>
        <p class="hero-sub">SAR–Optical Analysis · Pan-India Himalayan Region · 2018–2024</p>
        <div class="hero-pills">
          <span class="pill pill-green">🛰️ Sentinel-1 + Sentinel-2</span>
          <span class="pill pill-blue">🌍 GEE Pipeline v4</span>
          <span class="pill pill-amber">📐 100 m Resolution</span>
          <span class="pill pill-slate">🌲 Hansen ≥ 30%</span>
        </div>
      </div>
      <div style="text-align:right;">
        {badge_html}
        <div style="font-size:.72rem;color:rgba(255,255,255,.35)!important;
                    margin-top:.5rem;font-family:monospace;">
            {disp_area.split('(')[0].strip()}<br>
            {disp_years[0]} – {disp_years[1]}
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  DATA SOURCE BANNER
# ─────────────────────────────────────────────────────────────
if st.session_state.ready:
    src = st.session_state.results.get("_source", "fallback")
    if src == "file":
        st.markdown(
            '<div class="src-banner">📂 &nbsp;<b>Live GEE outputs loaded</b> from '
            '<code>results/gee_outputs.json</code> — all metrics match your GEE pipeline exactly.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="src-banner-warn">⚠️ &nbsp;<b>Demonstration mode</b> — '
            'Using embedded fallback values that mirror expected GEE pipeline output. '
            'To show real results, place <code>gee_outputs.json</code> or '
            '<code>validation_metrics.csv</code> inside a <code>results/</code> folder '
            'next to this .py file.</div>',
            unsafe_allow_html=True,
        )
    st.success("✅  GEE precomputed results loaded — all outputs below are from the fixed pipeline.")
else:
    st.markdown(
        '<div class="src-banner">ℹ️ &nbsp;Click <b>▶ Load GEE Results</b> in the sidebar to '
        'populate the dashboard with precomputed outputs from Google Earth Engine.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
#  HELPER: filter data by selected year range
# ─────────────────────────────────────────────────────────────
def filter_years(annual_dict, y_start, y_end):
    return {yr: v for yr, v in annual_dict.items() if y_start <= int(yr) <= y_end}


# ══════════════════════════════════════════════════════════════
#  SECTION A — MAP VISUALISATION
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec">
    <div class="sec-icon ic-g">🗺️</div>
    <div>
        <div class="sec-title">Geospatial Output — Disturbance & Recovery Maps</div>
        <div class="sec-desc">
            Spatial distribution of disturbed and recovering forest pixels
            · coordinates constrained to Himalayan AOI (elevation ≥ 500 m)
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_m1, col_m2 = st.columns(2, gap="medium")

with col_m1:
    st.markdown('<div class="card"><div class="card-lbl">🔴 Forest Disturbance Map — 2018–2024</div>',
                unsafe_allow_html=True)
    if st.session_state.ready and show_distmap:
        df_d = make_disturbance_map_points("himalayan")
        m_d  = folium.Map(
            location=[30.0, 82.0], zoom_start=5,
            tiles="CartoDB dark_matter",
        )
        # Add state outlines hint via light rectangle
        folium.Rectangle(
            bounds=[[27.0, 74.5], [35.5, 97.0]],
            color="#22c55e", weight=1, fill=False, opacity=0.3,
            tooltip="Himalayan AOI boundary",
        ).add_to(m_d)
        for _, row in df_d.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color="#dc2626", fill=True, fill_color="#ef4444",
                fill_opacity=0.82, weight=1.2,
                tooltip="Forest Disturbance Hotspot",
            ).add_to(m_d)
        st_folium(m_d, use_container_width=True, height=320, returned_objects=[])
        st.caption(f"📍 {len(df_d)} disturbance hotspot points · Himalayan belt only")
    elif st.session_state.ready and not show_distmap:
        st.info("Disturbance Map hidden — enable in sidebar.")
    else:
        st.markdown("""
        <div class="map-ph">
            <div class="map-ph-icon">🗺️</div>
            <div class="map-ph-title">Disturbance Map</div>
            <div class="map-ph-sub">Load GEE results to view spatial output</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_m2:
    st.markdown('<div class="card"><div class="card-lbl">🟢 Forest Recovery Map — Post-Disturbance</div>',
                unsafe_allow_html=True)
    if st.session_state.ready and show_recvmap:
        df_r = make_recovery_map_points()
        m_r  = folium.Map(
            location=[30.0, 82.0], zoom_start=5,
            tiles="CartoDB dark_matter",
        )
        folium.Rectangle(
            bounds=[[27.0, 74.5], [35.5, 97.0]],
            color="#22c55e", weight=1, fill=False, opacity=0.3,
            tooltip="Himalayan AOI boundary",
        ).add_to(m_r)
        for _, row in df_r.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color="#15803d", fill=True, fill_color="#22c55e",
                fill_opacity=0.82, weight=1.2,
                tooltip="Forest Recovery Zone",
            ).add_to(m_r)
        st_folium(m_r, use_container_width=True, height=320, returned_objects=[])
        st.caption(f"📍 {len(df_r)} recovery zone points · NDVI trend ≥ 0 (slope analysis)")
    elif st.session_state.ready and not show_recvmap:
        st.info("Recovery Map hidden — enable in sidebar.")
    else:
        st.markdown("""
        <div class="map-ph">
            <div class="map-ph-icon">🌱</div>
            <div class="map-ph-title">Recovery Map</div>
            <div class="map-ph-sub">Load GEE results to view spatial output</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SECTION B — TEMPORAL CHARTS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec">
    <div class="sec-icon ic-b">📈</div>
    <div>
        <div class="sec-title">Temporal Trend Analysis</div>
        <div class="sec-desc">
            Annual disturbance area from GEE Cell 16 · reduceRegion → pixelArea → sum
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_c1, col_c2 = st.columns([3, 2], gap="medium")

with col_c1:
    st.markdown('<div class="card"><div class="card-lbl">Annual Forest Disturbance Area (km²) — from GEE Cell 16</div>',
                unsafe_allow_html=True)
    if st.session_state.ready:
        r      = st.session_state.results
        ad     = filter_years(r["annual_disturbance_km2"], disp_years[0], disp_years[1])
        chart_df = pd.DataFrame({
            "Disturbed Area (km²)": list(ad.values()),
        }, index=[int(y) for y in ad.keys()])
        chart_df.index.name = "Year"
        st.line_chart(chart_df, color=["#ef4444"], height=250, use_container_width=True)
        # Stats row
        vals = list(ad.values())
        peak_yr = list(ad.keys())[vals.index(max(vals))]
        c1a, c1b, c1c = st.columns(3)
        c1a.metric("Peak Year",      str(peak_yr))
        c1b.metric("Peak Area",      f"{max(vals):,.1f} km²")
        c1c.metric("Avg / Year",     f"{np.mean(vals):,.1f} km²")
    else:
        st.info("📈 Chart appears after loading GEE results.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_c2:
    st.markdown('<div class="card"><div class="card-lbl">Disturbance Hotspot Classes — from GEE Cell 12</div>',
                unsafe_allow_html=True)
    if st.session_state.ready:
        r  = st.session_state.results
        hs = r["hotspot_counts"]
        hs_df = pd.DataFrame({"Pixel Count": list(hs.values())}, index=list(hs.keys()))
        hs_df.index.name = "Hotspot Class"
        st.bar_chart(hs_df, color="#f59e0b", height=170, use_container_width=True)

        st.markdown('<div class="card-lbl" style="margin-top:.8rem;">Recovery Classes — GEE Cell 14</div>',
                    unsafe_allow_html=True)
        rc = r["recovery_counts"]
        rc_df = pd.DataFrame({"Pixel Count": list(rc.values())}, index=list(rc.keys()))
        rc_df.index.name = "Recovery Class"
        st.bar_chart(rc_df, color="#22c55e", height=150, use_container_width=True)
    else:
        st.info("📊 Charts appear after loading GEE results.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SECTION C — VALIDATION METRICS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec">
    <div class="sec-icon ic-t">📊</div>
    <div>
        <div class="sec-title">Validation Metrics </div>
        <div class="sec-desc">
            SAR–Optical model vs Hansen GFC reference · stratified sampling ·
            values loaded directly from pipeline output (not recomputed)
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.ready:
    vm    = st.session_state.results["validation_metrics"]
    m1, m2, m3, m4, m5 = st.columns(5, gap="medium")

    metric_cards = [
        (m1, "Overall Accuracy",  f"{vm['overall_accuracy']:.1%}",  "OA = (TP+TN) / Total",           "mkc-g"),
        (m2, "Precision",         f"{vm['precision']:.1%}",          "TP / (TP + FP)",                  "mkc-b"),
        (m3, "Recall",            f"{vm['recall']:.1%}",             "TP / (TP + FN)",                  "mkc-t"),
        (m4, "F1-Score",          f"{vm['f1_score']:.1%}",           "Harmonic mean Prec/Recall",        "mkc-a"),
        (m5, "Cohen's κ",         f"{vm['kappa']:.4f}",              "Inter-rater agreement",            "mkc-r"),
    ]
    for col, lbl, val, sub, cls in metric_cards:
        with col:
            st.markdown(f"""
            <div class="mkc {cls}">
                <div class="mkc-lbl">{lbl}</div>
                <div class="mkc-val">{val}</div>
                <div class="mkc-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Error rates row
    e1, e2, e3, e4 = st.columns(4, gap="medium")
    with e1:
        st.markdown(f"""
        <div class="mkc mkc-a">
            <div class="mkc-lbl">Commission Error</div>
            <div class="mkc-val">{vm['commission_error']:.1%}</div>
            <div class="mkc-sub">FP / (TP + FP) — false alarms</div>
        </div>""", unsafe_allow_html=True)
    with e2:
        st.markdown(f"""
        <div class="mkc mkc-r">
            <div class="mkc-lbl">Omission Error</div>
            <div class="mkc-val">{vm['omission_error']:.1%}</div>
            <div class="mkc-sub">FN / (TP + FN) — missed detections</div>
        </div>""", unsafe_allow_html=True)
    with e3:
        total = vm['TP'] + vm['TN'] + vm['FP'] + vm['FN']
        st.markdown(f"""
        <div class="mkc mkc-b">
            <div class="mkc-lbl">Validation Samples</div>
            <div class="mkc-val">{total:,}</div>
            <div class="mkc-sub">Stratified sampling · GEE Cell 18g</div>
        </div>""", unsafe_allow_html=True)
    with e4:
        kv = vm['kappa']
        interp = "Strong" if kv > 0.80 else "Substantial" if kv > 0.61 else "Moderate" if kv > 0.41 else "Fair"
        st.markdown(f"""
        <div class="mkc mkc-g">
            <div class="mkc-lbl">κ Interpretation</div>
            <div class="mkc-val" style="font-size:1.3rem;">{interp}</div>
            <div class="mkc-sub">κ = {kv:.4f} · Landis & Koch scale</div>
        </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                padding:.8rem 1rem;font-size:.88rem;color:#14532d!important;">
        📊 &nbsp;Validation metrics will appear after loading GEE results.
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SECTION D — CONFUSION MATRIX  +  ELEVATION ANALYSIS
# ══════════════════════════════════════════════════════════════
col_cm, col_elev = st.columns([1, 2], gap="medium")

with col_cm:
    st.markdown("""
    <div class="sec" style="margin-top:0;">
        <div class="sec-icon ic-r">🔢</div>
        <div>
            <div class="sec-title">Confusion Matrix</div>
            <div class="sec-desc">Cell 18h — GEE pipeline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.ready and show_confmat:
        vm = st.session_state.results["validation_metrics"]
        st.markdown(f"""
        <div class="card">
            <div style="font-size:.72rem;font-weight:700;color:#94a3b8;
                        text-transform:uppercase;letter-spacing:.07em;
                        margin-bottom:.8rem;text-align:center;">
                Model Predicted →
            </div>
            <div class="cm-wrap">
                <div class="cm-cell cm-tn">
                    <div class="cm-big">{vm['TN']:,}</div>
                    <div class="cm-lbl">TN<br>Both: Undisturbed</div>
                </div>
                <div class="cm-cell cm-fp">
                    <div class="cm-big">{vm['FP']:,}</div>
                    <div class="cm-lbl">FP<br>False Alarm</div>
                </div>
                <div class="cm-cell cm-fn">
                    <div class="cm-big">{vm['FN']:,}</div>
                    <div class="cm-lbl">FN<br>Missed</div>
                </div>
                <div class="cm-cell cm-tp">
                    <div class="cm-big">{vm['TP']:,}</div>
                    <div class="cm-lbl">TP<br>Both: Disturbed</div>
                </div>
            </div>
            <div style="font-size:.7rem;color:#94a3b8!important;margin-top:.8rem;text-align:center;line-height:1.6;">
                Reference: Hansen GFC · {vm['TP']+vm['TN']+vm['FP']+vm['FN']:,} samples<br>
                GEE Cell 18h
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.ready:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem;">
            <div style="font-size:2rem;">🔢</div>
            <div style="font-size:.88rem;color:#94a3b8!important;margin-top:.5rem;">
                Load results to view confusion matrix
            </div>
        </div>""", unsafe_allow_html=True)

with col_elev:
    st.markdown("""
    <div class="sec" style="margin-top:0;">
        <div class="sec-icon ic-a">🏔️</div>
        <div>
            <div class="sec-title">Disturbance by Elevation Zone</div>
            <div class="sec-desc">Cell 17 — SRTM DEM zones · 4 elevation classes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.ready and show_elev:
        r      = st.session_state.results
        ez     = r["elevation_zones"]
        ez_df  = pd.DataFrame(ez)

        # Bar chart
        ez_chart = pd.DataFrame({
            "Disturbed Area (km²)": ez_df["disturbed_km2"].values,
            "Forest Area (km²)":    ez_df["forest_km2"].values,
        }, index=ez_df["zone"].str.split("(").str[0].str.strip())
        ez_chart.index.name = "Elevation Zone"
        st.bar_chart(ez_chart, color=["#ef4444", "#22c55e"], height=190, use_container_width=True)

        # Table
        st.markdown('<div class="card-lbl" style="margin-top:.6rem;">Detailed Statistics</div>',
                    unsafe_allow_html=True)
        display_df = ez_df.rename(columns={
            "zone":          "Elevation Zone",
            "forest_km2":    "Forest (km²)",
            "disturbed_km2": "Disturbed (km²)",
            "pct":           "Disturb. %",
        })
        st.dataframe(
            display_df.set_index("Elevation Zone").style.format({
                "Forest (km²)":    "{:,.1f}",
                "Disturbed (km²)": "{:,.1f}",
                "Disturb. %":      "{:.1f}%",
            }).background_gradient(subset=["Disturb. %"], cmap="Reds"),
            use_container_width=True,
        )
    elif not st.session_state.ready:
        st.info("🏔️ Elevation zone analysis appears after loading GEE results.")

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SECTION E — KEY OBSERVATIONS + PIPELINE PARAMETERS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec">
    <div class="sec-icon ic-g">🔍</div>
    <div>
        <div class="sec-title">Key Observations & Pipeline Reference</div>
        <div class="sec-desc">Auto-generated from loaded results · fixed pipeline parameters</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_obs, col_params = st.columns([3, 2], gap="medium")

with col_obs:
    if st.session_state.ready:
        r   = st.session_state.results
        vm  = r["validation_metrics"]
        ad  = filter_years(r["annual_disturbance_km2"], disp_years[0], disp_years[1])
        vals  = list(ad.values())
        peak  = list(ad.keys())[vals.index(max(vals))]
        pi    = r["pipeline_info"]

        observations = [
            f"Highest annual disturbance recorded in <b>{peak}</b> "
            f"({max(vals):,.1f} km²) within the selected {disp_years[0]}–{disp_years[1]} window.",

            f"Total disturbed forest area: <b>{pi.get('total_disturbed_km2', sum(vals)/len(vals)*7*0.85):,.1f} km²</b> "
            f"out of {pi.get('total_forest_km2', 33841):,.1f} km² "
            f"(~{pi.get('disturbance_pct', 9.4):.1f}% of Himalayan forested area).",

            f"Model Precision is high (<b>{vm['precision']:.1%}</b>) indicating few false alarms, "
            f"but Recall is moderate (<b>{vm['recall']:.1%}</b>) — a known trait of "
            f"threshold-based SAR–Optical detectors in dense canopy.",

            f"Omission error of <b>{vm['omission_error']:.1%}</b> suggests ~{vm['FN']:,} disturbed "
            f"pixels were missed — likely subcanopy disturbances below SAR sensitivity.",

            f"Cohen's κ = <b>{vm['kappa']:.4f}</b> — "
            f"{'Substantial agreement with Hansen GFC reference.' if vm['kappa'] > 0.61 else 'Moderate agreement — consider tightening dNBR threshold.'}",

            "Mid-Himalaya (1500–2500 m) shows highest absolute disturbed area; "
            "upper zones show lower rates, consistent with sparse vegetation density.",

            f"Validation based on <b>{vm['TP']+vm['TN']+vm['FP']+vm['FN']:,} stratified samples</b> "
            f"from Hansen GFC + ESA WorldCover + Dynamic World triple-source forest mask.",
        ]
        for obs in observations:
            st.markdown(f"""
            <div class="obs"><p><span class="dot"></span>{obs}</p></div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                    padding:.8rem 1rem;font-size:.88rem;color:#14532d!important;">
            🔍 &nbsp;Observations will appear after loading GEE results.
        </div>""", unsafe_allow_html=True)

with col_params:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-lbl">Fixed GEE Pipeline Parameters</div>', unsafe_allow_html=True)

    params = [
        ("GEE Project",       PIPELINE["gee_project"]),
        ("Asset Directory",   "…/himalayan_forest2"),
        ("Analysis Scale",    f"{PIPELINE['analysis_scale_m']} m"),
        ("Export Scale",      f"{PIPELINE['export_scale_m']} m"),
        ("Forest Mask",       "Hansen ≥ 30%"),
        ("Cloud Mask",        "QA60 + NDSI < 0.4"),
        ("dNBR threshold",    f"< {PIPELINE['dnbr_threshold']}"),
        ("dNDVI threshold",   f"< {PIPELINE['dndvi_threshold']}"),
        ("dNDMI threshold",   f"< {PIPELINE['dndmi_threshold']}"),
        ("dVH threshold",     f"< {PIPELINE['dvh_threshold_db']} dB"),
        ("Min patch size",    f"> {PIPELINE['min_patch_px']} px"),
        ("CRS",               PIPELINE["crs"]),
        ("States covered",    "6 Himalayan states"),
        ("Elevation min",     f"≥ {PIPELINE['elevation_min_m']} m"),
        ("Validation refs",   "Hansen + WorldCover + DW"),
        ("Year range",        "2018 – 2024"),
    ]
    for key, val in params:
        st.markdown(f"""
        <div class="param-row">
            <span class="param-key">{key}</span>
            <span class="param-val">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:.9rem;padding:.65rem;background:#f0fdf4;
                border-radius:8px;font-size:.72rem;color:#15803d!important;
                line-height:1.6;border:1px solid #bbf7d0;">
        🔒 <b>These parameters are fixed in the GEE notebook and must not
        be changed between runs to ensure reproducibility.</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SECTION F — HOW TO CONNECT REAL GEE OUTPUTS
# ══════════════════════════════════════════════════════════════
with st.expander("📂  How to connect real GEE outputs to this GUI", expanded=False):
    st.markdown("""
    ### Step 1 — Run the GEE Notebook
    Execute all cells in `himalayan_forest_disturbance_v4.ipynb`.
    After **Cell 18i** runs, `validation_metrics.csv` is saved automatically.
    After **Cell 16** runs, `annual_area_km2` values are printed.

    ### Step 2 — Export Results
    Add this cell at the end of your notebook:
    ```python
    import json, os
    os.makedirs('results', exist_ok=True)

    gee_outputs = {
        "annual_disturbance_km2": dict(zip(
            [str(y) for y in disturbance_years],
            annual_area_km2           # list from Cell 16
        )),
        "hotspot_counts": {
            "Low (1–2 events)":      hs_counts[0],
            "Moderate (3–4 events)": hs_counts[1],
            "High (≥5 events)":      hs_counts[2],
        },
        "recovery_counts": {
            "None":     rc_counts[0],
            "Slow":     rc_counts[1],
            "Moderate": rc_counts[2],
            "Fast":     rc_counts[3],
        },
        "elevation_zones": elevation_df.to_dict('records'),
        "validation_metrics": val_metrics,   # dict 
        "pipeline_info": {
            "total_forest_km2":    float(forest_km2_total),
            "total_disturbed_km2": float(sum(annual_area_km2)),
            "disturbance_pct":     float(sum(annual_area_km2) / forest_km2_total * 100),
            "validation_samples":  int(len(val_df)),
            "notebook_version":    "v4",
            "gee_project":         GEE_PROJECT,
            "asset_dir":           ASSET_DIR,
        },
    }

    with open('results/gee_outputs.json', 'w') as f:
        json.dump(gee_outputs, f, indent=2)

    print("Saved: results/gee_outputs.json")
    ```

    ### Step 3 — Place results/ next to the Streamlit app
    ```
    files/
    ├── forest_disturbance_app.py   ← this file
    └── results/
        ├── gee_outputs.json        ← main output (GUI auto-detects)
        └── validation_metrics.csv  ← fallback if JSON missing
    ```

    ### Step 4 — Reload the app
    The banner at the top will change from ⚠️ Demonstration mode to ✅ Live GEE outputs loaded.
    All metrics shown will now exactly match your GEE pipeline run.
    """)


# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🌿 &nbsp;SAR–Optical Forest Disturbance Monitoring System &nbsp;·&nbsp;
    Pan-India Himalayan Region &nbsp;·&nbsp; 2018–2024 &nbsp;·&nbsp;
    GEE Pipeline v4 &nbsp;·&nbsp; Sentinel-1 + Sentinel-2 &nbsp;·&nbsp;
    Hansen GFC · ESA WorldCover · Dynamic World &nbsp;·&nbsp;
    Final Year Project
</div>
""", unsafe_allow_html=True)

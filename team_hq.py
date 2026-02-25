# team_hq.py — Team Statistics Dashboard
# Single-page vertical scroll layout (no tabs)

import io
import os
import re
import math
import unicodedata
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests

st.set_page_config(page_title="TEAM-HQ", layout="wide")
st.title("⚽ TEAM-HQ")
st.caption("Team-level Wyscout data. Use the sidebar to filter leagues and metrics.")

# ─────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

@st.cache_data(show_spinner=False)
def _read_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

csv_candidates = list(Path.cwd().glob("all_leagues_stats.csv")) + list(Path.cwd().glob("*.csv"))
csv_candidates = [c for c in csv_candidates if not c.name.startswith("WORLD")]

if csv_candidates:
    selected_csv = st.selectbox("Select team stats CSV:", [c.name for c in csv_candidates])
    df_raw = _read_csv_path(str(Path.cwd() / selected_csv))
else:
    up = st.file_uploader("Upload your team stats CSV", type=["csv"])
    if up is None:
        st.info("Please upload your team stats CSV exported from the Wyscout scraper.")
        st.stop()
    df_raw = _read_csv_bytes(up.getvalue())

# ─────────────────────────────────────────────
# COLUMN NAME NORMALISATION
# ─────────────────────────────────────────────
COL_MAP = {
    "League":            ["league"],
    "Team":              ["team"],
    "Matches":           ["matches"],
    "Wins":              ["wins"],
    "Draws":             ["draws"],
    "Losses":            ["losses"],
    "Points":            ["points"],
    "Expected Points":   ["expected points", "xpoints", "x points", "expected_points"],
    "Goals For":         ["goals for", "goals scored", "goals_for"],
    "Goals Against":     ["goals against", "goals conceded", "goals_against"],
    "Goal Difference":   ["goal difference", "goal diff", "goal_difference"],
    "Avg Age":           ["avg age", "average age", "avg_age"],
    "Possession %":      ["possession %", "possession", "possession_pct"],
    "Goals p90":         ["goals p90", "goals per 90", "goals_p90"],
    "xG p90":            ["xg p90", "xg per 90", "xg_p90"],
    "Shots p90":         ["shots p90", "shots per 90", "shots_p90"],
    "Shot Accuracy %":   ["shot accuracy %", "shooting accuracy %", "shot_accuracy_pct"],
    "Crosses p90":       ["crosses p90", "crosses per 90", "crosses_p90"],
    "Cross Accuracy %":  ["cross accuracy %", "crossing accuracy %", "cross_accuracy_pct"],
    "Dribbles p90":      ["dribbles p90", "dribbles per 90", "dribbles_p90"],
    "Touches in Box p90":["touches in box p90", "touches in box per 90", "touches_in_box_p90"],
    "Shots Against p90": ["shots against p90", "shots vs p90", "shots_against_p90"],
    "Defensive Duels p90":["defensive duels p90", "defensive_duels_p90"],
    "Defensive Duels Won %":["defensive duels won %", "defensive_duels_won_pct", "def duels won %"],
    "Aerial Duels p90": ["aerial duels p90", "aerial_duels_p90"],
    "Aerial Duels Won %":["aerial duels won %", "aerial_duels_won_pct"],
    "PPDA":              ["ppda"],
    "Passes p90":        ["passes p90", "passes per 90", "passes_p90"],
    "Pass Accuracy %":   ["pass accuracy %", "passing accuracy %", "pass_accuracy_pct", "accurate passes %"],
    "Through Passes p90":["through passes p90", "through_passes_p90"],
    "Passes to Final Third p90":["passes to final third p90", "passes_to_final_third_p90", "passes to final 3rd p90"],
    "Passes to Final Third Acc %":["passes to final third acc %", "passes_to_final_third_acc_pct"],
    "Long Passes p90":   ["long passes p90", "long_passes_p90"],
    "Long Pass Accuracy %":["long pass accuracy %", "long_pass_accuracy_pct"],
    "Progressive Passes p90":["progressive passes p90", "progressive_passes_p90"],
    "Progressive Runs p90":["progressive runs p90", "progressive_runs_p90"],
    "xG Against p90":    ["xg against p90", "xga p90", "xg_against_p90", "xg against"],
    "Goals Against p90": ["goals against p90", "goals conceded p90", "goals_against_p90"],
}

def normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    existing_lower = {c.lower().strip(): c for c in df.columns}
    for canonical, aliases in COL_MAP.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in existing_lower:
                rename[existing_lower[alias]] = canonical
                break
    return df.rename(columns=rename)

df_raw = normalise_cols(df_raw)

NUMERIC_COLS = [c for c in COL_MAP.keys() if c not in ("League","Team")]
for c in NUMERIC_COLS:
    if c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

# ── Derive Goals Against p90 from Goals Against / Matches if the p90 col is
#    missing or all-zero (common Wyscout export issue) ──────────────────────
if ("Goals Against" in df_raw.columns and "Matches" in df_raw.columns):
    _ga_col = pd.to_numeric(df_raw.get("Goals Against p90"), errors="coerce")
    _all_zero_or_missing = _ga_col.isna().all() or (_ga_col.fillna(0) == 0).all()
    if _all_zero_or_missing:
        _matches = pd.to_numeric(df_raw["Matches"], errors="coerce").replace(0, np.nan)
        df_raw["Goals Against p90"] = (
            pd.to_numeric(df_raw["Goals Against"], errors="coerce") / _matches
        ).round(2)

# ─────────────────────────────────────────────
# METRIC DISPLAY LABELS
# ─────────────────────────────────────────────
METRIC_LABELS = {
    "Crosses p90":              "Crosses",
    "Cross Accuracy %":         "Crossing Accuracy %",
    "Goals p90":                "Goals Scored",
    "xG p90":                   "xG",
    "Shots p90":                "Shots",
    "Shot Accuracy %":          "Shooting %",
    "Touches in Box p90":       "Touches in Box",
    "Aerial Duels Won %":       "Aerial Duel Success %",
    "Goals Against p90":        "Goals Against",
    "xG Against p90":           "xG Against",
    "Defensive Duels p90":      "Defensive Duels",
    "Defensive Duels Won %":    "Defensive Duel Win %",
    "Shots Against p90":        "Shots Against",
    "PPDA":                     "PPDA",
    "Aerial Duels p90":         "Aerial Duels",
    "Dribbles p90":             "Dribbles",
    "Passes p90":               "Passes",
    "Pass Accuracy %":          "Passing Accuracy %",
    "Long Passes p90":          "Long Passes",
    "Long Pass Accuracy %":     "Long Passing %",
    "Possession %":             "Possession",
    "Passes to Final Third p90":"Passes to Final 3rd",
    "Progressive Passes p90":   "Progressive Passes",
    "Progressive Runs p90":     "Progressive Runs",
    "Expected Points":          "xPoints",
    "Points":                   "Points",
    "Goals For":                "Goals For",
    "Goals Against":            "Goals Against (Total)",
    "Matches":                  "Matches",
    "Avg Age":                  "Avg Age",
}

def mlabel(col):
    return METRIC_LABELS.get(col, col)

# ─────────────────────────────────────────────
# REGION / PRESET MAPS
# ─────────────────────────────────────────────
PRESET_LEAGUES = {
    "Top 5 Europe": {"England 1", "Spain 1", "Germany 1", "Italy 1", "France 1"},
    "Top 10 Europe": {"England 1","Spain 1","Germany 1","Italy 1","France 1",
                      "Netherlands 1","Portugal 1","Belgium 1","Turkey 1","England 2"},
    "EFL": {"England 2","England 3","England 4"},
    "Australia": {"Australia 1"},
}

COUNTRY_TO_REGION = {
    "England":"Europe","Spain":"Europe","Germany":"Europe","Italy":"Europe","France":"Europe",
    "Belgium":"Europe","Portugal":"Europe","Netherlands":"Europe","Croatia":"Europe",
    "Switzerland":"Europe","Norway":"Europe","Sweden":"Europe","Cyprus":"Europe",
    "Czech":"Europe","Greece":"Europe","Austria":"Europe","Hungary":"Europe",
    "Romania":"Europe","Scotland":"Europe","Slovenia":"Europe","Slovakia":"Europe",
    "Ukraine":"Europe","Bulgaria":"Europe","Serbia":"Europe","Albania":"Europe",
    "Bosnia":"Europe","Kosovo":"Europe","Ireland":"Europe","Finland":"Europe",
    "Armenia":"Europe","Georgia":"Europe","Poland":"Europe","Iceland":"Europe",
    "North Macedonia":"Europe","Latvia":"Europe","Montenegro":"Europe","Denmark":"Europe",
    "Estonia":"Europe","Northern Ireland":"Europe","Wales":"Europe","Russia":"Europe",
    "Kazakhstan":"Europe","Lithuania":"Europe","Malta":"Europe","Moldova":"Europe",
    "Israel":"Europe","Turkey":"Asia","Australia":"Oceania",
}

def league_country(lg: str) -> str:
    return re.sub(r"\s*\d+\s*$", "", str(lg)).strip()

def league_region(lg: str) -> str:
    return COUNTRY_TO_REGION.get(league_country(lg), "Other")

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Filters")

    all_leagues = sorted(df_raw["League"].dropna().unique().tolist()) if "League" in df_raw.columns else []
    all_regions = sorted({league_region(lg) for lg in all_leagues})

    sel_regions = st.multiselect("Regions", all_regions, default=all_regions, key="ts_regions")

    st.markdown("#### League Presets")
    pc1, pc2, pc3 = st.columns(3)
    use_top5  = pc1.checkbox("Top 5",  False, key="ts_top5")
    use_top10 = pc2.checkbox("Top 10", False, key="ts_top10")
    use_efl   = pc3.checkbox("EFL",    False, key="ts_efl")
    use_aus   = st.checkbox("Australia", False, key="ts_aus")

    seed = set()
    if use_top5:  seed |= PRESET_LEAGUES["Top 5 Europe"]
    if use_top10: seed |= PRESET_LEAGUES["Top 10 Europe"]
    if use_efl:   seed |= PRESET_LEAGUES["EFL"]
    if use_aus:   seed |= PRESET_LEAGUES["Australia"]

    region_leagues = [lg for lg in all_leagues if league_region(lg) in sel_regions]
    seed = {x for x in seed if x in region_leagues}
    default_leagues = sorted(seed) if seed else region_leagues

    preset_sig = (tuple(sorted(sel_regions)), use_top5, use_top10, use_efl, use_aus)
    if st.session_state.get("ts_preset_sig") != preset_sig:
        st.session_state["ts_preset_sig"] = preset_sig
        st.session_state["ts_leagues_sel"] = default_leagues

    leagues_sel = st.multiselect(
        "Leagues", region_leagues,
        default=st.session_state.get("ts_leagues_sel", default_leagues),
        key="ts_leagues_sel"
    )

    st.markdown("---")
    st.markdown("#### Metric Filter")
    avail_numeric = [c for c in NUMERIC_COLS if c in df_raw.columns]
    metric_filter_col = st.selectbox(
        "Filter by metric", ["None"] + avail_numeric,
        format_func=lambda x: "None" if x == "None" else mlabel(x),
        key="ts_metric_filter_col"
    )
    if metric_filter_col != "None":
        filter_mode = st.radio("Filter mode", ["Raw value", "Percentile"], horizontal=True, key="ts_filter_mode")
        if filter_mode == "Raw value":
            col_min = float(df_raw[metric_filter_col].min()) if metric_filter_col in df_raw.columns else 0.0
            col_max = float(df_raw[metric_filter_col].max()) if metric_filter_col in df_raw.columns else 100.0
            metric_min_val = st.slider(f"Min {mlabel(metric_filter_col)}",
                                       col_min, col_max, col_min, key="ts_metric_raw_min")
        else:
            metric_min_pct = st.slider("Min percentile", 0, 100, 0, key="ts_metric_pct_min")

    st.markdown("---")
    st.markdown("#### Score Filter")
    score_filter_type = st.selectbox(
        "Filter by score", ["None","Overall","Attack","Defense","Possession"],
        key="ts_score_filter"
    )
    score_threshold = st.slider("Min percentile score", 0, 100, 0, key="ts_score_thresh")

# ─────────────────────────────────────────────
# APPLY LEAGUE FILTER
# ─────────────────────────────────────────────
df = df_raw[df_raw["League"].isin(leagues_sel)].copy() if leagues_sel else df_raw.copy()

if df.empty:
    st.warning("No teams match current filters.")
    st.stop()

# ─────────────────────────────────────────────
# COMPOSITE SCORES
# ─────────────────────────────────────────────
INVERT_METRICS = {"xG Against p90", "Goals Against p90", "Shots Against p90", "PPDA", "Goals Against"}

def pct_rank(series: pd.Series, invert: bool = False) -> pd.Series:
    r = series.rank(pct=True) * 100
    return 100 - r if invert else r

for col in NUMERIC_COLS:
    if col not in df.columns:
        continue
    inv = col in INVERT_METRICS
    df[f"_pct_{col}"] = df.groupby("League")[col].transform(
        lambda s, i=inv: pct_rank(s, i)
    )

def score_col(name): return f"_pct_{name}"

def compute_overall(row):
    ep = row.get(score_col("Expected Points"), np.nan)
    xg = row.get(score_col("xG p90"), np.nan)
    xga = row.get(score_col("xG Against p90"), np.nan)
    vals = [v for v in [ep, xg, xga] if pd.notna(v)]
    if not vals: return np.nan
    w = [0.5, 0.25, 0.25][:len(vals)]
    tw = sum(w); return sum(v*ww for v,ww in zip(vals,w)) / tw

def compute_attack(row):
    weights = [
        (row.get(score_col("xG p90"), np.nan), 0.5),
        (row.get(score_col("Goals p90"), np.nan), 0.3),
        (row.get(score_col("Shots p90"), np.nan), 0.05),
        (row.get(score_col("Touches in Box p90"), np.nan), 0.15),
    ]
    vals = [(v,w) for v,w in weights if pd.notna(v)]
    if not vals: return np.nan
    tw = sum(w for _,w in vals)
    return sum(v*w for v,w in vals) / tw

def compute_defense(row):
    weights = [
        (row.get(score_col("xG Against p90"), np.nan), 0.5),
        (row.get(score_col("Goals Against p90"), np.nan), 0.3),
        (row.get(score_col("Shots Against p90"), np.nan), 0.2),
    ]
    vals = [(v,w) for v,w in weights if pd.notna(v)]
    if not vals: return np.nan
    tw = sum(w for _,w in vals)
    return sum(v*w for v,w in vals) / tw

def compute_possession(row):
    weights = [
        (row.get(score_col("Possession %"), np.nan), 0.35),
        (row.get(score_col("Passes p90"), np.nan), 0.30),
        (row.get(score_col("Pass Accuracy %"), np.nan), 0.10),
        (row.get(score_col("Passes to Final Third p90"), np.nan), 0.25),
    ]
    vals = [(v,w) for v,w in weights if pd.notna(v)]
    if not vals: return np.nan
    tw = sum(w for _,w in vals)
    return sum(v*w for v,w in vals) / tw

df["OVR"] = df.apply(compute_overall, axis=1)
df["ATT"] = df.apply(compute_attack, axis=1)
df["DEF"] = df.apply(compute_defense, axis=1)
df["POS"] = df.apply(compute_possession, axis=1)

# Apply score filter
if score_filter_type != "None" and score_threshold > 0:
    scol = {"Overall":"OVR","Attack":"ATT","Defense":"DEF","Possession":"POS"}[score_filter_type]
    df = df[df[scol] >= score_threshold]

# Apply metric filter
if metric_filter_col != "None" and metric_filter_col in df.columns:
    if filter_mode == "Raw value":
        df = df[pd.to_numeric(df[metric_filter_col], errors="coerce") >= metric_min_val]
    else:
        _pct_col = f"_pct_{metric_filter_col}"
        if _pct_col in df.columns:
            df = df[df[_pct_col] >= metric_min_pct]

if df.empty:
    st.warning("No teams after filters. Adjust thresholds.")
    st.stop()

# ─────────────────────────────────────────────
# SORT / RANKING
# ─────────────────────────────────────────────
rank_by = st.radio(
    "Sort teams by",
    ["Overall (OVR)","Attack (ATT)","Defense (DEF)","Possession (POS)","Raw metric"],
    horizontal=True, key="ts_rank_by"
)

raw_metric_options = [c for c in NUMERIC_COLS if c in df.columns]
if rank_by == "Raw metric":
    raw_pick = st.selectbox("Raw metric", raw_metric_options,
                            format_func=mlabel, key="ts_raw_pick")
    asc = raw_pick in INVERT_METRICS
    df["_sort"] = df[raw_pick]
else:
    col_map2 = {"Overall (OVR)":"OVR","Attack (ATT)":"ATT","Defense (DEF)":"DEF","Possession (POS)":"POS"}
    sort_col = col_map2[rank_by]
    asc = False
    df["_sort"] = df[sort_col]

df_sorted = df.dropna(subset=["_sort"]).sort_values("_sort", ascending=asc).reset_index(drop=True)

display_league_filter = st.selectbox(
    "Display league (does not change pool)",
    ["All leagues"] + sorted(df["League"].dropna().unique().tolist()),
    key="ts_disp_league"
)
if display_league_filter != "All leagues":
    df_sorted = df_sorted[df_sorted["League"] == display_league_filter]

team_options = sorted(df["Team"].dropna().unique().tolist())

# ─────────────────────────────────────────────
# BADGE / FLAG HELPERS
# ─────────────────────────────────────────────
try:
    from team_fotmob_urls import FOTMOB_TEAM_URLS as _FOTMOB_URLS
except Exception:
    _FOTMOB_URLS = {}

try:
    from league_logo_urls import get_league_logo_url as _get_league_logo_url
except Exception:
    def _get_league_logo_url(lg): return ""

@st.cache_data(show_spinner=False)
def load_remote_img(url: str):
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return plt.imread(io.BytesIO(r.content))
    except Exception:
        return None

def fotmob_crest_url(team: str) -> str:
    raw = (_FOTMOB_URLS.get(team) or "").strip()
    if not raw: return ""
    m = re.search(r"/teams/(\d+)/", raw)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{m.group(1)}.png" if m else ""

@st.cache_data(show_spinner=False)
def get_team_badge(team: str):
    url = fotmob_crest_url(team)
    if url:
        img = load_remote_img(url)
        if img is not None:
            return img
    return None

def zoom_fit(img, target=32):
    try:
        h, w = img.shape[0], img.shape[1]
        return float(target) / max(h, w)
    except Exception:
        return 1.0

TWEMOJI_SPECIAL = {
    "eng": "1f3f4-e0067-e0062-e0065-e006e-e0067-e007f",
    "sct": "1f3f4-e0067-e0062-e0073-e0063-e0074-e007f",
    "wls": "1f3f4-e0067-e0062-e0077-e006c-e0073-e007f",
}
COUNTRY_TO_CC = {
    "england":"eng","scotland":"sct","wales":"wls","northern ireland":"gb",
    "australia":"au","austria":"at","belgium":"be","bulgaria":"bg","croatia":"hr",
    "cyprus":"cy","czech":"cz","denmark":"dk","france":"fr","germany":"de",
    "hungary":"hu","italy":"it","netherlands":"nl","norway":"no","poland":"pl",
    "portugal":"pt","romania":"ro","russia":"ru","spain":"es",
    "sweden":"se","switzerland":"ch","turkey":"tr","ukraine":"ua","ireland":"ie",
}

def _norm(s): return unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().strip().lower()

def flag_html(league_name: str) -> str:
    country = league_country(league_name)
    n = _norm(country)
    cc = COUNTRY_TO_CC.get(n,"")
    if not cc: return ""
    if cc in TWEMOJI_SPECIAL:
        code = TWEMOJI_SPECIAL[cc]
    else:
        if len(cc)!=2: return ""
        base=0x1F1E6
        code=f"{base+(ord(cc[0].upper())-65):x}-{base+(ord(cc[1].upper())-65):x}"
    src=f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
    return f"<img src='{src}' style='height:18px;vertical-align:middle;margin-right:4px;'>"

def rating_color(v):
    v=float(v)
    if v>=85: return "#2E6114"
    if v>=75: return "#5C9E2E"
    if v>=66: return "#7FBC41"
    if v>=54: return "#A7D763"
    if v>=44: return "#F6D645"
    if v>=25: return "#D77A2E"
    return "#C63733"

def fmt2(n):
    try: return f"{max(0,min(99,int(float(n)))):02d}"
    except: return "00"

def metric_pct(row, col):
    pc = f"_pct_{col}"
    v = row.get(pc, np.nan)
    return float(v) if pd.notna(v) else 0.0

def metric_val(row, col):
    v = row.get(col, np.nan)
    if pd.isna(v): return "—"
    return f"{float(v):.2f}".rstrip("0").rstrip(".")

# ─────────────────────────────────────────────
# LEAGUE POSITION HELPERS
# ─────────────────────────────────────────────
def get_league_pos(row, df_all, metric, ascending=False):
    """Return rank of team within its own league for given metric."""
    lg = row.get("League","")
    if not lg or metric not in df_all.columns:
        return None, None
    lg_df = df_all[df_all["League"] == lg].copy()
    lg_df = lg_df.dropna(subset=[metric])
    n = len(lg_df)
    if n == 0:
        return None, None
    sorted_df = lg_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    team = row.get("Team","")
    matches = sorted_df[sorted_df["Team"] == team]
    if matches.empty:
        return None, n
    pos = matches.index[0] + 1
    return pos, n

# ─────────────────────────────────────────────
# METRIC SECTIONS (new order per spec)
# ─────────────────────────────────────────────
TEAM_METRICS_ATT = [
    ("Crosses",                "Crosses p90"),
    ("Crossing Accuracy %",    "Cross Accuracy %"),
    ("Goals Scored",           "Goals p90"),
    ("xG",                     "xG p90"),
    ("Shots",                  "Shots p90"),
    ("Shooting %",             "Shot Accuracy %"),
    ("Touches in Box",         "Touches in Box p90"),
]
TEAM_METRICS_DEF = [
    ("Goals Against",          "Goals Against p90"),
    ("xG Against",             "xG Against p90"),
    ("Aerial Duels",           "Aerial Duels p90"),
    ("Aerial Duel Success %",  "Aerial Duels Won %"),
    ("Defensive Duels",        "Defensive Duels p90"),
    ("Defensive Duel Win %",   "Defensive Duels Won %"),
    ("Shots Against",          "Shots Against p90"),
    ("PPDA",                   "PPDA"),
]
TEAM_METRICS_POS = [
    ("Dribbles",              "Dribbles p90"),
    ("Possession",            "Possession %"),
    ("Passes",                "Passes p90"),
    ("Passing Accuracy %",    "Pass Accuracy %"),
    ("Long Passes",           "Long Passes p90"),
    ("Long Passing %",        "Long Pass Accuracy %"),
    ("Passes to Final 3rd",   "Passes to Final Third p90"),
    ("Progressive Passes",    "Progressive Passes p90"),
    ("Progressive Runs",      "Progressive Runs p90"),
]

def avail_pairs(pairs, row_or_df):
    if hasattr(row_or_df, "columns"):
        return [(lab,col) for lab,col in pairs if col in row_or_df.columns]
    else:
        return [(lab,col) for lab,col in pairs if col in row_or_df.index]

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 1 – LEAGUE TABLE
# ══════════════════════════════════════════════════════
st.subheader("📊 League Table")
display_cols = ["League","Team","Matches","Wins","Draws","Losses","Points","Expected Points",
                "Goals For","Goals Against","Goal Difference","xG p90","OVR","ATT","DEF","POS"]
show_cols = [c for c in display_cols if c in df_sorted.columns]
st.dataframe(
    df_sorted[show_cols].style.format(
        {c: "{:.1f}" for c in ["OVR","ATT","DEF","POS","xG p90","Expected Points"]}
    ),
    use_container_width=True
)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 2 – PRO LAYOUT (Team Cards)
# ══════════════════════════════════════════════════════
st.subheader("🃏 Pro Layout — Team Cards")

st.markdown("""
<style>
.team-card{position:relative;width:min(440px,96%);display:grid;grid-template-columns:96px 1fr 48px;
    gap:12px;align-items:start;background:#141823;border:1px solid rgba(255,255,255,.06);
    border-radius:20px;padding:16px;margin-bottom:12px;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.03),0 6px 24px rgba(0,0,0,.35);}
.team-badge{width:96px;height:96px;border-radius:12px;border:1px solid #2a3145;
    overflow:hidden;background:#0b0d12;display:flex;align-items:center;justify-content:center;}
.team-badge img{width:100%;height:100%;object-fit:contain;}
.tc-name{font-weight:800;font-size:20px;color:#e8ecff;margin-bottom:6px;}
.tc-sub{color:#a8b3cf;font-size:14px;opacity:.9;}
.tc-pill{padding:2px 6px;min-width:36px;border-radius:6px;font-weight:700;font-size:17px;
    line-height:1;color:#0b0d12;text-align:center;display:inline-block;}
.tc-pill-row{display:flex;gap:8px;align-items:center;margin:3px 0;}
.tc-rank{position:absolute;top:10px;right:14px;color:#b7bfe1;font-weight:800;font-size:18px;}
.tc-wrap{display:flex;justify-content:center;}
.tc-badge-col{display:flex;flex-direction:column;align-items:center;gap:4px;}
.tc-meta{font-size:11px;color:#8899c0;text-align:center;line-height:1.5;}
.tc-meta-label{color:#6b7a9f;font-size:10px;}
.tc-meta-val{color:#c8d4f0;font-weight:700;}
</style>
""", unsafe_allow_html=True)

pro_top_n = st.number_input("Top N teams", 5, 200, 20, 5, key="ts_pro_topn")
pro_search = st.text_input("Search team", "", key="ts_pro_search")
pro_league_filter = st.selectbox(
    "Filter by league", ["All"]+sorted(df_sorted["League"].dropna().unique()),
    key="ts_pro_league"
)

df_pro = df_sorted.copy()
if pro_search:
    df_pro = df_pro[df_pro["Team"].str.contains(pro_search, case=False, na=False)]
if pro_league_filter != "All":
    df_pro = df_pro[df_pro["League"] == pro_league_filter]
df_pro = df_pro.head(int(pro_top_n))

for i, (_, row) in enumerate(df_pro.iterrows()):
    team = str(row.get("Team",""))
    league = str(row.get("League",""))
    ovr = fmt2(row.get("OVR",0))
    att = fmt2(row.get("ATT",0))
    defv= fmt2(row.get("DEF",0))
    pos = fmt2(row.get("POS",0))

    # Points & xPoints league positions
    pts_pos, pts_n = get_league_pos(row, df, "Points", ascending=False)
    xpts_pos, xpts_n = get_league_pos(row, df, "Expected Points", ascending=False)

    pts_str = f"{pts_pos}/{pts_n}" if pts_pos is not None else "—"
    xpts_str = f"{xpts_pos}/{xpts_n}" if xpts_pos is not None else "—"

    avg_age_str = ""
    if "Avg Age" in row and pd.notna(row.get("Avg Age")):
        avg_age_str = f"Avg Age: {float(row['Avg Age']):.1f}"

    badge_url = fotmob_crest_url(team)
    badge_html = f"<img src='{badge_url}' style='width:80px;height:80px;object-fit:contain;'>" if badge_url else "🏟️"

    flag = flag_html(league)

    pill_rows = "".join([
        f"<div class='tc-pill-row'><span class='tc-pill' style='background:{rating_color(row.get(sc,0))}'>"
        f"{fmt2(row.get(sc,0))}</span><span class='tc-sub'>{label}</span></div>"
        for label, sc in [("Overall","OVR"),("Attack","ATT"),("Defense","DEF"),("Possession","POS")]
    ])

    meta_rows = f"""
<div class='tc-meta'>
  <span class='tc-meta-label'>Pos</span> <span class='tc-meta-val'>{pts_str}</span><br>
  <span class='tc-meta-label'>xPos</span> <span class='tc-meta-val'>{xpts_str}</span>
</div>"""
    if avg_age_str:
        meta_rows += f"<div class='tc-meta' style='margin-top:2px;'>{avg_age_str}</div>"

    st.markdown(f"""
    <div class='tc-wrap'><div class='team-card'>
      <div class='tc-badge-col'>
        <div class='team-badge'>{badge_html}</div>
        {meta_rows}
      </div>
      <div>
        <div class='tc-name'>{team}</div>
        {pill_rows}
        <div style='margin-top:8px;font-size:13px;color:#dbe3ff;'>{flag}{league}</div>
      </div>
      <div class='tc-rank'>#{i+1:02d}</div>
    </div></div>
    """, unsafe_allow_html=True)

    with st.expander("Metrics", expanded=False):
        def sec_html(title, pairs):
            rows_html = []
            for lab, col in avail_pairs(pairs, row.to_frame().T):
                p = metric_pct(row, col)
                v = metric_val(row, col)
                rows_html.append(
                    f"<div style='display:flex;align-items:center;gap:10px;padding:6px 8px;border-radius:8px;'>"
                    f"<div style='flex:1;color:#c9d3f2;font-size:14px;'>{lab}</div>"
                    f"<div style='color:#a8b3cf;font-size:12px;min-width:50px;text-align:right;'>{v}</div>"
                    f"<div style='min-width:40px;text-align:center;padding:2px 8px;border-radius:7px;"
                    f"background:{rating_color(p)};font-weight:800;font-size:17px;color:#0b0d12;'>{fmt2(p)}</div>"
                    f"</div>"
                )
            return (f"<div style='background:#121621;border:1px solid #242b3b;border-radius:14px;"
                    f"padding:10px 12px;'><div style='color:#e8ecff;font-weight:800;margin-bottom:8px;'>"
                    f"{title}</div>{''.join(rows_html)}</div>")

        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(sec_html("Attacking", TEAM_METRICS_ATT), unsafe_allow_html=True)
        with col2: st.markdown(sec_html("Defensive", TEAM_METRICS_DEF), unsafe_allow_html=True)
        with col3: st.markdown(sec_html("Possession", TEAM_METRICS_POS), unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 3 – TEAM PROFILE
# ══════════════════════════════════════════════════════
st.subheader("🎯 Team Profile")

sel_team = st.selectbox("Select team", team_options, key="ts_profile_team")
team_row = df[df["Team"] == sel_team]
if team_row.empty:
    st.info("Team not found.")
else:
    team_row = team_row.iloc[0]
    team_league = str(team_row.get("League",""))

    comp_leagues = st.multiselect(
        "Comparison pool (default = own league)",
        sorted(df["League"].dropna().unique()),
        default=[team_league],
        key="ts_profile_comp_leagues"
    )
    pool = df[df["League"].isin(comp_leagues)] if comp_leagues else df[df["League"] == team_league]

    RADAR_METRICS_TEAM = [
        "xG p90", "Goals p90", "Touches in Box p90",
        "xG Against p90", "Goals Against p90", "PPDA",
        "Possession %", "Passes p90", "Long Passes p90",
        "Passes to Final Third p90", "Points", "Expected Points"
    ]
    radar_metrics = [m for m in RADAR_METRICS_TEAM if m in df.columns]

    def team_pct(t_row, pool_df, col, invert=False):
        if col not in pool_df.columns or col not in t_row.index: return 50.0
        s = pd.to_numeric(pool_df[col], errors="coerce").dropna()
        v = float(t_row[col]) if pd.notna(t_row.get(col)) else np.nan
        if pd.isna(v) or s.empty: return 50.0
        p = (s < v).mean()*100 + (s==v).mean()*50
        return (100-p) if invert else p

    pcts = [team_pct(team_row, pool, m, m in INVERT_METRICS) for m in radar_metrics]
    labels_clean = [mlabel(m) for m in radar_metrics]

    # ── Radar styled like attached Python example ──
    color_scale = ["#be2a3e","#e25f48","#f88f4d","#f4d166","#90b960","#4b9b5f","#22763f"]
    cmap = LinearSegmentedColormap.from_list("cs", color_scale)
    bar_colors = [cmap(p/100) for p in pcts]

    N = len(radar_metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)[::-1]
    rotation_shift = np.deg2rad(75) - angles[0]
    rotated_angles = [(a + rotation_shift) % (2*np.pi) for a in angles]
    bar_width = 2 * np.pi / N

    fig = plt.figure(figsize=(8, 6.5))
    fig.patch.set_facecolor('#e6e6e6')
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.70], polar=True)
    ax.set_facecolor('#e6e6e6')
    ax.set_rlim(0, 100)

    # Draw bars
    for i in range(N):
        ax.bar(rotated_angles[i], pcts[i],
               width=bar_width, color=bar_colors[i],
               edgecolor='black', linewidth=1, zorder=2)
        if pcts[i] > 10:
            label_pos = pcts[i] - 8
            ax.text(rotated_angles[i], label_pos, f"{int(round(pcts[i]))}",
                    ha='center', va='center', fontsize=9, weight='bold', color='white', zorder=3)

    # Outer ring
    outer_circle = plt.Circle((0, 0), 100, transform=ax.transData._b,
                               color='black', fill=False, linewidth=2.4)
    ax.add_artist(outer_circle)

    # Dividers
    for i in range(N):
        sep_angle = (rotated_angles[i] - bar_width / 2) % (2*np.pi)
        is_cross = any(np.isclose(sep_angle, a, atol=0.01)
                       for a in [0, np.pi/2, np.pi, 3*np.pi/2])
        ax.plot([sep_angle, sep_angle], [0, 100],
                color='black' if is_cross else '#b0b0b0',
                linewidth=1.8 if is_cross else 1, zorder=4)

    # Metric labels
    label_radius = 125
    for i, label in enumerate(labels_clean):
        ax.text(rotated_angles[i], label_radius, label.upper(),
                ha='center', va='center', fontsize=8, weight='bold', color='black', zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    # Title lines like the Python example (top-left)
    comp_label = ", ".join(comp_leagues) if comp_leagues else team_league
    fig.text(0.05, 0.96, f"{sel_team} Style & Performance", fontsize=14,
             weight='bold', ha='left', color='#111111')
    fig.text(0.05, 0.935, f"Percentile Scores vs {comp_label}", fontsize=9,
             ha='left', color='gray')

    # Badge in top-right corner
    badge_img = get_team_badge(sel_team)
    if badge_img is not None:
        crest_ax = fig.add_axes([0.83, 0.82, 0.14, 0.14])
        crest_ax.imshow(badge_img)
        crest_ax.axis('off')

    st.pyplot(fig, use_container_width=True)

    # Download button
    buf_profile = io.BytesIO()
    fig.savefig(buf_profile, format="png", dpi=200, bbox_inches='tight', facecolor='#e6e6e6')
    st.download_button("⬇️ Download Radar", buf_profile.getvalue(),
                       f"{sel_team.replace(' ','_')}_radar.png", "image/png")
    plt.close(fig)

    # ── Style / Strengths / Weaknesses ──
    STYLE_TEAM = {
        "Possession %":    {"style":"Possession-based","sw":"Possession"},
        "Passes p90":      {"style":"High passing volume","sw":"Passing Volume"},
        "Pass Accuracy %": {"style":"Technical passing","sw":"Passing Accuracy"},
        "PPDA":            {"style":"High press","sw":"Pressing"},
        "xG p90":          {"style":"Creates many chances","sw":"Chance Creation"},
        "Goals p90":       {"style":"Clinical in front of goal","sw":"Goalscoring"},
        "Shots p90":       {"style":"High shot volume","sw":"Shot Volume"},
        "Touches in Box p90":{"style":"Gets into the box often","sw":"Box Penetration"},
        "xG Against p90":  {"style":"Solid defensively","sw":"Defensive Solidity"},
        "Goals Against p90":{"style":"Hard to beat","sw":"Goals Conceded"},
        "Shots Against p90":{"style":"Limits opponent shots","sw":"Shot Prevention"},
        "Progressive Passes p90":{"style":"Progressive ball movement","sw":"Progression"},
        "Progressive Runs p90":{"style":"Dynamic runners","sw":"Carries"},
    }

    HI, LO, STYLE_T = 70, 30, 65
    strengths, weaknesses, styles = [], [], []
    for m in RADAR_METRICS_TEAM:
        if m not in df.columns: continue
        p = team_pct(team_row, pool, m, m in INVERT_METRICS)
        cfg = STYLE_TEAM.get(m,{})
        sw = cfg.get("sw"); sty = cfg.get("style")
        if sw:
            if p >= HI: strengths.append(sw)
            elif p <= LO: weaknesses.append(sw)
        if sty and p >= STYLE_T:
            styles.append(sty)

    def chips_html(items, bg):
        if not items: return "_None identified._"
        spans = [f"<span style='background:{bg};color:#111;padding:2px 8px;border-radius:10px;margin:0 5px 5px 0;display:inline-block;font-size:14px;'>{t}</span>" for t in items[:8]]
        return " ".join(spans)

    st.markdown("**Style:**")
    st.markdown(chips_html(list(dict.fromkeys(styles)), "#bfdbfe"), unsafe_allow_html=True)
    st.markdown("**Strengths:**")
    st.markdown(chips_html(list(dict.fromkeys(strengths)), "#a7f3d0"), unsafe_allow_html=True)
    st.markdown("**Weaknesses:**")
    st.markdown(chips_html(list(dict.fromkeys(weaknesses)), "#fecaca"), unsafe_allow_html=True)

    # ── Scores table with diverging 0-100 colour ──
    st.markdown("**Scores:**")

    def _safe_score(row, key):
        try:
            v = row[key]
            return float(v) if pd.notna(v) else np.nan
        except Exception:
            return np.nan

    score_data = {
        "Score": ["OVR","ATT","DEF","POS"],
        "Value": [_safe_score(team_row, "OVR"), _safe_score(team_row, "ATT"),
                  _safe_score(team_row, "DEF"), _safe_score(team_row, "POS")]
    }
    sdf = pd.DataFrame(score_data).set_index("Score")

    # Diverging red→orange→gold→green 0-100
    SCORE_RED   = np.array([190, 42,  62])
    SCORE_ORG   = np.array([230, 120, 50])
    SCORE_GOLD  = np.array([244, 197, 102])
    SCORE_GREEN = np.array([34,  197, 94])

    def sc_color_div(v):
        if pd.isna(v): return "background:#1a1a2e;color:#fff"
        v = float(np.clip(v, 0, 100))
        if v <= 33:
            t = v / 33
            c = SCORE_RED + (SCORE_ORG - SCORE_RED) * t
        elif v <= 66:
            t = (v - 33) / 33
            c = SCORE_ORG + (SCORE_GOLD - SCORE_ORG) * t
        else:
            t = (v - 66) / 34
            c = SCORE_GOLD + (SCORE_GREEN - SCORE_GOLD) * t
        r, g, b = int(np.clip(c[0],0,255)), int(np.clip(c[1],0,255)), int(np.clip(c[2],0,255))
        text = "#000" if v > 45 else "#fff"
        return f"background:rgb({r},{g},{b});color:{text}"

    try:
        styled_sdf = sdf.style.map(sc_color_div, subset=["Value"])
    except AttributeError:
        styled_sdf = sdf.style.applymap(sc_color_div, subset=["Value"])

    st.dataframe(
        styled_sdf.format({"Value": lambda x: f"{int(round(x))}" if pd.notna(x) else "—"}),
        use_container_width=True
    )

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 4 – FEATURE F (Percentile Board)
# ══════════════════════════════════════════════════════
st.subheader("📋 Feature F — Team Percentile Board")

sel_team_f = st.selectbox("Select team", team_options, key="ts_f_team")
t_row_f = df[df["Team"] == sel_team_f]
if t_row_f.empty:
    st.info("Team not found.")
else:
    t_row_f = t_row_f.iloc[0]
    t_league_f = str(t_row_f.get("League",""))
    pool_f = df[df["League"] == t_league_f]

    def pct_f(col, invert=False):
        if col not in df.columns: return 0.0
        s = pd.to_numeric(pool_f[col], errors="coerce").dropna()
        v = float(t_row_f[col]) if pd.notna(t_row_f.get(col)) else np.nan
        if pd.isna(v) or s.empty: return 0.0
        p = (s<v).mean()*100 + (s==v).mean()*50
        return float(np.clip((100-p) if invert else p, 0, 100))

    def val_f(col):
        v = t_row_f.get(col, np.nan)
        if pd.isna(v): return "—"
        return f"{float(v):.2f}".rstrip("0").rstrip(".")

    # New order per spec
    ATTACKING_F = [
        ("Crosses",                "Crosses p90",         False),
        ("Crossing Accuracy %",    "Cross Accuracy %",    False),
        ("Goals Scored",           "Goals p90",           False),
        ("xG",                     "xG p90",              False),
        ("Shots",                  "Shots p90",           False),
        ("Shooting %",             "Shot Accuracy %",     False),
        ("Touches in Box",         "Touches in Box p90",  False),
    ]
    DEFENSIVE_F = [
        ("Goals Against",          "Goals Against p90",        True),
        ("xG Against",             "xG Against p90",           True),
        ("Aerial Duels",           "Aerial Duels p90",         False),
        ("Aerial Duel Success %",  "Aerial Duels Won %",       False),
        ("Defensive Duels",        "Defensive Duels p90",      False),
        ("Defensive Duel Win %",   "Defensive Duels Won %",    False),
        ("Shots Against",          "Shots Against p90",        True),
        ("PPDA",                   "PPDA",                     True),
    ]
    POSSESSION_F = [
        ("Dribbles",              "Dribbles p90",              False),
        ("Possession",            "Possession %",              False),
        ("Passes",                "Passes p90",                False),
        ("Passing Accuracy %",    "Pass Accuracy %",           False),
        ("Long Passes",           "Long Passes p90",           False),
        ("Long Passing %",        "Long Pass Accuracy %",      False),
        ("Passes to Final 3rd",   "Passes to Final Third p90", False),
        ("Progressive Passes",    "Progressive Passes p90",    False),
        ("Progressive Runs",      "Progressive Runs p90",      False),
    ]

    sections_f = [
        ("Attacking",  [(lab,pct_f(col,inv),val_f(col)) for lab,col,inv in ATTACKING_F  if col in df.columns]),
        ("Defensive",  [(lab,pct_f(col,inv),val_f(col)) for lab,col,inv in DEFENSIVE_F  if col in df.columns]),
        ("Possession", [(lab,pct_f(col,inv),val_f(col)) for lab,col,inv in POSSESSION_F if col in df.columns]),
    ]

    PAGE_BG="#0a0f1c"; AX_BG="#0f151f"; TRACK="#1b2636"
    TITLE_C="#f3f5f7"; LABEL_C="#e8eef8"; DIVIDER="#ffffff"
    TAB_RED=np.array([199,54,60]); TAB_GOLD=np.array([240,197,106]); TAB_GREEN=np.array([61,166,91])

    # Editable footer
    footer_text_f = st.text_input("Footer text (optional)", "Percentile Rank",
                                   key="ts_f_footer")

    def blend(c1,c2,t):
        c=c1+(c2-c1)*np.clip(t,0,1)
        return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"

    def p2rgb(v):
        v=float(np.clip(v,0,100))
        return blend(TAB_RED,TAB_GOLD,v/50) if v<=50 else blend(TAB_GOLD,TAB_GREEN,(v-50)/50)

    total_rows = sum(len(lst) for _,lst in sections_f)
    fig_f = plt.figure(figsize=(10,8),dpi=100)
    fig_f.patch.set_facecolor(PAGE_BG)

    left_m=0.035; right_m=0.02; top_m=0.04; bot_m=0.09
    header_h=0.06; gap=0.018
    rows_space=1-(top_m+bot_m)-header_h*len(sections_f)-gap*(len(sections_f)-1)
    row_slot=rows_space/max(total_rows,1)
    BAR_FRAC=0.85; gutter=0.225; ticks=np.arange(0,101,10)
    LEFT=left_m+0.015

    y_top=1-top_m
    for idx,(title,data) in enumerate(sections_f):
        is_last=(idx==len(sections_f)-1)
        n=len(data)
        fig_f.text(LEFT, y_top-0.008, title, ha="left", va="top", fontsize=16, fontweight="900", color=TITLE_C)
        ax_f=fig_f.add_axes([LEFT+gutter, y_top-header_h-n*row_slot, 1-LEFT-right_m-gutter, n*row_slot])
        ax_f.set_facecolor(AX_BG); ax_f.set_xlim(0,100); ax_f.set_ylim(-0.5,n-0.5)
        for s in ax_f.spines.values(): s.set_visible(False)
        ax_f.tick_params(axis="x",bottom=False,labelbottom=False,length=0)
        for i in range(n): ax_f.add_patch(plt.Rectangle((0,i-BAR_FRAC/2),100,BAR_FRAC,color=TRACK,ec="none",zorder=0.5))
        for gx in ticks: ax_f.vlines(gx,-0.5,n-0.5,colors=(1,1,1,0.16),lw=0.8,zorder=0.75)
        for i,(lab,pct,val_str) in enumerate(data[::-1]):
            bw=float(np.clip(pct,0,100))
            ax_f.add_patch(plt.Rectangle((0,i-BAR_FRAC/2),bw,BAR_FRAC,color=p2rgb(bw),ec="none",zorder=1))
            ax_f.text(1,i,val_str,ha="left",va="center",fontsize=8,fontweight="400",color="#0B0B0B",zorder=2)
        ax_f.axvline(50,color="#FFFFFF",ls=(0,(4,4)),lw=1.5,alpha=0.85,zorder=3.5)
        for i,(lab,_,_) in enumerate(data[::-1]):
            yf=(y_top-header_h-n*row_slot)+((i+0.5)*row_slot)
            fig_f.text(LEFT, yf, lab, ha="left", va="center", fontsize=10, fontweight="bold", color=LABEL_C)
        if is_last:
            from matplotlib.transforms import ScaledTranslation
            trans=ax_f.get_xaxis_transform()
            off_in=ScaledTranslation(7/72,0,fig_f.dpi_scale_trans)
            off_0=ScaledTranslation(4/72,0,fig_f.dpi_scale_trans)
            off_100=ScaledTranslation(10/72,0,fig_f.dpi_scale_trans)
            yl=-0.075
            for gx in ticks:
                ax_f.plot([gx,gx],[-0.03,0],transform=trans,color=(1,1,1,0.6),lw=1.1,clip_on=False,zorder=4)
                ax_f.text(gx,yl,f"{int(gx)}",transform=trans,ha="center",va="top",fontsize=10,fontweight="700",color="#FFF",zorder=4,clip_on=False)
                off=off_0 if gx==0 else (off_100 if gx==100 else off_in)
                ax_f.text(gx,yl,"%",transform=trans+off,ha="left",va="top",fontsize=10,fontweight="700",color="#FFF",zorder=4,clip_on=False)
        else:
            y0=y_top-header_h-n*row_slot-0.008
            fig_f.lines.append(plt.Line2D([LEFT,1-right_m],[y0,y0],transform=fig_f.transFigure,color=DIVIDER,lw=1.2,alpha=0.95))
        y_top=y_top-header_h-n*row_slot-gap

    fig_f.text((LEFT+gutter+(1-right_m))/2, bot_m*0.3, footer_text_f if footer_text_f.strip() else "Percentile Rank",
               ha="center",va="center",fontsize=11,fontweight="bold",color=LABEL_C)
    st.pyplot(fig_f, use_container_width=True)
    buf_f=io.BytesIO(); fig_f.savefig(buf_f,format="png",dpi=130,bbox_inches="tight",facecolor=PAGE_BG)
    st.download_button("⬇️ Download Feature F", buf_f.getvalue(),
                       f"{sel_team_f.replace(' ','_')}_featureF.png","image/png")
    plt.close(fig_f)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 5 – FEATURE Y (Polar Radar)
# ══════════════════════════════════════════════════════
st.subheader("🌀 Feature Y — Team Polar Radar")

sel_team_y = st.selectbox("Select team", team_options, key="ts_y_team")
t_row_y = df[df["Team"] == sel_team_y]
if t_row_y.empty:
    st.info("Team not found.")
else:
    t_row_y = t_row_y.iloc[0]
    t_league_y = str(t_row_y.get("League",""))

    comp_y = st.multiselect(
        "Comparison pool", sorted(df["League"].dropna().unique()),
        default=[t_league_y], key="ts_y_comp"
    )
    pool_y = df[df["League"].isin(comp_y)] if comp_y else df[df["League"]==t_league_y]

    # Custom title option (default off)
    use_custom_title_y = st.checkbox("Add custom title", value=False, key="ts_y_custom_title_toggle")
    custom_title_y = ""
    if use_custom_title_y:
        custom_title_y = st.text_input("Custom title text", "", key="ts_y_custom_title")

    METRICS_Y = [
        "xG p90","Goals p90","Touches in Box p90",
        "xG Against p90","Goals Against p90","PPDA",
        "Possession %","Passes p90","Passes to Final Third p90",
        "Long Passes p90","Points","Expected Points"
    ]
    metrics_y = [m for m in METRICS_Y if m in df.columns]

    def pct_y(col):
        inv = col in INVERT_METRICS
        if col not in pool_y.columns: return 50
        s = pd.to_numeric(pool_y[col], errors="coerce").dropna()
        v = float(t_row_y[col]) if pd.notna(t_row_y.get(col)) else np.nan
        if pd.isna(v) or s.empty: return 50
        p = (s<v).mean()*100 + (s==v).mean()*50
        return float(np.clip((100-p) if inv else p, 0, 100))

    pcts_y = [pct_y(m) for m in metrics_y]
    labels_y = [mlabel(m) for m in metrics_y]

    N_y = len(metrics_y)
    color_scale_y = ["#be2a3e","#e25f48","#f88f4d","#f4d166","#90b960","#4b9b5f","#22763f"]
    cmap_y = LinearSegmentedColormap.from_list("csy", color_scale_y)
    bar_colors_y = [cmap_y(p/100) for p in pcts_y]

    angles_y = np.linspace(0, 2*np.pi, N_y, endpoint=False)[::-1]
    rot_shift_y = np.deg2rad(75) - angles_y[0]
    rot_angles_y = [(a+rot_shift_y)%(2*np.pi) for a in angles_y]
    bar_w_y = (2*np.pi/N_y)*0.85

    fig_y = plt.figure(figsize=(8, 6.5))
    fig_y.patch.set_facecolor("#0a0f1c")
    ax_y = fig_y.add_axes([0.05, 0.05, 0.9, 0.85], polar=True)
    ax_y.set_facecolor("#0a0f1c")
    ax_y.set_rlim(0, 100)

    # Background track bars
    for i in range(N_y):
        ax_y.bar(rot_angles_y[i], 100, width=bar_w_y, color="#444", edgecolor="none", zorder=0)

    for i, p in enumerate(pcts_y):
        c = bar_colors_y[i]
        ax_y.bar(rot_angles_y[i], p, width=bar_w_y, color=c, edgecolor="white", linewidth=1.5, zorder=2)
        if p >= 20:
            lp = p - 10 if p >= 30 else p * 0.7
            ax_y.text(rot_angles_y[i], lp, f"{int(round(p))}",
                      ha='center', va='center', fontsize=11, weight='bold', color='white', zorder=3)

    # Dividers
    for i in range(N_y):
        sep = (rot_angles_y[i] - bar_w_y / 2) % (2*np.pi)
        is_cross = any(np.isclose(sep, a, atol=0.01) for a in [0, np.pi/2, np.pi, 3*np.pi/2])
        ax_y.plot([sep, sep], [0, 100],
                  color=(1, 1, 1, 1.0) if is_cross else (1, 1, 1, 0.25),
                  linewidth=1.8 if is_cross else 1, zorder=4)

    # Reference rings
    for rp in [90, 75, 50, 25]:
        theta_ref = np.linspace(0, 2*np.pi, 500)
        ax_y.plot(theta_ref, [rp]*500, linestyle="dotted", lw=1.2, color="lightgrey", zorder=1)

    # Labels
    for i, lab in enumerate(labels_y):
        ax_y.text(rot_angles_y[i], 145, lab.upper(),
                  ha='center', va='center', fontsize=9, weight='bold', color='white', zorder=5)

    ax_y.set_xticks([]); ax_y.set_yticks([])
    ax_y.spines['polar'].set_visible(False); ax_y.grid(False)

    # Custom title (only if enabled and not empty)
    if use_custom_title_y and custom_title_y.strip():
        fig_y.text(0.5, 0.97, custom_title_y.strip(), ha='center', fontsize=13,
                   weight='bold', color='white')

    st.pyplot(fig_y, use_container_width=True)
    buf_y=io.BytesIO(); fig_y.savefig(buf_y,format="png",dpi=300,bbox_inches="tight",facecolor="#0a0f1c")
    st.download_button("⬇️ Download Feature Y", buf_y.getvalue(),
                       f"{sel_team_y.replace(' ','_')}_featureY.png","image/png")
    plt.close(fig_y)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 6 – LEADERBOARD
# ══════════════════════════════════════════════════════
st.subheader("📉 Leaderboard")

with st.expander("Leaderboard settings", expanded=False):
    lb_metric = st.selectbox("Metric", [c for c in NUMERIC_COLS if c in df.columns],
                             format_func=mlabel, key="ts_lb_metric")
    lb_n = st.slider("Top N", 5, 40, 20, 5, key="ts_lb_n")
    lb_theme = st.radio("Theme", ["Light","Dark"], horizontal=True, key="ts_lb_theme")
    lb_palette = st.selectbox("Palette", ["Red–Gold–Green","All Black","All Blue","Light→Dark Blue"], key="ts_lb_pal")
    show_team_names_lb = st.checkbox("Show league in label", True, key="ts_lb_names")

if lb_metric not in df.columns:
    st.info("Metric not available.")
else:
    lb_df = df[["Team","League",lb_metric]].dropna(subset=[lb_metric]).copy()
    lb_df[lb_metric] = pd.to_numeric(lb_df[lb_metric], errors="coerce")
    lb_df = lb_df.dropna().sort_values(lb_metric, ascending=(lb_metric in INVERT_METRICS)).reset_index(drop=True)
    lb_df = lb_df.head(lb_n)

    if lb_theme=="Light":
        PBG="#ebebeb"; ABG="#ebebeb"; TXT="#111111"; GRID="#d7d7d7"; SPINE="#c8c8c8"
    else:
        PBG="#0a0f1c"; ABG="#0f151f"; TXT="#f5f5f5"; GRID="#3a4050"; SPINE="#6b7280"

    vals_lb = lb_df[lb_metric].values
    vmin,vmax=(float(vals_lb.min()),float(vals_lb.max())) if len(vals_lb)>1 else (0.0,1.0)
    if vmin==vmax: vmax=vmin+1e-6
    ts_lb_arr = (vals_lb-vmin)/(vmax-vmin)

    if lb_metric in INVERT_METRICS:
        ts_lb_arr = 1.0 - ts_lb_arr

    TAB_R=np.array([199,54,60]); TAB_G=np.array([240,197,106]); TAB_GN=np.array([61,166,91])

    def lb_color(t: float):
        t = float(np.clip(t, 0, 1))
        if lb_palette == "Red–Gold–Green":
            if t <= 0.5:
                a, b2, t2 = TAB_R, TAB_G, t/0.5
            else:
                a, b2, t2 = TAB_G, TAB_GN, (t-0.5)/0.5
            return tuple(np.clip(a+(b2-a)*t2, 0, 255)/255)
        if lb_palette == "All Black":
            return (0,0,0)
        if lb_palette == "All Blue":
            return (15/255,70/255,180/255)
        if lb_palette == "Light→Dark Blue":
            lo=np.array([191,210,255]); hi=np.array([10,42,102])
            return tuple(np.clip(lo+(hi-lo)*t, 0, 255)/255)
        return (0,0,0)

    bar_colors_lb = [lb_color(float(t)) for t in ts_lb_arr]

    fig_lb,ax_lb=plt.subplots(figsize=(11.5,6.2))
    fig_lb.patch.set_facecolor(PBG); ax_lb.set_facecolor(ABG)
    ypos=np.arange(len(vals_lb))
    bars=ax_lb.barh(ypos, vals_lb, color=bar_colors_lb, edgecolor="none", zorder=2)
    ax_lb.invert_yaxis()
    ytlabs=[f"{row['Team']} ({row['League']})" if show_team_names_lb else row["Team"] for _,row in lb_df.iterrows()]
    ax_lb.set_yticks(ypos); ax_lb.set_yticklabels(ytlabs, fontsize=10, color=TXT)
    ax_lb.set_xlabel(mlabel(lb_metric), color=TXT, fontsize=11, fontweight="semibold")
    ax_lb.grid(axis="x", color=GRID, lw=0.8, zorder=1)
    for s in ["top","right","left"]: ax_lb.spines[s].set_visible(False)
    ax_lb.spines["bottom"].set_color(SPINE)
    ax_lb.tick_params(axis="y", length=0)
    for tick in ax_lb.get_xticklabels(): tick.set_color(TXT)
    xmax=float(vals_lb.max())*1.12 if len(vals_lb) else 1
    ax_lb.set_xlim(0,xmax)
    pad=(ax_lb.get_xlim()[1]-ax_lb.get_xlim()[0])*0.012
    for rect,v in zip(bars,vals_lb):
        ax_lb.text(rect.get_width()+pad, rect.get_y()+rect.get_height()/2,
                   f"{v:.2f}".rstrip("0").rstrip("."), va="center",ha="left",fontsize=8.5,color=TXT)
    fig_lb.suptitle(f"Top {lb_n} — {mlabel(lb_metric)}", fontsize=22, fontweight="bold", color=TXT)
    plt.subplots_adjust(left=0.3,right=0.96,bottom=0.14,top=0.90)
    st.pyplot(fig_lb, use_container_width=True)
    plt.close(fig_lb)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 7 – SCATTER
# ══════════════════════════════════════════════════════
st.subheader("🔵 Scatter Chart")

num_cols_sc = [c for c in NUMERIC_COLS if c in df.columns]
with st.expander("Scatter settings", expanded=False):
    sc_x = st.selectbox("X axis", num_cols_sc,
                        index=num_cols_sc.index("xG p90") if "xG p90" in num_cols_sc else 0,
                        format_func=mlabel, key="ts_sc_x")
    sc_y = st.selectbox("Y axis", num_cols_sc,
                        index=num_cols_sc.index("xG Against p90") if "xG Against p90" in num_cols_sc else 1,
                        format_func=mlabel, key="ts_sc_y")
    sc_theme = st.radio("Theme",["Light","Dark"],horizontal=True,key="ts_sc_theme")
    sc_labels = st.checkbox("Show labels", True, key="ts_sc_labels")
    sc_medians = st.checkbox("Show medians", True, key="ts_sc_medians")
    sc_size = st.slider("Point size",30,300,150,key="ts_sc_size")

PAGE_BG_SC="#ebebeb" if sc_theme=="Light" else "#0a0f1c"
PLOT_BG_SC="#f3f3f3" if sc_theme=="Light" else "#0f151f"
GRID_SC="#d7d7d7" if sc_theme=="Light" else "#3a4050"
TXT_SC="#111111" if sc_theme=="Light" else "#f5f5f5"
STROKE_SC="#ffffff" if sc_theme=="Light" else "#1e293b"

sc_df = df[["Team","League",sc_x,sc_y]].dropna(subset=[sc_x,sc_y]).copy()
if sc_df.empty:
    st.info("No data for selected metrics.")
else:
    fig_sc,ax_sc=plt.subplots(figsize=(14,8),dpi=100)
    fig_sc.patch.set_facecolor(PAGE_BG_SC); ax_sc.set_facecolor(PLOT_BG_SC)
    ax_sc.scatter(sc_df[sc_x], sc_df[sc_y], s=sc_size, color="#1D4ED8", alpha=0.75, edgecolors="none", zorder=2)
    if sc_medians:
        mx=float(sc_df[sc_x].median()); my=float(sc_df[sc_y].median())
        mc="#000" if sc_theme=="Light" else "#fff"
        ax_sc.axvline(mx,color=mc,ls=(0,(4,4)),lw=2,zorder=3)
        ax_sc.axhline(my,color=mc,ls=(0,(4,4)),lw=2,zorder=3)
    if sc_labels:
        from matplotlib import patheffects as pe
        for _,row in sc_df.iterrows():
            t=ax_sc.annotate(row["Team"],(row[sc_x],row[sc_y]),xytext=(6,8),
                textcoords="offset points",fontsize=8,color=TXT_SC,fontweight="semibold",zorder=5)
            t.set_path_effects([pe.withStroke(linewidth=1.8,foreground=STROKE_SC,alpha=0.9)])
    ax_sc.set_xlabel(mlabel(sc_x),color=TXT_SC,fontsize=13,fontweight="semibold")
    ax_sc.set_ylabel(mlabel(sc_y),color=TXT_SC,fontsize=13,fontweight="semibold")
    ax_sc.grid(True,color=GRID_SC,lw=0.8)
    for s in ax_sc.spines.values(): s.set_color("#9ca3af")
    for tick in ax_sc.get_xticklabels()+ax_sc.get_yticklabels(): tick.set_color(TXT_SC)
    plt.tight_layout()
    st.pyplot(fig_sc,use_container_width=True)
    plt.close(fig_sc)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 8 – COMPARISON RADAR
# ══════════════════════════════════════════════════════
st.subheader("⚡ Team Comparison Radar")

RADAR_COMP_METRICS = [
    "xG p90","Goals p90","Touches in Box p90",
    "xG Against p90","Goals Against p90","PPDA",
    "Possession %","Passes p90","Passes to Final Third p90",
    "Long Passes p90","Points","Expected Points"
]
radar_comp_avail = [m for m in RADAR_COMP_METRICS if m in df.columns]

with st.expander("Radar settings", expanded=False):
    comp_theme = st.radio("Theme",["Light","Dark"],horizontal=True,key="ts_comp_theme")

if comp_theme=="Dark":
    PBG_C="#0a0f1c"; AX_C="#0a0f1c"; LABEL_C_R="#f5f5f5"; TICK_C="#e5e7eb"; MINS_C="#f5f5f5"
    RING_IN="#3a4050"; RING_OUT="#cbd5e1"
else:
    PBG_C="#ffffff"; AX_C="#ebebeb"; LABEL_C_R="#0f172a"; TICK_C="#6b7280"; MINS_C="#374151"
    RING_IN=RING_OUT="#d1d5db"

team_a = st.selectbox("Team A (red)", team_options, key="ts_comp_a")
team_b = st.selectbox("Team B (blue)", [t for t in team_options if t!=team_a], key="ts_comp_b")

row_a = df[df["Team"]==team_a].iloc[0] if not df[df["Team"]==team_a].empty else None
row_b = df[df["Team"]==team_b].iloc[0] if not df[df["Team"]==team_b].empty else None

if row_a is not None and row_b is not None:
    leagues_ab = {str(row_a.get("League","")), str(row_b.get("League",""))}
    pool_ab = df[df["League"].isin(leagues_ab)]

    def pct_ab(t_row, col):
        inv = col in INVERT_METRICS
        if col not in pool_ab.columns: return 50.0
        s = pd.to_numeric(pool_ab[col], errors="coerce").dropna()
        v = float(t_row[col]) if pd.notna(t_row.get(col)) else np.nan
        if pd.isna(v) or s.empty: return 50.0
        p = (s<v).mean()*100 + (s==v).mean()*50
        return float(np.clip((100-p) if inv else p,0,100))

    A_r = np.array([pct_ab(row_a,m) for m in radar_comp_avail])
    B_r = np.array([pct_ab(row_b,m) for m in radar_comp_avail])
    labels_c = [mlabel(m) for m in radar_comp_avail]

    N_c=len(radar_comp_avail)
    theta_c=np.linspace(0,2*np.pi,N_c,endpoint=False)
    theta_cc=np.concatenate([theta_c,theta_c[:1]])
    Ar_c=np.concatenate([A_r,A_r[:1]]); Br_c=np.concatenate([B_r,B_r[:1]])

    COL_A="#C81E1E"; COL_B="#1D4ED8"
    FILL_A=(200/255,30/255,30/255,0.55); FILL_B=(29/255,78/255,216/255,0.55)
    INNER_HOLE=10

    fig_c=plt.figure(figsize=(13.2,8.0),dpi=220)
    fig_c.patch.set_facecolor(PBG_C)
    ax_c=plt.subplot(111,polar=True); ax_c.set_facecolor(AX_C)
    ax_c.set_theta_offset(np.pi/2); ax_c.set_theta_direction(-1)
    ax_c.set_xticks(theta_c); ax_c.set_xticklabels([])
    ax_c.set_yticks([]); ax_c.grid(False)
    for s in ax_c.spines.values(): s.set_visible(False)

    ring_edges=np.linspace(INNER_HOLE,100,11)
    for i in range(10):
        r0,r1=ring_edges[i],ring_edges[i+1]
        band="#162235" if ((9-i)%2==0 and comp_theme=="Dark") else ("#e5e7eb" if (9-i)%2==0 else AX_C)
        ax_c.add_artist(mpatches.Wedge((0,0),r1,0,360,width=(r1-r0),
            transform=ax_c.transData._b,facecolor=band,edgecolor="none",zorder=0.8))

    ring_t=np.linspace(0,2*np.pi,361)
    for j,r in enumerate(ring_edges):
        col=RING_OUT if j==len(ring_edges)-1 else RING_IN
        ax_c.plot(ring_t,np.full_like(ring_t,r),color=col,lw=1.0,zorder=0.9)

    for i,ang in enumerate(theta_c):
        for rr in ring_edges[2:]:
            pct_val=int(round((rr-INNER_HOLE)/(100-INNER_HOLE)*100))
            ax_c.text(ang,rr-1.8,f"{pct_val}",ha="center",va="center",fontsize=6,color=TICK_C,zorder=1.1)

    OUTER_R=107
    for ang,lab in zip(theta_c,labels_c):
        rot=np.degrees(ax_c.get_theta_direction()*ang+ax_c.get_theta_offset())-90
        rn=((rot+180)%360)-180
        if rn>90 or rn<-90: rot+=180
        ax_c.text(ang,OUTER_R,lab,rotation=rot,rotation_mode="anchor",
                  ha="center",va="center",fontsize=9,color=LABEL_C_R,fontweight=600,clip_on=False,zorder=2.2)

    ax_c.add_artist(plt.Circle((0,0),radius=INNER_HOLE-0.6,transform=ax_c.transData._b,
                               color=PBG_C,zorder=1.2,ec="none"))
    ax_c.plot(theta_cc,Ar_c,color=COL_A,lw=2.2,zorder=3)
    ax_c.fill(theta_cc,Ar_c,color=FILL_A,zorder=2.5)
    ax_c.plot(theta_cc,Br_c,color=COL_B,lw=2.2,zorder=3)
    ax_c.fill(theta_cc,Br_c,color=FILL_B,zorder=2.5)
    ax_c.set_rlim(0,100)

    fig_c.text(0.12,0.96,team_a,color=COL_A,fontsize=22,fontweight="bold",ha="left")
    fig_c.text(0.12,0.935,str(row_a.get("League","")),color=COL_A,fontsize=11,ha="left")
    fig_c.text(0.88,0.96,team_b,color=COL_B,fontsize=22,fontweight="bold",ha="right")
    fig_c.text(0.88,0.935,str(row_b.get("League","")),color=COL_B,fontsize=11,ha="right")

    st.pyplot(fig_c, use_container_width=True)
    buf_c=io.BytesIO(); fig_c.savefig(buf_c,format="png",dpi=220,facecolor=PBG_C)
    st.download_button("⬇️ Download Comparison Radar", buf_c.getvalue(),
                       f"comparison_{team_a.replace(' ','_')}_vs_{team_b.replace(' ','_')}.png","image/png")
    plt.close(fig_c)
else:
    st.info("Select two teams above.")

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
team_league_map = df.set_index("Team")["League"].to_dict() if "League" in df.columns else {}

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
team_league = team_league_map.get(sel_team, "")  # always defined for Feature F/Y below
team_row = df[df["Team"] == sel_team]
if team_row.empty:
    st.info("Team not found in current filter.")
else:
    team_row = team_row.iloc[0]
    team_league = str(team_row["League"]) if "League" in team_row.index else ""

    _avail_leagues = sorted(df["League"].dropna().unique())

    comp_leagues = st.multiselect(
        "Comparison pool (default = own league)",
        _avail_leagues,
        default=[team_league] if team_league in _avail_leagues else [],
        key=f"ts_profile_comp_{sel_team}",
    )
    pool = df[df["League"].isin(comp_leagues)] if comp_leagues else df[df["League"] == team_league]

    # Editable title & subtitle — value= reseeds automatically when sel_team changes
    _pool_label = ", ".join(comp_leagues) if comp_leagues else team_league
    c_title, c_sub = st.columns(2)
    with c_title:
        profile_title = st.text_input(
            "Chart title",
            value=f"{sel_team} Style & Performance",
            key=f"ts_profile_title_{sel_team}",
        )
    with c_sub:
        profile_subtitle = st.text_input(
            "Chart subtitle",
            value=f"Percentile Scores vs {_pool_label}",
            key=f"ts_profile_subtitle_{sel_team}",
        )

    RADAR_METRICS_TEAM = [
        "xG p90", "Goals p90", "Touches in Box p90",
        "xG Against p90", "Goals Against p90", "PPDA",
        "Possession %", "Passes p90", "Long Passes p90",
        "Passes to Final Third p90", "Points", "Expected Points"
    ]
    radar_metrics = [m for m in RADAR_METRICS_TEAM if m in df.columns]

    # Custom label override for Team Profile radar only
    PROFILE_LABEL_OVERRIDE = {
        "Passes to Final Third p90": "Passes Final 3rd",
    }

    def team_pct(t_row, pool_df, col, invert=False):
        if col not in pool_df.columns or col not in t_row.index: return 50.0
        s = pd.to_numeric(pool_df[col], errors="coerce").dropna()
        v = float(t_row[col]) if pd.notna(t_row.get(col)) else np.nan
        if pd.isna(v) or s.empty: return 50.0
        p = (s < v).mean()*100 + (s==v).mean()*50
        return (100-p) if invert else p

    pcts = [team_pct(team_row, pool, m, m in INVERT_METRICS) for m in radar_metrics]
    labels_clean = [PROFILE_LABEL_OVERRIDE.get(m, mlabel(m)) for m in radar_metrics]

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

    # Title lines (editable)
    if profile_title.strip():
        fig.text(0.05, 0.96, profile_title.strip(), fontsize=14,
                 weight='bold', ha='left', color='#111111')
    if profile_subtitle.strip():
        fig.text(0.05, 0.935, profile_subtitle.strip(), fontsize=9,
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

    # ── Scores — render as HTML to avoid pandas styler issues ──
    st.markdown("**Scores:**")

    SCORE_RED_S   = np.array([190, 42,  62])
    SCORE_ORG_S   = np.array([230, 120, 50])
    SCORE_GOLD_S  = np.array([244, 197, 102])
    SCORE_GREEN_S = np.array([34,  197, 94])

    def score_bg(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "#333", "#fff"
        v = float(np.clip(v, 0, 100))
        if v <= 33:
            t = v / 33; c = SCORE_RED_S + (SCORE_ORG_S - SCORE_RED_S) * t
        elif v <= 66:
            t = (v - 33) / 33; c = SCORE_ORG_S + (SCORE_GOLD_S - SCORE_ORG_S) * t
        else:
            t = (v - 66) / 34; c = SCORE_GOLD_S + (SCORE_GREEN_S - SCORE_GOLD_S) * t
        r2, g2, b2 = int(np.clip(c[0],0,255)), int(np.clip(c[1],0,255)), int(np.clip(c[2],0,255))
        fg = "#000" if v > 45 else "#fff"
        return f"rgb({r2},{g2},{b2})", fg

    def _get_score(row, key):
        try:
            v = row[key]
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    score_rows_html = ""
    for label, key in [("Overall","OVR"),("Attack","ATT"),("Defense","DEF"),("Possession","POS")]:
        val = _get_score(team_row, key)
        bg_c, fg_c = score_bg(val)
        val_str = f"{int(round(val))}" if val is not None else "—"
        score_rows_html += (
            f"<tr>"
            f"<td style='padding:8px 14px;font-weight:700;color:#e8ecff;background:#1a2035;"
            f"border-bottom:1px solid #2a3145;'>{label}</td>"
            f"<td style='padding:8px 14px;text-align:center;font-weight:800;font-size:18px;"
            f"background:{bg_c};color:{fg_c};border-bottom:1px solid #2a3145;min-width:80px;'>"
            f"{val_str}</td>"
            f"</tr>"
        )

    st.markdown(
        f"<table style='border-collapse:collapse;border-radius:10px;overflow:hidden;"
        f"width:220px;'>{score_rows_html}</table>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 4 – FEATURE F (Percentile Board)
# ══════════════════════════════════════════════════════
st.subheader("📋 Feature F — Team Percentile Board")

sel_team_f = sel_team
t_row_f = df[df["Team"] == sel_team_f]
if t_row_f.empty:
    st.info("Team not found.")
else:
    t_row_f = t_row_f.iloc[0]
    t_league_f = str(t_row_f["League"]) if "League" in t_row_f.index else team_league
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

sel_team_y = sel_team
t_row_y = df[df["Team"] == sel_team_y]
if t_row_y.empty:
    st.info("Team not found.")
else:
    t_row_y = t_row_y.iloc[0]
    t_league_y = str(t_row_y["League"]) if "League" in t_row_y.index else team_league

    _avail_y = sorted(df["League"].dropna().unique())

    comp_y = st.multiselect(
        "Comparison pool", _avail_y,
        default=[t_league_y] if t_league_y in _avail_y else [],
        key=f"ts_y_comp_{sel_team_y}"
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
# ══════════════════════════════════════════════════════
# SECTION 6 – LEADERBOARD
# ══════════════════════════════════════════════════════
import re as _re
from matplotlib.ticker import FuncFormatter as _FuncFormatter

st.subheader("📉 Leaderboard")
st.markdown("---")

with st.expander("Leaderboard settings", expanded=False):
    _lb_metric_opts = [c for c in NUMERIC_COLS if c in df.columns]
    _lb_def = "xG p90" if "xG p90" in _lb_metric_opts else _lb_metric_opts[0]
    lb_metric   = st.selectbox("Metric", _lb_metric_opts, index=_lb_metric_opts.index(_lb_def),
                               format_func=mlabel, key="ts_lb_metric")
    lb_n        = st.slider("Top N", 5, 40, 20, 5, key="ts_lb_n")
    lb_theme    = st.radio("Theme", ["Light","Dark"], horizontal=True, key="ts_lb_theme")

    _lb_pal_opts = [
        "Red–Gold–Green (diverging)",
        "Light-grey → Black",
        "Light-Red → Dark-Red",
        "Light-Blue → Dark-Blue",
        "Light-Green → Dark-Green",
        "Purple ↔ Gold (diverging)",
        "All White", "All Black", "All Red", "All Blue", "All Green",
    ]
    lb_palette     = st.selectbox("Palette", _lb_pal_opts,
                                  index=_lb_pal_opts.index("All Black"), key="ts_lb_palette")
    lb_rev         = st.checkbox("Reverse colours", False, key="ts_lb_rev")
    lb_show_league = st.checkbox("Show league in label", False, key="ts_lb_show_league")
    lb_show_title  = st.checkbox("Show custom title", False, key="ts_lb_show_title")
    lb_custom_title= st.text_input("Custom title", f"Top N – {mlabel(lb_metric)}", key="ts_lb_custom_title")
    lb_highlight   = st.selectbox("Highlight team", ["(None)"] + team_options, key="ts_lb_highlight")

# ── theme colours ──
if lb_theme == "Light":
    _LB_PBG="#ebebeb"; _LB_ABG="#ebebeb"; _LB_TXT="#111111"
    _LB_GRID="#d7d7d7"; _LB_SPINE="#c8c8c8"; _LB_TICK="#111111"
else:
    _LB_PBG="#0a0f1c"; _LB_ABG="#0a0f1c"; _LB_TXT="#f5f5f5"
    _LB_GRID="#3a4050"; _LB_SPINE="#6b7280"; _LB_TICK="#ffffff"

# ── data ──
if lb_metric not in df.columns:
    st.info("Metric not available.")
else:
    _lb_asc = lb_metric in INVERT_METRICS
    _lb_df = df[["Team","League",lb_metric]].dropna(subset=[lb_metric]).copy()
    _lb_df[lb_metric] = pd.to_numeric(_lb_df[lb_metric], errors="coerce")
    _lb_df = _lb_df.dropna().sort_values(lb_metric, ascending=_lb_asc).reset_index(drop=True).head(int(lb_n))

    _lb_vals = _lb_df[lb_metric].values
    if len(_lb_vals) > 1:
        _vmin, _vmax = float(_lb_vals.min()), float(_lb_vals.max())
        if _vmin == _vmax: _vmax = _vmin + 1e-6
        _ts = (_lb_vals - _vmin) / (_vmax - _vmin)
    else:
        _ts = np.zeros(len(_lb_vals))
    if lb_rev: _ts = 1.0 - _ts

    def _lb_cmap(palette, t):
        def _ci(a, b, u): return (np.array(a,float) + (np.array(b,float)-np.array(a,float))*np.clip(u,0,1))/255.0
        t = float(t)
        if palette == "Red–Gold–Green (diverging)":
            return _ci([199,54,60],[240,197,106],t/0.5) if t<=0.5 else _ci([240,197,106],[61,166,91],(t-0.5)/0.5)
        if palette == "Light-grey → Black":        return _ci([210,214,220],[20,23,31],t)
        if palette == "Light-Red → Dark-Red":      return _ci([252,190,190],[139,0,0],t)
        if palette == "Light-Blue → Dark-Blue":    return _ci([191,210,255],[10,42,102],t)
        if palette == "Light-Green → Dark-Green":  return _ci([196,235,203],[12,92,48],t)
        if palette == "Purple ↔ Gold (diverging)":
            return _ci([96,55,140],[180,150,210],t/0.5) if t<=0.5 else _ci([180,150,210],[240,197,106],(t-0.5)/0.5)
        if palette == "All White":  return np.array([255,255,255])/255.0
        if palette == "All Black":  return np.array([0,0,0])/255.0
        if palette == "All Red":    return np.array([197,30,30])/255.0
        if palette == "All Blue":   return np.array([15,70,180])/255.0
        if palette == "All Green":  return np.array([20,120,60])/255.0
        return np.array([0,0,0])/255.0

    _lb_colors = [tuple(_lb_cmap(lb_palette, float(t))) for t in _ts]

    fig_lb, ax_lb = plt.subplots(figsize=(11.5, 6.2))
    fig_lb.patch.set_facecolor(_LB_PBG)
    ax_lb.set_facecolor(_LB_ABG)

    _ypos = np.arange(len(_lb_vals))
    _bars = ax_lb.barh(_ypos, _lb_vals, color=_lb_colors, edgecolor="none", zorder=2)

    # Highlight selected team
    if lb_highlight != "(None)":
        for _bi, (_br, _brow) in enumerate(zip(_bars, _lb_df.itertuples())):
            if _brow.Team == lb_highlight:
                _br.set_color("#f59e0b"); _br.set_edgecolor("white")
                _br.set_linewidth(1.6); _br.set_zorder(5)

    ax_lb.invert_yaxis()
    ax_lb.set_yticks(_ypos)
    _ytlabs = [
        f"{row['Team']},  {row['League']}" if lb_show_league else row["Team"]
        for _, row in _lb_df.iterrows()
    ]
    ax_lb.set_yticklabels(_ytlabs, fontsize=10.5, color=_LB_TXT)
    ax_lb.set_ylabel("")
    ax_lb.set_xlabel(mlabel(lb_metric), color=_LB_TXT, labelpad=6, fontsize=10.5, fontweight="semibold")
    ax_lb.grid(axis="x", color=_LB_GRID, linewidth=0.8, zorder=1)
    for _s in ["top","right","left"]: ax_lb.spines[_s].set_visible(False)
    ax_lb.spines["bottom"].set_color(_LB_SPINE)
    ax_lb.tick_params(axis="y", length=0)

    def _lb_fmt(x, _): return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"
    ax_lb.xaxis.set_major_formatter(_FuncFormatter(_lb_fmt))
    for _tick in ax_lb.get_xticklabels():
        _tick.set_fontweight("medium"); _tick.set_color(_LB_TICK)

    _xmax = float(_lb_vals.max()) * 1.1 if len(_lb_vals) else 1.0
    ax_lb.set_xlim(0, _xmax)
    _pad_lb = (ax_lb.get_xlim()[1] - ax_lb.get_xlim()[0]) * 0.012
    for _rect, _v in zip(_bars, _lb_vals):
        ax_lb.text(_rect.get_width() + _pad_lb, _rect.get_y() + _rect.get_height()/2,
                   _lb_fmt(_v, None), va="center", ha="left", fontsize=8.5, color=_LB_TXT)

    _lb_title = lb_custom_title.strip() if (lb_show_title and lb_custom_title.strip()) \
                else f"Top {len(_lb_df)} – {mlabel(lb_metric)}"
    fig_lb.suptitle(_lb_title, fontsize=26, fontweight="bold", color=_LB_TXT, y=0.985)
    plt.subplots_adjust(top=0.90, left=0.30, right=0.965, bottom=0.14)

    st.pyplot(fig_lb, use_container_width=True)
    _buf_lb = io.BytesIO()
    fig_lb.savefig(_buf_lb, format="png", dpi=200, bbox_inches="tight", facecolor=_LB_PBG)
    st.download_button("⬇️ Download Leaderboard", _buf_lb.getvalue(),
                       f"leaderboard_{lb_metric.replace(' ','_')}.png", "image/png")
    plt.close(fig_lb)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 7 – SCATTER
# ══════════════════════════════════════════════════════
import math as _math
from matplotlib.ticker import MultipleLocator as _MLocator, FormatStrFormatter as _FmtStr
from matplotlib import patheffects as _pe

st.subheader("🔵 Scatter Chart")
st.markdown("---")

_sc_num_cols = [c for c in NUMERIC_COLS if c in df.columns]

with st.expander("Scatter settings", expanded=False):
    _sc_x_def = "xG p90" if "xG p90" in _sc_num_cols else _sc_num_cols[0]
    _sc_y_def = "xG Against p90" if "xG Against p90" in _sc_num_cols else (_sc_num_cols[1] if len(_sc_num_cols)>1 else _sc_num_cols[0])
    sc_x = st.selectbox("X axis", _sc_num_cols, index=_sc_num_cols.index(_sc_x_def),
                        format_func=mlabel, key="ts_sc_x")
    sc_y = st.selectbox("Y axis", _sc_num_cols, index=_sc_num_cols.index(_sc_y_def),
                        format_func=mlabel, key="ts_sc_y")
    sc_colour_metric = st.selectbox("Colour dots by", _sc_num_cols,
                                    index=_sc_num_cols.index(_sc_x_def), format_func=mlabel, key="ts_sc_colour")

    _sc_pal_opts = [
        "Red–Gold–Green (diverging)", "Light-grey → Black",
        "Light-Red → Dark-Red", "Light-Blue → Dark-Blue",
        "Light-Green → Dark-Green", "Purple ↔ Gold (diverging)",
        "All White", "All Black",
    ]
    sc_palette  = st.selectbox("Palette", _sc_pal_opts, index=_sc_pal_opts.index("All Black"), key="ts_sc_palette")
    sc_rev      = st.checkbox("Reverse colours", False, key="ts_sc_rev")
    sc_theme    = st.radio("Theme", ["Light","Dark"], horizontal=True, key="ts_sc_theme")

    sc_show_labels  = st.toggle("Show team labels", True, key="ts_sc_labels")
    sc_hl_team      = st.selectbox("Highlight team", ["(None)"] + team_options, key="ts_sc_hl")
    sc_show_medians = st.checkbox("Show median lines", True, key="ts_sc_medians")
    sc_shade_iqr    = st.checkbox("Shade IQR (25–75%)", True, key="ts_sc_iqr")
    sc_point_size   = st.slider("Point size", 24, 300, 200, key="ts_sc_size")
    sc_alpha        = st.slider("Point opacity", 0.2, 1.0, 0.88, 0.02, key="ts_sc_alpha")
    sc_marker       = st.selectbox("Marker", ["o","s","^","D"], key="ts_sc_marker")
    sc_tick_mode    = st.selectbox("Tick spacing", ["Auto","0.05","0.1","0.2","0.5","1.0"], key="ts_sc_tick")
    sc_canvas       = st.selectbox("Canvas size", ["1280×720","1600×900","1920×820","1920×1080"], index=1, key="ts_sc_canvas")
    sc_show_title   = st.checkbox("Show custom title", False, key="ts_sc_show_title")
    sc_custom_title = st.text_input("Custom title", f"{mlabel(sc_x)} vs {mlabel(sc_y)}", key="ts_sc_title")
    sc_top_gap      = st.slider("Top gap (px)", 0, 240, 80, 5, key="ts_sc_topgap")
    sc_render_exact = st.checkbox("Render exact pixels (PNG)", True, key="ts_sc_exact")

    # Include selected team (from Team Profile)
    sc_include_sel  = st.toggle(f"Highlight selected team ({sel_team})", True, key="ts_sc_incl_sel")

_sc_w, _sc_h = map(int, sc_canvas.replace("×","x").replace(" ","").split("x"))

# ── theme ──
_SC_PBG  = "#ebebeb" if sc_theme=="Light" else "#0a0f1c"
_SC_ABG  = "#f3f3f3" if sc_theme=="Light" else "#0f151f"
_SC_GRID = "#d7d7d7" if sc_theme=="Light" else "#3a4050"
_SC_TXT  = "#111111" if sc_theme=="Light" else "#f5f5f5"
_SC_STR  = "#ffffff" if sc_theme=="Light" else "#1e293b"

def _sc_nice_step(vmin, vmax, target=6):
    span = abs(vmax-vmin)
    if span<=0 or not _math.isfinite(span): return 1.0
    raw = span/max(target,2); pw = 10**_math.floor(_math.log10(raw)); m = raw/pw
    if m<=1: k=1
    elif m<=2: k=2
    elif m<=2.5: k=2.5
    elif m<=5: k=5
    else: k=10
    return k*pw

def _sc_pad_lims(arr, pad=0.06, head=0.03):
    a,b = float(np.nanmin(arr)), float(np.nanmax(arr))
    if a==b: a-=1e-6; b+=1e-6
    sp = b-a; pad_ = sp*pad
    return a-pad_, b+pad_+sp*head

def _sc_map_colors(cvals, palette, rev):
    cmin,cmax = float(np.nanmin(cvals)), float(np.nanmax(cvals))
    if cmin==cmax: cmax=cmin+1e-6
    t = (cvals-cmin)/(cmax-cmin)
    if rev: t = 1.0-t
    def _ci(a,b,u): return (np.array(a,float)+(np.array(b,float)-np.array(a,float))*np.clip(u,0,1))/255.0
    def _mc(palette, v):
        if palette=="Red–Gold–Green (diverging)":
            return _ci([199,54,60],[240,197,106],v/0.5) if v<=0.5 else _ci([240,197,106],[61,166,91],(v-0.5)/0.5)
        if palette=="Light-grey → Black":        return _ci([210,214,220],[20,23,31],v)
        if palette=="Light-Red → Dark-Red":      return _ci([252,190,190],[139,0,0],v)
        if palette=="Light-Blue → Dark-Blue":    return _ci([191,210,255],[10,42,102],v)
        if palette=="Light-Green → Dark-Green":  return _ci([196,235,203],[12,92,48],v)
        if palette=="Purple ↔ Gold (diverging)":
            return _ci([96,55,140],[180,150,210],v/0.5) if v<=0.5 else _ci([180,150,210],[240,197,106],(v-0.5)/0.5)
        if palette=="All White": return np.array([255,255,255])/255.0
        return np.array([0,0,0])/255.0
    return [tuple(_mc(palette, float(v))) for v in t]

_sc_pool = df[sc_x].notna() & df[sc_y].notna() & df[sc_colour_metric].notna()
_sc_df   = df[_sc_pool].copy()
for _c in [sc_x, sc_y, sc_colour_metric]:
    _sc_df[_c] = pd.to_numeric(_sc_df[_c], errors="coerce")
_sc_df = _sc_df.dropna(subset=[sc_x, sc_y, sc_colour_metric, "Team"])

if _sc_df.empty:
    st.info("No data for selected metrics.")
else:
    try:
        _x_vals = _sc_df[sc_x].to_numpy(float)
        _y_vals = _sc_df[sc_y].to_numpy(float)
        _xlim   = _sc_pad_lims(_x_vals)
        _ylim   = _sc_pad_lims(_y_vals)
        _cvals  = _sc_df[sc_colour_metric].to_numpy(float)
        _cols   = _sc_map_colors(_cvals, sc_palette, sc_rev)
        _col_s  = pd.Series(_cols, index=_sc_df.index)

        _sel_name = sel_team
        _others   = _sc_df[_sc_df["Team"] != _sel_name] if sc_include_sel else _sc_df
        _sel_df   = _sc_df[_sc_df["Team"] == _sel_name] if sc_include_sel else _sc_df.iloc[0:0]

        plt.rcParams.update({"figure.dpi":100,"savefig.dpi":100,"text.antialiased":True})
        fig_sc, ax_sc = plt.subplots(figsize=(_sc_w/100, _sc_h/100), dpi=100)
        fig_sc.patch.set_facecolor(_SC_PBG)
        ax_sc.set_facecolor(_SC_ABG)
        ax_sc.set_xlim(*_xlim); ax_sc.set_ylim(*_ylim)

        # All other teams
        ax_sc.scatter(_others[sc_x], _others[sc_y], s=sc_point_size,
                      c=list(_col_s.loc[_others.index]), alpha=float(sc_alpha),
                      edgecolors="none", marker=sc_marker, zorder=2)

        # Selected team (red outlined)
        if not _sel_df.empty:
            ax_sc.scatter(_sel_df[sc_x], _sel_df[sc_y], s=sc_point_size,
                          c="#C81E1E", edgecolors="white", linewidths=1.8,
                          marker=sc_marker, zorder=4)

        # Highlighted team (amber)
        if sc_hl_team != "(None)":
            _hl = _sc_df[_sc_df["Team"] == sc_hl_team]
            if not _hl.empty:
                ax_sc.scatter(_hl[sc_x], _hl[sc_y], s=sc_point_size,
                              c="#f59e0b", edgecolors="white", linewidths=1.6,
                              marker=sc_marker, zorder=5)

        # IQR shading
        if sc_shade_iqr:
            _xq1,_xq3 = np.nanpercentile(_x_vals,[25,75])
            _yq1,_yq3 = np.nanpercentile(_y_vals,[25,75])
            _iqr_c = "#cfd3da" if sc_theme=="Light" else "#9aa4b1"
            ax_sc.axvspan(_xq1,_xq3,color=_iqr_c,alpha=0.25,zorder=1)
            ax_sc.axhspan(_yq1,_yq3,color=_iqr_c,alpha=0.25,zorder=1)

        # Medians
        if sc_show_medians:
            _mc2 = "#000000" if sc_theme=="Light" else "#ffffff"
            ax_sc.axvline(float(np.nanmedian(_x_vals)),color=_mc2,ls=(0,(4,4)),lw=2.2,zorder=3)
            ax_sc.axhline(float(np.nanmedian(_y_vals)),color=_mc2,ls=(0,(4,4)),lw=2.2,zorder=3)

        # Axes
        ax_sc.set_xlabel(mlabel(sc_x), fontsize=14, fontweight="semibold", color=_SC_TXT)
        ax_sc.set_ylabel(mlabel(sc_y), fontsize=14, fontweight="semibold", color=_SC_TXT)

        if sc_tick_mode == "Auto":
            _sx = _sc_nice_step(*_xlim, target=12)
            _sy = _sc_nice_step(*_ylim, target=12)
        else:
            _sx = _sy = float(sc_tick_mode)
        ax_sc.xaxis.set_major_locator(_MLocator(base=_sx))
        ax_sc.yaxis.set_major_locator(_MLocator(base=_sy))
        def _sc_dec(s): return 0 if s>=1 else (1 if s>=0.1 else (2 if s>=0.01 else 3))
        ax_sc.xaxis.set_major_formatter(_FmtStr(f"%.{_sc_dec(_sx)}f"))
        ax_sc.yaxis.set_major_formatter(_FmtStr(f"%.{_sc_dec(_sy)}f"))
        ax_sc.minorticks_off()
        for _t in ax_sc.get_xticklabels()+ax_sc.get_yticklabels():
            _t.set_fontweight("semibold"); _t.set_color(_SC_TXT)
        ax_sc.grid(True, which="major", linewidth=0.9, color=_SC_GRID)
        for _s in ax_sc.spines.values():
            _s.set_linewidth(0.9)
            _s.set_color("#9ca3af" if sc_theme=="Light" else "#6b7280")

        # Top gap
        _tgap_px = 75 if sc_show_title else sc_top_gap
        _top_frac = 1.0 - (_tgap_px / float(_sc_h))
        fig_sc.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=_top_frac)

        # Custom title
        if sc_show_title and sc_custom_title.strip():
            _tc = "#111111" if sc_theme=="Light" else "#f5f5f5"
            fig_sc.text(0.5, _top_frac+(1-_top_frac)*0.44, sc_custom_title.strip(),
                        ha="center", va="center", color=_tc, fontsize=26, fontweight="semibold")

        # Labels
        if sc_show_labels:
            for _, _lr in _sc_df.iterrows():
                _lab = ax_sc.annotate(
                    str(_lr["Team"]), (float(_lr[sc_x]), float(_lr[sc_y])),
                    xytext=(8,10), textcoords="offset points",
                    fontsize=9, fontweight="semibold", color=_SC_TXT,
                    ha="left", va="bottom", zorder=6
                )
                _lab.set_path_effects([_pe.withStroke(linewidth=2.0, foreground=_SC_STR, alpha=0.9)])

        if sc_render_exact:
            _buf_sc = io.BytesIO()
            fig_sc.savefig(_buf_sc, format="png", dpi=100, facecolor=fig_sc.get_facecolor(),
                           bbox_inches=None, pad_inches=0)
            _buf_sc.seek(0)
            st.image(_buf_sc, width=_sc_w)
        else:
            st.pyplot(fig_sc, use_container_width=False)

        _buf_sc2 = io.BytesIO()
        fig_sc.savefig(_buf_sc2, format="png", dpi=150, facecolor=_SC_PBG, bbox_inches=None)
        st.download_button("⬇️ Download Scatter", _buf_sc2.getvalue(),
                           f"scatter_{mlabel(sc_x)}_vs_{mlabel(sc_y)}.png".replace(" ","_"), "image/png")
        plt.close(fig_sc)
    except Exception as _e:
        st.info(f"Scatter error: {_e}")

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 8 – COMPARISON RADAR
# ══════════════════════════════════════════════════════
st.subheader("⚡ Team Comparison Radar")
st.markdown("---")

RADAR_COMP_METRICS = [
    "xG p90","Goals p90","Touches in Box p90",
    "xG Against p90","Goals Against p90","PPDA",
    "Possession %","Passes p90","Passes to Final Third p90",
    "Long Passes p90","Points p90","Expected Points p90",
]

# Compute per-game versions of Points & xPoints if raw columns exist
for _src, _dst in [("Points","Points p90"),("Expected Points","Expected Points p90")]:
    if _src in df.columns and "Matches" in df.columns and _dst not in df.columns:
        _matches = pd.to_numeric(df["Matches"], errors="coerce").replace(0, np.nan)
        df[_dst] = pd.to_numeric(df[_src], errors="coerce") / _matches
    elif _src in df.columns and _dst not in df.columns:
        df[_dst] = pd.to_numeric(df[_src], errors="coerce")

radar_comp_avail = [m for m in RADAR_COMP_METRICS if m in df.columns]

with st.expander("Radar settings", expanded=False):
    comp_theme  = st.radio("Theme", ["Light","Dark"], horizontal=True, key="ts_comp_theme")
    # Team A defaults to sel_team from Team Profile
    _comp_a_idx = team_options.index(sel_team) if sel_team in team_options else 0
    _comp_b_opts= [t for t in team_options if t != sel_team]
    comp_team_a_label = st.text_input("Edit Team A label", sel_team, key="ts_comp_a_label")
    comp_team_b_label_default = _comp_b_opts[0] if _comp_b_opts else ""
    comp_team_b_sel   = st.selectbox("Team B (blue)", _comp_b_opts, key="ts_comp_b")
    comp_team_b_label = st.text_input("Edit Team B label", comp_team_b_sel, key="ts_comp_b_label")
    comp_show_title   = st.checkbox("Show custom title", False, key="ts_comp_show_title")
    comp_custom_title = st.text_input("Custom title", "", key="ts_comp_custom_title")

# Theme colours (matching player radar exactly)
if comp_theme == "Dark":
    _CR_PBG="#0a0f1c"; _CR_AX="#0a0f1c"
    _CR_BAND_OUT="#162235"; _CR_BAND_IN="#0d1524"
    _CR_RING_IN="#3a4050"; _CR_RING_OUT="#cbd5e1"
    _CR_LABEL="#f5f5f5"; _CR_TICK="#e5e7eb"; _CR_MINS="#f5f5f5"
else:
    _CR_PBG="#ffffff"; _CR_AX="#ebebeb"
    _CR_BAND_OUT="#e5e7eb"; _CR_BAND_IN="#ffffff"
    _CR_RING_IN="#d1d5db"; _CR_RING_OUT="#d1d5db"
    _CR_LABEL="#0f172a"; _CR_TICK="#6b7280"; _CR_MINS="#374151"

team_a = sel_team
team_b = comp_team_b_sel

row_a = df[df["Team"]==team_a].iloc[0] if not df[df["Team"]==team_a].empty else None
row_b = df[df["Team"]==team_b].iloc[0] if not df[df["Team"]==team_b].empty else None

if row_a is not None and row_b is not None and radar_comp_avail:
    _leagues_ab = {str(row_a.get("League","")), str(row_b.get("League",""))}
    _pool_ab    = df[df["League"].isin(_leagues_ab)].copy()
    for _m in radar_comp_avail:
        _pool_ab[_m] = pd.to_numeric(_pool_ab[_m], errors="coerce")

    def _pct_ab(t_row, col):
        inv = col in INVERT_METRICS
        if col not in _pool_ab.columns: return 50.0
        s = pd.to_numeric(_pool_ab[col], errors="coerce").dropna()
        v = float(t_row[col]) if pd.notna(t_row.get(col)) else np.nan
        if pd.isna(v) or s.empty: return 50.0
        p = (s<v).mean()*100 + (s==v).mean()*50
        return float(np.clip((100-p) if inv else p, 0, 100))

    def _actual_val(t_row, col):
        """Return actual value formatted to 2dp for display on ring ticks."""
        v = t_row.get(col, np.nan) if hasattr(t_row, 'get') else t_row[col]
        try: v = float(v)
        except: return "—"
        return f"{v:.2f}".rstrip("0").rstrip(".")

    A_r = np.array([_pct_ab(row_a, m) for m in radar_comp_avail])
    B_r = np.array([_pct_ab(row_b, m) for m in radar_comp_avail])

    # Actual values for tick labels (show real values, plotted by percentile)
    A_actual = [_actual_val(row_a, m) for m in radar_comp_avail]
    B_actual = [_actual_val(row_b, m) for m in radar_comp_avail]

    _labels_c = [mlabel(m) for m in radar_comp_avail]
    _N_c      = len(radar_comp_avail)
    _theta_c  = np.linspace(0, 2*np.pi, _N_c, endpoint=False)
    _theta_cc = np.concatenate([_theta_c, _theta_c[:1]])
    _Ar_c     = np.concatenate([A_r, A_r[:1]])
    _Br_c     = np.concatenate([B_r, B_r[:1]])

    _COL_A="#C81E1E"; _COL_B="#1D4ED8"
    _FILL_A=(200/255,30/255,30/255,0.60); _FILL_B=(29/255,78/255,216/255,0.60)
    _INNER=10; _RING_LW=1.0; _OUTER_R=107

    # TRUE decile values per metric (for ring tick labels)
    _qs = np.linspace(0,100,11)
    _axis_ticks = [np.nanpercentile(_pool_ab[m].dropna().values, _qs) for m in radar_comp_avail]

    fig_c = plt.figure(figsize=(13.2, 8.0), dpi=260)
    fig_c.patch.set_facecolor(_CR_PBG)
    ax_c = plt.subplot(111, polar=True)
    ax_c.set_facecolor(_CR_AX)
    ax_c.set_theta_offset(np.pi/2); ax_c.set_theta_direction(-1)
    ax_c.set_xticks(_theta_c); ax_c.set_xticklabels([])
    ax_c.set_yticks([]); ax_c.grid(False)
    for _s in ax_c.spines.values(): _s.set_visible(False)

    # Alternating radial bands
    _ring_edges = np.linspace(_INNER, 100, 11)
    for _i in range(10):
        _r0,_r1 = _ring_edges[_i], _ring_edges[_i+1]
        _band = _CR_BAND_OUT if (9-_i)%2==0 else _CR_BAND_IN
        ax_c.add_artist(mpatches.Wedge((0,0),_r1,0,360,width=(_r1-_r0),
            transform=ax_c.transData._b,facecolor=_band,edgecolor="none",zorder=0.8))

    # Ring outlines
    _ring_t = np.linspace(0, 2*np.pi, 361)
    for _j, _r in enumerate(_ring_edges):
        _rc = _CR_RING_OUT if _j==len(_ring_edges)-1 else _CR_RING_IN
        ax_c.plot(_ring_t, np.full_like(_ring_t,_r), color=_rc, lw=_RING_LW, zorder=0.9)

    # Actual value tick labels at each ring (decile values from pool)
    for _i, _ang in enumerate(_theta_c):
        _tv = _axis_ticks[_i]
        for _rr, _v in zip(_ring_edges[2:], _tv[2:]):
            ax_c.text(_ang, _rr-1.8, f"{float(_v):.1f}",
                      ha="center", va="center", fontsize=7, color=_CR_TICK, zorder=1.1)

    # Outer metric labels (upright, centered)
    for _ang, _lab in zip(_theta_c, _labels_c):
        _rot = np.degrees(ax_c.get_theta_direction()*_ang + ax_c.get_theta_offset()) - 90.0
        _rn  = ((_rot+180)%360)-180
        if _rn>90 or _rn<-90: _rot+=180
        ax_c.text(_ang, _OUTER_R, _lab, rotation=_rot, rotation_mode="anchor",
                  ha="center", va="center", fontsize=9, color=_CR_LABEL,
                  fontweight=600, clip_on=False, zorder=2.2)

    # Center hole
    ax_c.add_artist(plt.Circle((0,0), radius=_INNER-0.6, transform=ax_c.transData._b,
                               color=_CR_PBG, zorder=1.2, ec="none"))

    # Polygons
    ax_c.plot(_theta_cc, _Ar_c, color=_COL_A, lw=2.2, zorder=3)
    ax_c.fill(_theta_cc, _Ar_c, color=_FILL_A, zorder=2.5)
    ax_c.plot(_theta_cc, _Br_c, color=_COL_B, lw=2.2, zorder=3)
    ax_c.fill(_theta_cc, _Br_c, color=_FILL_B, zorder=2.5)
    ax_c.set_rlim(0, 100)

    # Actual value dots on each team's polygon
    for _i, (_ang, _av_a, _av_b) in enumerate(zip(_theta_c, A_actual, B_actual)):
        _pa = A_r[_i]; _pb = B_r[_i]
        ax_c.plot(_ang, _pa, 'o', color=_COL_A, markersize=4, zorder=4)
        ax_c.plot(_ang, _pb, 'o', color=_COL_B, markersize=4, zorder=4)

    # Headers (editable labels)
    _sub_a = f"{row_a.get('League','') if hasattr(row_a,'get') else row_a['League']}"
    _sub_b = f"{row_b.get('League','') if hasattr(row_b,'get') else row_b['League']}"
    _ppg_a = f"{float(row_a.get('Points p90', row_a.get('Points','?'))):.2f} pts/g" if "Points p90" in df.columns else ""
    _ppg_b = f"{float(row_b.get('Points p90', row_b.get('Points','?'))):.2f} pts/g" if "Points p90" in df.columns else ""

    fig_c.text(0.12, 0.96,  comp_team_a_label, color=_COL_A, fontsize=22, fontweight="bold",  ha="left")
    fig_c.text(0.12, 0.935, _sub_a,            color=_COL_A, fontsize=11, ha="left")
    fig_c.text(0.12, 0.915, _ppg_a,            color=_CR_MINS, fontsize=10, ha="left")
    fig_c.text(0.88, 0.96,  comp_team_b_label, color=_COL_B, fontsize=22, fontweight="bold",  ha="right")
    fig_c.text(0.88, 0.935, _sub_b,            color=_COL_B, fontsize=11, ha="right")
    fig_c.text(0.88, 0.915, _ppg_b,            color=_CR_MINS, fontsize=10, ha="right")

    if comp_show_title and comp_custom_title.strip():
        fig_c.text(0.5, 0.985, comp_custom_title.strip(), ha="center", fontsize=16,
                   fontweight="bold", color=_CR_LABEL)

    st.caption("Ring labels show **actual dataset values at each decile** (20th–100th percentile). "
               "Polygons are plotted by **percentile rank** vs the combined league pool.")
    st.pyplot(fig_c, use_container_width=True)
    _buf_c = io.BytesIO()
    fig_c.savefig(_buf_c, format="png", dpi=220, facecolor=_CR_PBG)
    st.download_button("⬇️ Download Comparison Radar", _buf_c.getvalue(),
                       f"comparison_{team_a.replace(' ','_')}_vs_{team_b.replace(' ','_')}.png", "image/png")
    plt.close(fig_c)
else:
    st.info("Select two teams with available data.")

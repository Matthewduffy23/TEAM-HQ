# team_hq.py — Team Statistics Dashboard  PART 1/2
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
from matplotlib.transforms import ScaledTranslation
from matplotlib import patheffects as pe
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
    "League":                       ["league"],
    "Team":                         ["team"],
    "Matches":                      ["matches"],
    "Wins":                         ["wins"],
    "Draws":                        ["draws"],
    "Losses":                       ["losses"],
    "Points":                       ["points"],
    "Expected Points":              ["expected points","xpoints","x points","expected_points"],
    "Goals For":                    ["goals for","goals scored","goals_for"],
    "Goals Against":                ["goals against","goals conceded","goals_against"],
    "Goal Difference":              ["goal difference","goal diff","goal_difference"],
    "Avg Age":                      ["avg age","average age","avg_age"],
    "Possession %":                 ["possession %","possession","possession_pct"],
    "Goals p90":                    ["goals p90","goals per 90","goals_p90"],
    "xG p90":                       ["xg p90","xg per 90","xg_p90"],
    "Shots p90":                    ["shots p90","shots per 90","shots_p90"],
    "Shot Accuracy %":              ["shot accuracy %","shots on target %","shot_accuracy_pct","shooting %"],
    "Crosses p90":                  ["crosses p90","crosses per 90","crosses_p90"],
    "Dribbles p90":                 ["dribbles p90","dribbles per 90","dribbles_p90"],
    "Touches in Box p90":           ["touches in box p90","touches in box per 90","touches_in_box_p90"],
    "Shots Against p90":            ["shots against p90","shots vs p90","shots_against_p90"],
    "Defensive Duels p90":          ["defensive duels p90","defensive_duels_p90"],
    "Defensive Duels Won %":        ["defensive duels won %","defensive_duels_won_pct","def duels won %"],
    "Aerial Duels p90":             ["aerial duels p90","aerial_duels_p90"],
    "Aerial Duels Won %":           ["aerial duels won %","aerial_duels_won_pct"],
    "PPDA":                         ["ppda"],
    "Passes p90":                   ["passes p90","passes per 90","passes_p90"],
    "Pass Accuracy %":              ["pass accuracy %","passing accuracy %","pass_accuracy_pct","accurate passes %"],
    "Through Passes p90":           ["through passes p90","through_passes_p90"],
    "Passes to Final Third p90":    ["passes to final third p90","passes_to_final_third_p90","passes to final 3rd p90"],
    "Passes to Final Third Acc %":  ["passes to final third acc %","passes_to_final_third_acc_pct"],
    "Long Passes p90":              ["long passes p90","long_passes_p90"],
    "Long Pass Accuracy %":         ["long pass accuracy %","long_pass_accuracy_pct"],
    "Progressive Passes p90":       ["progressive passes p90","progressive_passes_p90"],
    "Progressive Runs p90":         ["progressive runs p90","progressive_runs_p90"],
    "xG Against p90":               ["xg against p90","xga p90","xg_against_p90","xg against"],
    "Goals Against p90":            ["goals against p90","goals conceded p90","goals_against_p90"],
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

# ─────────────────────────────────────────────
# METRIC DISPLAY LABELS
# ─────────────────────────────────────────────
METRIC_LABELS = {
    "Crosses p90":                  "Crosses",
    "Shot Accuracy %":              "Shooting %",
    "Goals p90":                    "Goals Scored",
    "xG p90":                       "xG",
    "Shots p90":                    "Shots",
    "Touches in Box p90":           "Touches in Box",
    "Aerial Duels Won %":           "Aerial Duel Success %",
    "Aerial Duels p90":             "Aerial Duels",
    "Goals Against p90":            "Goals Conceded",
    "xG Against p90":               "xG Against",
    "Defensive Duels p90":          "Defensive Duels",
    "Defensive Duels Won %":        "Defensive Duel Win %",
    "Shots Against p90":            "Shots Against",
    "PPDA":                         "PPDA",
    "Dribbles p90":                 "Dribbles",
    "Passes p90":                   "Passes",
    "Pass Accuracy %":              "Passing Accuracy %",
    "Long Passes p90":              "Long Passes",
    "Long Pass Accuracy %":         "Long Passing %",
    "Possession %":                 "Possession",
    "Passes to Final Third p90":    "Passes to Final 3rd",
    "Passes to Final Third Acc %":  "Passes to Final 3rd %",
    "Progressive Passes p90":       "Progressive Passes",
    "Progressive Runs p90":         "Progressive Runs",
    "Expected Points":              "xPoints",
    "Points":                       "Points",
    "Goals For":                    "Goals For",
    "Goals Against":                "Goals Against",
    "Matches":                      "Matches",
}
def mlabel(col): return METRIC_LABELS.get(col, col)

# ─────────────────────────────────────────────
# INVERT METRICS
# ─────────────────────────────────────────────
INVERT_METRICS = {"xG Against p90","Goals Against p90","Shots Against p90","PPDA","Goals Against"}

# ─────────────────────────────────────────────
# REGION / PRESET MAPS
# ─────────────────────────────────────────────
PRESET_LEAGUES = {
    "Top 5 Europe":  {"England 1","Spain 1","Germany 1","Italy 1","France 1"},
    "Top 10 Europe": {"England 1","Spain 1","Germany 1","Italy 1","France 1",
                      "Netherlands 1","Portugal 1","Belgium 1","Turkey 1","England 2"},
    "EFL":           {"England 2","England 3","England 4"},
    "Australia":     {"Australia 1"},
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
def league_country(lg): return re.sub(r"\s*\d+\s*$","",str(lg)).strip()
def league_region(lg):  return COUNTRY_TO_REGION.get(league_country(lg),"Other")

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Filters")
    all_leagues = sorted(df_raw["League"].dropna().unique().tolist()) if "League" in df_raw.columns else []
    all_regions = sorted({league_region(lg) for lg in all_leagues})
    sel_regions = st.multiselect("Regions", all_regions, default=all_regions, key="ts_regions")

    st.markdown("#### League Presets")
    pc1,pc2,pc3 = st.columns(3)
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

    preset_sig = (tuple(sorted(sel_regions)),use_top5,use_top10,use_efl,use_aus)
    if st.session_state.get("ts_preset_sig") != preset_sig:
        st.session_state["ts_preset_sig"] = preset_sig
        st.session_state["ts_leagues_sel"] = default_leagues

    leagues_sel = st.multiselect("Leagues", region_leagues,
                                 default=st.session_state.get("ts_leagues_sel",default_leagues),
                                 key="ts_leagues_sel")

    st.markdown("---")
    st.markdown("#### Metric Filter")
    avail_numeric = [c for c in NUMERIC_COLS if c in df_raw.columns]
    metric_filter_col = st.selectbox("Filter by metric",["None"]+avail_numeric,
                                     format_func=lambda x:"None" if x=="None" else mlabel(x),
                                     key="ts_metric_filter_col")
    if metric_filter_col != "None":
        filter_mode = st.radio("Filter mode",["Raw value","Percentile"],horizontal=True,key="ts_filter_mode")
        if filter_mode == "Raw value":
            col_min = float(df_raw[metric_filter_col].min()) if metric_filter_col in df_raw.columns else 0.0
            col_max = float(df_raw[metric_filter_col].max()) if metric_filter_col in df_raw.columns else 100.0
            metric_min_val = st.slider(f"Min {mlabel(metric_filter_col)}",col_min,col_max,col_min,key="ts_metric_raw_min")
        else:
            metric_min_pct = st.slider("Min percentile",0,100,0,key="ts_metric_pct_min")

    st.markdown("---")
    st.markdown("#### Score Filter")
    score_filter_type = st.selectbox("Filter by score",["None","Overall","Attack","Defense","Possession"],key="ts_score_filter")
    score_threshold   = st.slider("Min percentile score",0,100,0,key="ts_score_thresh")

# ─────────────────────────────────────────────
# APPLY LEAGUE FILTER
# ─────────────────────────────────────────────
df = df_raw[df_raw["League"].isin(leagues_sel)].copy() if leagues_sel else df_raw.copy()
if df.empty:
    st.warning("No teams match current filters."); st.stop()

# ─────────────────────────────────────────────
# PERCENTILE RANK HELPER
# ─────────────────────────────────────────────
def pct_rank(series, invert=False):
    r = series.rank(pct=True)*100
    return 100-r if invert else r

for col in NUMERIC_COLS:
    if col not in df.columns: continue
    inv = col in INVERT_METRICS
    df[f"_pct_{col}"] = df.groupby("League")[col].transform(lambda s,i=inv: pct_rank(s,i))

def score_col(name): return f"_pct_{name}"

def compute_overall(row):
    ep  = row.get(score_col("Expected Points"),np.nan)
    xg  = row.get(score_col("xG p90"),np.nan)
    xga = row.get(score_col("xG Against p90"),np.nan)
    vals= [v for v in [ep,xg,xga] if pd.notna(v)]
    if not vals: return np.nan
    w=[0.5,0.25,0.25][:len(vals)]; tw=sum(w)
    return sum(v*ww for v,ww in zip(vals,w))/tw

def compute_attack(row):
    weights=[(row.get(score_col("xG p90"),np.nan),0.5),(row.get(score_col("Goals p90"),np.nan),0.3),
             (row.get(score_col("Shots p90"),np.nan),0.05),(row.get(score_col("Touches in Box p90"),np.nan),0.15)]
    vals=[(v,w) for v,w in weights if pd.notna(v)]
    if not vals: return np.nan
    tw=sum(w for _,w in vals); return sum(v*w for v,w in vals)/tw

def compute_defense(row):
    weights=[(row.get(score_col("xG Against p90"),np.nan),0.5),(row.get(score_col("Goals Against p90"),np.nan),0.3),
             (row.get(score_col("Shots Against p90"),np.nan),0.2)]
    vals=[(v,w) for v,w in weights if pd.notna(v)]
    if not vals: return np.nan
    tw=sum(w for _,w in vals); return sum(v*w for v,w in vals)/tw

def compute_possession(row):
    weights=[(row.get(score_col("Possession %"),np.nan),0.35),(row.get(score_col("Passes p90"),np.nan),0.30),
             (row.get(score_col("Pass Accuracy %"),np.nan),0.10),(row.get(score_col("Passes to Final Third p90"),np.nan),0.25)]
    vals=[(v,w) for v,w in weights if pd.notna(v)]
    if not vals: return np.nan
    tw=sum(w for _,w in vals); return sum(v*w for v,w in vals)/tw

df["OVR"] = df.apply(compute_overall,   axis=1)
df["ATT"] = df.apply(compute_attack,    axis=1)
df["DEF"] = df.apply(compute_defense,   axis=1)
df["POS"] = df.apply(compute_possession,axis=1)

if score_filter_type != "None" and score_threshold > 0:
    scol = {"Overall":"OVR","Attack":"ATT","Defense":"DEF","Possession":"POS"}[score_filter_type]
    df = df[df[scol] >= score_threshold]

if metric_filter_col != "None" and metric_filter_col in df.columns:
    if filter_mode == "Raw value":
        df = df[pd.to_numeric(df[metric_filter_col],errors="coerce") >= metric_min_val]
    else:
        _pc = f"_pct_{metric_filter_col}"
        if _pc in df.columns: df = df[df[_pc] >= metric_min_pct]

if df.empty:
    st.warning("No teams after filters. Adjust thresholds."); st.stop()

# ─────────────────────────────────────────────
# SORT / RANKING
# ─────────────────────────────────────────────
rank_by = st.radio("Sort teams by",["Overall (OVR)","Attack (ATT)","Defense (DEF)","Possession (POS)","Raw metric"],
                   horizontal=True,key="ts_rank_by")
raw_metric_options = [c for c in NUMERIC_COLS if c in df.columns]
if rank_by == "Raw metric":
    raw_pick = st.selectbox("Raw metric",raw_metric_options,format_func=mlabel,key="ts_raw_pick")
    asc = raw_pick in INVERT_METRICS; df["_sort"] = df[raw_pick]
else:
    sort_col = {"Overall (OVR)":"OVR","Attack (ATT)":"ATT","Defense (DEF)":"DEF","Possession (POS)":"POS"}[rank_by]
    asc=False; df["_sort"] = df[sort_col]

df_sorted = df.dropna(subset=["_sort"]).sort_values("_sort",ascending=asc).reset_index(drop=True)

display_league_filter = st.selectbox("Display league (does not change pool)",
    ["All leagues"]+sorted(df["League"].dropna().unique().tolist()),key="ts_disp_league")
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

def fotmob_crest_url(team):
    raw = (_FOTMOB_URLS.get(team) or "").strip()
    if not raw: return ""
    m = re.search(r"/teams/(\d+)/",raw)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{m.group(1)}.png" if m else ""

TWEMOJI_SPECIAL = {
    "eng":"1f3f4-e0067-e0062-e0065-e006e-e0067-e007f",
    "sct":"1f3f4-e0067-e0062-e0073-e0063-e0074-e007f",
    "wls":"1f3f4-e0067-e0062-e0077-e006c-e0073-e007f",
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

def flag_html(league_name):
    country=league_country(league_name); n=_norm(country); cc=COUNTRY_TO_CC.get(n,"")
    if not cc: return ""
    if cc in TWEMOJI_SPECIAL: code=TWEMOJI_SPECIAL[cc]
    else:
        if len(cc)!=2: return ""
        base=0x1F1E6; code=f"{base+(ord(cc[0].upper())-65):x}-{base+(ord(cc[1].upper())-65):x}"
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
    v = row.get(f"_pct_{col}",np.nan)
    return float(v) if pd.notna(v) else 0.0

def metric_val(row, col):
    v = row.get(col,np.nan)
    if pd.isna(v): return "—"
    return f"{float(v):.2f}".rstrip("0").rstrip(".")

# ─────────────────────────────────────────────
# LEAGUE POSITION HELPER
# ─────────────────────────────────────────────
def get_league_position(team_name, sort_metric="Points"):
    t_row = df[df["Team"]==team_name]
    if t_row.empty: return (None,None)
    league = t_row.iloc[0].get("League","")
    league_df = df[df["League"]==league].copy()
    total = len(league_df)
    if sort_metric in league_df.columns:
        league_df = league_df.sort_values(sort_metric,ascending=False).reset_index(drop=True)
    else:
        return (None,total)
    idx = league_df[league_df["Team"]==team_name].index.tolist()
    return (idx[0]+1 if idx else None, total)

def get_xpoints_position(team_name):
    t_row = df[df["Team"]==team_name]
    if t_row.empty: return (None,None)
    league = t_row.iloc[0].get("League","")
    league_df = df[df["League"]==league].copy()
    total = len(league_df)
    if "Expected Points" not in league_df.columns: return (None,total)
    league_df = league_df.sort_values("Expected Points",ascending=False).reset_index(drop=True)
    idx = league_df[league_df["Team"]==team_name].index.tolist()
    return (idx[0]+1 if idx else None, total)

# ─────────────────────────────────────────────
# METRIC SECTION DEFINITIONS
# ─────────────────────────────────────────────
TEAM_METRICS_ATT = [
    ("Crosses",           "Crosses p90"),
    ("Goals Scored",      "Goals p90"),
    ("xG",                "xG p90"),
    ("Shots",             "Shots p90"),
    ("Shooting %",        "Shot Accuracy %"),
    ("Touches in Box",    "Touches in Box p90"),
]
TEAM_METRICS_DEF = [
    ("Goals Conceded",     "Goals Against p90"),
    ("xG Against",         "xG Against p90"),
    ("Aerial Duels",       "Aerial Duels p90"),
    ("Aerial Duel Succ %", "Aerial Duels Won %"),
    ("Defensive Duels",    "Defensive Duels p90"),
    ("Def Duel Win %",     "Defensive Duels Won %"),
    ("Shots Against",      "Shots Against p90"),
    ("PPDA",               "PPDA"),
]
TEAM_METRICS_POS = [
    ("Dribbles",           "Dribbles p90"),
    ("Possession",         "Possession %"),
    ("Passes",             "Passes p90"),
    ("Passing Acc %",      "Pass Accuracy %"),
    ("Long Passes",        "Long Passes p90"),
    ("Long Passing %",     "Long Pass Accuracy %"),
    ("Passes to Final 3rd","Passes to Final Third p90"),
    ("Progressive Passes", "Progressive Passes p90"),
    ("Progressive Runs",   "Progressive Runs p90"),
]

def avail_pairs(pairs, row_or_df):
    if hasattr(row_or_df,"columns"):
        return [(lab,col) for lab,col in pairs if col in row_or_df.columns]
    return [(lab,col) for lab,col in pairs if col in row_or_df.index]

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 1 – LEAGUE TABLE
# ══════════════════════════════════════════════════════
st.subheader("📊 League Table")
display_cols = ["League","Team","Matches","Wins","Draws","Losses","Points","Expected Points",
                "Goals For","Goals Against","Goal Difference","xG p90","OVR","ATT","DEF","POS"]
show_cols = [c for c in display_cols if c in df_sorted.columns]
st.dataframe(df_sorted[show_cols].style.format({c:"{:.1f}" for c in ["OVR","ATT","DEF","POS","xG p90","Expected Points"]}),
             use_container_width=True)
st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 2 – PRO LAYOUT (Team Cards)
# ══════════════════════════════════════════════════════
st.subheader("🃏 Pro Layout — Team Cards")
st.markdown("""
<style>
.team-card{position:relative;width:min(460px,96%);background:#141823;
    border:1px solid rgba(255,255,255,.06);border-radius:20px;padding:16px;
    margin-bottom:12px;box-shadow:inset 0 1px 0 rgba(255,255,255,.03),0 6px 24px rgba(0,0,0,.35);}
.tc-inner{display:grid;grid-template-columns:96px 1fr 52px;gap:12px;align-items:start;}
.team-badge{width:96px;height:96px;border-radius:12px;border:1px solid #2a3145;
    overflow:hidden;background:#0b0d12;display:flex;align-items:center;justify-content:center;}
.team-badge img{width:100%;height:100%;object-fit:contain;}
.tc-badge-sub{font-size:11px;color:#8899c0;margin-top:6px;text-align:center;line-height:1.65;}
.tc-name{font-weight:800;font-size:20px;color:#e8ecff;margin-bottom:6px;}
.tc-sub{color:#a8b3cf;font-size:14px;opacity:.9;}
.tc-pill{padding:2px 6px;min-width:36px;border-radius:6px;font-weight:700;font-size:17px;
    line-height:1;color:#0b0d12;text-align:center;display:inline-block;}
.tc-pill-row{display:flex;gap:8px;align-items:center;margin:3px 0;}
.tc-rank{position:absolute;top:10px;right:14px;color:#b7bfe1;font-weight:800;font-size:18px;}
.tc-wrap{display:flex;justify-content:center;}
.m-sec{background:#121621;border:1px solid #242b3b;border-radius:16px;padding:10px 12px;margin-bottom:8px;}
.m-sec-title{color:#e8ecff;font-weight:800;letter-spacing:.02em;margin:4px 0 10px 0;}
.m-row{display:flex;align-items:center;gap:10px;padding:6px 8px;border-radius:8px;}
.m-label{color:#c9d3f2;font-size:14px;flex:1;}
.m-val{color:#a8b3cf;font-size:12px;min-width:50px;text-align:right;}
.m-badge{min-width:40px;text-align:center;padding:2px 8px;border-radius:7px;
    font-weight:800;font-size:17px;color:#0b0d12;}
</style>""",unsafe_allow_html=True)

pro_top_n = st.number_input("Top N teams",5,200,20,5,key="ts_pro_topn")
pro_search = st.text_input("Search team","",key="ts_pro_search")
pro_league_filter = st.selectbox("Filter by league",["All"]+sorted(df_sorted["League"].dropna().unique()),key="ts_pro_league")

df_pro = df_sorted.copy()
if pro_search:
    df_pro = df_pro[df_pro["Team"].str.contains(pro_search,case=False,na=False)]
if pro_league_filter != "All":
    df_pro = df_pro[df_pro["League"] == pro_league_filter]
df_pro = df_pro.head(int(pro_top_n))

for i,(_,row) in enumerate(df_pro.iterrows()):
    team=str(row.get("Team",""));   league=str(row.get("League",""))
    ovr=row.get("OVR",0); att=row.get("ATT",0); defv=row.get("DEF",0); posv=row.get("POS",0)

    avg_age=row.get("Avg Age",None)
    avg_age_str = f"Avg Age: {float(avg_age):.1f}" if pd.notna(avg_age) and avg_age is not None else ""

    pos_rank, total_teams = get_league_position(team)
    xp_rank,  _           = get_xpoints_position(team)
    pts_val   = row.get("Points",None); matches_val = row.get("Matches",None)

    pos_str  = f"League Pos: {pos_rank}/{total_teams}" if pos_rank else ""
    xpos_str = f"xPos: {xp_rank}/{total_teams}" if xp_rank else ""
    pts_str  = (f"Points: {int(pts_val)}/{int(matches_val)}"
                if pd.notna(pts_val) and pd.notna(matches_val) and float(matches_val)>0 else "")

    badge_url  = fotmob_crest_url(team)
    badge_html = f"<img src='{badge_url}' style='width:80px;height:80px;object-fit:contain;'>" if badge_url else "🏟️"
    flag       = flag_html(league)

    pill_rows = "".join([
        f"<div class='tc-pill-row'>"
        f"<span class='tc-pill' style='background:{rating_color(sv)}'>{fmt2(sv)}</span>"
        f"<span class='tc-sub'>{lab}</span></div>"
        for lab,sv in [("Overall",ovr),("Attack",att),("Defense",defv),("Possession",posv)]
    ])

    sub_lines = [s for s in [avg_age_str,pos_str,xpos_str,pts_str] if s]
    badge_sub_html = "".join(f"<div>{s}</div>" for s in sub_lines)

    st.markdown(f"""
    <div class='tc-wrap'><div class='team-card'>
      <div class='tc-inner'>
        <div>
          <div class='team-badge'>{badge_html}</div>
          <div class='tc-badge-sub'>{badge_sub_html}</div>
        </div>
        <div>
          <div class='tc-name'>{team}</div>
          {pill_rows}
          <div style='margin-top:8px;font-size:13px;color:#dbe3ff;'>{flag}{league}</div>
        </div>
        <div class='tc-rank'>#{i+1:02d}</div>
      </div>
    </div></div>""",unsafe_allow_html=True)

    with st.expander("Metrics",expanded=False):
        def _sec(title,pairs):
            rows_html=[]
            for lab,col in avail_pairs(pairs,row.to_frame().T):
                p=metric_pct(row,col); v=metric_val(row,col)
                rows_html.append(
                    f"<div class='m-row'><div class='m-label'>{lab}</div>"
                    f"<div class='m-val'>{v}</div>"
                    f"<div class='m-badge' style='background:{rating_color(p)}'>{fmt2(p)}</div></div>")
            return f"<div class='m-sec'><div class='m-sec-title'>{title}</div>{''.join(rows_html)}</div>"

        col1,col2,col3 = st.columns(3)
        with col1: st.markdown(_sec("ATTACKING",  TEAM_METRICS_ATT),unsafe_allow_html=True)
        with col2: st.markdown(_sec("DEFENSIVE",  TEAM_METRICS_DEF),unsafe_allow_html=True)
        with col3: st.markdown(_sec("POSSESSION", TEAM_METRICS_POS),unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 3 – TEAM PROFILE (Light, player-style)
# ══════════════════════════════════════════════════════
st.subheader("🎯 Team Profile")

sel_team = st.selectbox("Select team",team_options,key="ts_profile_team")
edit_name_on = st.checkbox("Edit display name",False,key="ts_edit_team_name")
team_display_name = st.text_input("Display name",sel_team,key="ts_team_disp_input") if edit_name_on else sel_team

team_row = df[df["Team"]==sel_team]
if team_row.empty:
    st.info("Team not found.")
else:
    team_row   = team_row.iloc[0]
    team_league= str(team_row.get("League",""))

    comp_leagues = st.multiselect("Comparison pool (default = own league)",
                                  sorted(df["League"].dropna().unique()),
                                  default=[team_league],key="ts_profile_comp_leagues")
    pool = df[df["League"].isin(comp_leagues)] if comp_leagues else df[df["League"]==team_league]

    # 12-metric radar with xGA + Goals vs between Final 3rd and Points
    RADAR_TUPLES = [
        ("xG",          "xG p90",                   False),
        ("Goals",       "Goals p90",                False),
        ("Touches",     "Touches in Box p90",        False),
        ("PPDA",        "PPDA",                      True),
        ("Possession",  "Possession %",              False),
        ("Passes",      "Passes p90",                False),
        ("Long Passes", "Long Passes p90",           False),
        ("Final 3rd",   "Passes to Final Third p90", False),
        ("xGA",         "xG Against p90",             True),
        ("Goals vs",    "Goals Against p90",          True),
        ("Points",      "Points",                    False),
        ("xPoints",     "Expected Points",           False),
    ]
    radar_tuples = [(lab,col,inv) for lab,col,inv in RADAR_TUPLES if col in df.columns]

    def team_pct(t_row,pool_df,col,invert=False):
        if col not in pool_df.columns or col not in t_row.index: return 50.0
        s=pd.to_numeric(pool_df[col],errors="coerce").dropna()
        v=float(t_row[col]) if pd.notna(t_row.get(col)) else np.nan
        if pd.isna(v) or s.empty: return 50.0
        p=(s<v).mean()*100+(s==v).mean()*50
        return float(np.clip((100-p) if invert else p,0,100))

    pcts        = [team_pct(team_row,pool,col,inv) for _,col,inv in radar_tuples]
    labels_clean= [lab for lab,_,_ in radar_tuples]
    cmap_profile = LinearSegmentedColormap.from_list("cp",["#be2a3e","#e25f48","#f88f4d","#f4d166","#90b960","#4b9b5f","#22763f"])

    N=len(radar_tuples)
    angles=np.linspace(0,2*np.pi,N,endpoint=False)[::-1]
    rot_shift=np.deg2rad(75)-angles[0]
    rot_angles=[(a+rot_shift)%(2*np.pi) for a in angles]
    bar_width=(2*np.pi/N)*0.85

    fig=plt.figure(figsize=(9,7.5)); fig.patch.set_facecolor("#f3f4f6")
    ax=fig.add_axes([0.05,0.05,0.9,0.85],polar=True); ax.set_facecolor("#f3f4f6"); ax.set_rlim(0,100)

    for i in range(N):
        ax.bar(rot_angles[i],100,width=bar_width,color="#d5d7db",edgecolor="none",zorder=0)
    for i,(p,c) in enumerate(zip(pcts,[cmap_profile(p/100) for p in pcts])):
        ax.bar(rot_angles[i],p,width=bar_width,color=c,edgecolor="black",linewidth=0.8,zorder=2)
        if p>=12:
            lp=p-10 if p>=30 else p*0.7
            ax.text(rot_angles[i],lp,f"{int(round(p))}",ha="center",va="center",fontsize=9,weight="bold",color="white",zorder=3)
    for i in range(N):
        sep=(rot_angles[i]-bar_width/2)%(2*np.pi)
        ax.plot([sep,sep],[0,100],color="black",lw=0.9,zorder=4)
    ax.plot(np.linspace(0,2*np.pi,500),[100]*500,color="black",lw=2.2,zorder=5)
    for i,lab in enumerate(labels_clean):
        ax.text(rot_angles[i],120,lab,ha="center",va="center",fontsize=8.5,weight="bold",color="#111827",zorder=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.spines["polar"].set_visible(False); ax.grid(False)

    matches_v  = int(team_row.get("Matches",0)) if pd.notna(team_row.get("Matches")) else 0
    gf_v       = int(team_row.get("Goals For",0)) if pd.notna(team_row.get("Goals For")) else 0
    ga_v       = int(team_row.get("Goals Against",0)) if pd.notna(team_row.get("Goals Against")) else 0
    pts_v      = int(team_row.get("Points",0)) if pd.notna(team_row.get("Points")) else 0

    st.pyplot(fig,use_container_width=True)

    buf_p=io.BytesIO()
    fig.savefig(buf_p,format="png",dpi=170,bbox_inches="tight",facecolor="#f3f4f6")
    st.download_button("⬇️ Download Team Profile",buf_p.getvalue(),
                       f"{team_display_name.replace(' ','_')}_profile.png","image/png",key="ts_profile_dl")
    plt.close(fig)

    # Style/Strength/Weakness chips
    STYLE_TEAM={
        "Possession %":           {"style":"Possession-based","sw":"Possession"},
        "Passes p90":             {"style":"High passing volume","sw":"Passing Volume"},
        "Pass Accuracy %":        {"style":"Technical passing","sw":"Passing Accuracy"},
        "PPDA":                   {"style":"High press","sw":"Pressing"},
        "xG p90":                 {"style":"Creates many chances","sw":"Chance Creation"},
        "Goals p90":              {"style":"Clinical in front of goal","sw":"Goalscoring"},
        "Shots p90":              {"style":"High shot volume","sw":"Shot Volume"},
        "Touches in Box p90":     {"style":"Gets into the box often","sw":"Box Penetration"},
        "xG Against p90":         {"style":"Solid defensively","sw":"Defensive Solidity"},
        "Goals Against p90":      {"style":"Hard to beat","sw":"Goals Conceded"},
        "Shots Against p90":      {"style":"Limits opponent shots","sw":"Shot Prevention"},
        "Progressive Passes p90": {"style":"Progressive ball movement","sw":"Progression"},
        "Progressive Runs p90":   {"style":"Dynamic runners","sw":"Carries"},
    }
    HI,LO,STYLE_T=70,30,65
    strengths,weaknesses,styles=[],[],[]
    for _,col,inv in radar_tuples:
        if col not in df.columns: continue
        p=team_pct(team_row,pool,col,inv); cfg=STYLE_TEAM.get(col,{})
        sw=cfg.get("sw"); sty=cfg.get("style")
        if sw:
            if p>=HI:   strengths.append(sw)
            elif p<=LO: weaknesses.append(sw)
        if sty and p>=STYLE_T: styles.append(sty)

    def chips_html(items,bg):
        if not items: return "_None identified._"
        return " ".join(f"<span style='background:{bg};color:#111;padding:2px 8px;border-radius:10px;"
                        f"margin:0 5px 5px 0;display:inline-block;font-size:14px;'>{t}</span>"
                        for t in list(dict.fromkeys(items))[:8])

    st.markdown("**Style:**");     st.markdown(chips_html(styles,"#bfdbfe"),unsafe_allow_html=True)
    st.markdown("**Strengths:**"); st.markdown(chips_html(strengths,"#a7f3d0"),unsafe_allow_html=True)
    st.markdown("**Weaknesses:**");st.markdown(chips_html(weaknesses,"#fecaca"),unsafe_allow_html=True)

    # Score table – diverging red→orange→gold→green
    sdf=pd.DataFrame({"Score":["OVR","ATT","DEF","POS"],
                      "Value":[team_row.get("OVR",np.nan),team_row.get("ATT",np.nan),
                               team_row.get("DEF",np.nan),team_row.get("POS",np.nan)]}).set_index("Score")

    def sc_color(v):
        if pd.isna(v): return "background:#f3f4f6"
        v=float(v)
        for thr,bg in [(85,"#2E6114"),(75,"#5C9E2E"),(66,"#7FBC41"),(54,"#A7D763"),
                       (44,"#F6D645"),(25,"#D77A2E"),(0,"#C63733")]:
            if v>=thr: return f"background:{bg};color:#0b0d12"
        return "background:#C63733;color:#0b0d12"

    st.dataframe(sdf.style.map(lambda x:sc_color(x),subset=["Value"])
                 .format({"Value":lambda x:f"{int(round(x))}" if pd.notna(x) else "—"}),
                 use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 4 – FEATURE F (Dark Percentile Board)
# ══════════════════════════════════════════════════════
st.subheader("📋 Feature F — Percentile Board (Dark)")

sel_team_f=st.selectbox("Select team",team_options,key="ts_f_team")
t_row_f=df[df["Team"]==sel_team_f]
if t_row_f.empty:
    st.info("Team not found.")
else:
    t_row_f=t_row_f.iloc[0]; t_league_f=str(t_row_f.get("League","")); pool_f=df[df["League"]==t_league_f]

    def pct_f(col,invert=False):
        if col not in df.columns: return 0.0
        s=pd.to_numeric(pool_f[col],errors="coerce").dropna()
        v=float(t_row_f[col]) if pd.notna(t_row_f.get(col)) else np.nan
        if pd.isna(v) or s.empty: return 0.0
        p=(s<v).mean()*100+(s==v).mean()*50
        return float(np.clip((100-p) if invert else p,0,100))

    def val_f(col):
        v=t_row_f.get(col,np.nan)
        if pd.isna(v): return "—"
        return f"{float(v):.2f}".rstrip("0").rstrip(".")

    ATT_F=[("Crosses","Crosses p90",False),("Goals Scored","Goals p90",False),("xG","xG p90",False),
           ("Shots","Shots p90",False),("Shooting %","Shot Accuracy %",False),("Touches in Box","Touches in Box p90",False)]
    DEF_F=[("Goals Conceded","Goals Against p90",True),("xG Against","xG Against p90",True),
           ("Aerial Duels","Aerial Duels p90",False),("Aerial Duel Succ %","Aerial Duels Won %",False),
           ("Defensive Duels","Defensive Duels p90",False),("Def Duel Win %","Defensive Duels Won %",False),
           ("Shots Against","Shots Against p90",True),("PPDA","PPDA",True)]
    POS_F=[("Dribbles","Dribbles p90",False),("Possession","Possession %",False),
           ("Passes","Passes p90",False),("Passing Acc %","Pass Accuracy %",False),
           ("Long Passes","Long Passes p90",False),("Long Passing %","Long Pass Accuracy %",False),
           ("Passes to Final 3rd","Passes to Final Third p90",False),
           ("Progressive Passes","Progressive Passes p90",False),("Progressive Runs","Progressive Runs p90",False)]

    sections_f=[("Attacking",[(lab,pct_f(col,inv),val_f(col)) for lab,col,inv in ATT_F if col in df.columns]),
                ("Defensive",[(lab,pct_f(col,inv),val_f(col)) for lab,col,inv in DEF_F if col in df.columns]),
                ("Possession",[(lab,pct_f(col,inv),val_f(col)) for lab,col,inv in POS_F if col in df.columns])]
    sections_f=[(t,lst) for t,lst in sections_f if lst]

    PAGE_BG_F="#0a0f1c"; AX_BG_F="#0f151f"; TRACK_F="#1b2636"
    TITLE_C_F="#f3f5f7"; LABEL_C_F="#e8eef8"
    TR=np.array([199,54,60]); TG=np.array([240,197,106]); TGN=np.array([61,166,91])
    def p2rgb_f(v):
        v=float(np.clip(v,0,100)); c1,c2=(TR,TG) if v<=50 else (TG,TGN); t2=v/50 if v<=50 else (v-50)/50
        c=c1+(c2-c1)*t2; return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"

    total_rf=sum(len(lst) for _,lst in sections_f)
    fig_f=plt.figure(figsize=(10,8),dpi=100); fig_f.patch.set_facecolor(PAGE_BG_F)
    lm=0.035; rm=0.02; tm=0.04; bm=0.09; hh=0.06; gp=0.018
    rs=1-(tm+bm)-hh*len(sections_f)-gp*(len(sections_f)-1); rslot=rs/max(total_rf,1)
    BF=0.85; gut=0.225; tks=np.arange(0,101,10); LF=lm+0.015
    fig_f.text(LF,0.965,f"{sel_team_f} — {t_league_f}",ha="left",va="top",fontsize=18,fontweight="900",color=TITLE_C_F)
    y_top_f=1-tm-0.06

    for idx,(title,data) in enumerate(sections_f):
        is_last=(idx==len(sections_f)-1); n=len(data)
        fig_f.text(LF,y_top_f-0.008,title,ha="left",va="top",fontsize=16,fontweight="900",color=TITLE_C_F)
        ax_f=fig_f.add_axes([LF+gut,y_top_f-hh-n*rslot,1-LF-rm-gut,n*rslot])
        ax_f.set_facecolor(AX_BG_F); ax_f.set_xlim(0,100); ax_f.set_ylim(-0.5,n-0.5)
        for s in ax_f.spines.values(): s.set_visible(False)
        ax_f.tick_params(axis="x",bottom=False,labelbottom=False,length=0)
        for i in range(n): ax_f.add_patch(plt.Rectangle((0,i-BF/2),100,BF,color=TRACK_F,ec="none",zorder=0.5))
        for gx in tks: ax_f.vlines(gx,-0.5,n-0.5,colors=(1,1,1,0.16),lw=0.8,zorder=0.75)
        for i,(lab,pct,vs) in enumerate(data[::-1]):
            bw=float(np.clip(pct,0,100))
            ax_f.add_patch(plt.Rectangle((0,i-BF/2),bw,BF,color=p2rgb_f(bw),ec="none",zorder=1))
            ax_f.text(1,i,vs,ha="left",va="center",fontsize=8,color="#0B0B0B",zorder=2)
        ax_f.axvline(50,color="#FFF",ls=(0,(4,4)),lw=1.5,alpha=0.85,zorder=3.5)
        for i,(lab,_,_) in enumerate(data[::-1]):
            yf=(y_top_f-hh-n*rslot)+((i+0.5)*rslot)
            fig_f.text(LF,yf,lab,ha="left",va="center",fontsize=10,fontweight="bold",color=LABEL_C_F)
        if is_last:
            trans=ax_f.get_xaxis_transform()
            oi=ScaledTranslation(7/72,0,fig_f.dpi_scale_trans)
            o0=ScaledTranslation(4/72,0,fig_f.dpi_scale_trans)
            o100=ScaledTranslation(10/72,0,fig_f.dpi_scale_trans)
            yl=-0.075
            for gx in tks:
                ax_f.plot([gx,gx],[-0.03,0],transform=trans,color=(1,1,1,0.6),lw=1.1,clip_on=False,zorder=4)
                ax_f.text(gx,yl,f"{int(gx)}",transform=trans,ha="center",va="top",fontsize=10,fontweight="700",color="#FFF",zorder=4,clip_on=False)
                off=o0 if gx==0 else (o100 if gx==100 else oi)
                ax_f.text(gx,yl,"%",transform=trans+off,ha="left",va="top",fontsize=10,fontweight="700",color="#FFF",zorder=4,clip_on=False)
        else:
            y0f=y_top_f-hh-n*rslot-0.008
            fig_f.lines.append(plt.Line2D([LF,1-rm],[y0f,y0f],transform=fig_f.transFigure,color="#fff",lw=1.2,alpha=0.95))
        y_top_f=y_top_f-hh-n*rslot-gp

    fig_f.text((LF+gut+(1-rm))/2,bm*0.3,"Percentile Rank",ha="center",va="center",fontsize=11,fontweight="bold",color=LABEL_C_F)
    st.pyplot(fig_f,use_container_width=True)
    buf_f=io.BytesIO(); fig_f.savefig(buf_f,format="png",dpi=130,bbox_inches="tight",facecolor=PAGE_BG_F)
    st.download_button("⬇️ Download Feature F",buf_f.getvalue(),f"{sel_team_f.replace(' ','_')}_featureF.png","image/png",key="ts_f_dl")
    plt.close(fig_f)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 5 – FEATURE Z (Light Percentile Board)
# ══════════════════════════════════════════════════════
st.subheader("📋 Feature Z — Percentile Board (Light)")

sel_team_z=st.selectbox("Select team",team_options,key="ts_z_team")
with st.expander("Feature Z options",expanded=False):
    fz_edit_footer=st.checkbox("Edit footer caption",False,key="ts_fz_edit_footer")
    fz_footer_text=st.text_input("Footer caption","Percentile Rank",disabled=not fz_edit_footer,key="ts_fz_footer")

t_row_z=df[df["Team"]==sel_team_z]
if t_row_z.empty:
    st.info("Team not found.")
else:
    t_row_z=t_row_z.iloc[0]; t_league_z=str(t_row_z.get("League","")); pool_z=df[df["League"]==t_league_z]

    def pct_z(col,invert=False):
        if col not in df.columns: return 0.0
        s=pd.to_numeric(pool_z[col],errors="coerce").dropna()
        v=float(t_row_z[col]) if pd.notna(t_row_z.get(col)) else np.nan
        if pd.isna(v) or s.empty: return 0.0
        p=(s<v).mean()*100+(s==v).mean()*50
        return float(np.clip((100-p) if invert else p,0,100))

    def val_z(col):
        v=t_row_z.get(col,np.nan)
        if pd.isna(v): return "—"
        return f"{float(v):.2f}".rstrip("0").rstrip(".")

    ATT_Z=[("Crosses","Crosses p90",False),("Goals Scored","Goals p90",False),("xG","xG p90",False),
           ("Shots","Shots p90",False),("Shooting %","Shot Accuracy %",False),("Touches in Box","Touches in Box p90",False)]
    DEF_Z=[("Goals Conceded","Goals Against p90",True),("xG Against","xG Against p90",True),
           ("Aerial Duels","Aerial Duels p90",False),("Aerial Duel Succ %","Aerial Duels Won %",False),
           ("Defensive Duels","Defensive Duels p90",False),("Def Duel Win %","Defensive Duels Won %",False),
           ("Shots Against","Shots Against p90",True),("PPDA","PPDA",True)]
    POS_Z=[("Dribbles","Dribbles p90",False),("Possession","Possession %",False),
           ("Passes","Passes p90",False),("Passing Acc %","Pass Accuracy %",False),
           ("Long Passes","Long Passes p90",False),("Long Passing %","Long Pass Accuracy %",False),
           ("Passes to Final 3rd","Passes to Final Third p90",False),
           ("Progressive Passes","Progressive Passes p90",False),("Progressive Runs","Progressive Runs p90",False)]

    sections_z=[("Attacking",[(lab,pct_z(col,inv),val_z(col)) for lab,col,inv in ATT_Z if col in df.columns]),
                ("Defensive",[(lab,pct_z(col,inv),val_z(col)) for lab,col,inv in DEF_Z if col in df.columns]),
                ("Possession",[(lab,pct_z(col,inv),val_z(col)) for lab,col,inv in POS_Z if col in df.columns])]
    sections_z=[(t,lst) for t,lst in sections_z if lst]

    PAGE_BG_Z="#ebebeb"; AX_BG_Z="#f3f3f3"; TRACK_Z="#d6d6d6"
    TITLE_C_Z="#111111"; LABEL_C_Z="#222222"
    TRZ=np.array([199,54,60]); TGZ=np.array([240,197,106]); TGNZ=np.array([61,166,91])
    def p2rgb_z(v):
        v=float(np.clip(v,0,100)); c1,c2=(TRZ,TGZ) if v<=50 else (TGZ,TGNZ); t2=v/50 if v<=50 else (v-50)/50
        c=c1+(c2-c1)*t2; return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"

    total_rz=sum(len(lst) for _,lst in sections_z)
    fig_z=plt.figure(figsize=(10,8),dpi=100); fig_z.patch.set_facecolor(PAGE_BG_Z)
    lmz=0.035; rmz=0.02; tmz=0.025; bmz=0.09; hhz=0.055; gpz=0.018
    rsz=1-(tmz+bmz)-hhz*len(sections_z)-gpz*(len(sections_z)-1); rslz=rsz/max(total_rz,1)
    BFZ=0.88; gutz=0.225; tksz=np.arange(0,101,10); LFZ=lmz+0.015
    y_top_z=1-tmz

    for idx,(title,data) in enumerate(sections_z):
        is_last_z=(idx==len(sections_z)-1); n=len(data)
        fig_z.text(LFZ,y_top_z-0.008,title,ha="left",va="top",fontsize=16,fontweight="900",color=TITLE_C_Z)
        ax_z=fig_z.add_axes([LFZ+gutz,y_top_z-hhz-n*rslz,1-LFZ-rmz-gutz,n*rslz])
        ax_z.set_facecolor(AX_BG_Z); ax_z.set_xlim(0,100); ax_z.set_ylim(-0.5,n-0.5)
        for s in ax_z.spines.values(): s.set_visible(False)
        ax_z.tick_params(axis="x",bottom=False,labelbottom=False,length=0)
        ax_z.tick_params(axis="y",left=False,labelleft=False,length=0)
        for i in range(n): ax_z.add_patch(plt.Rectangle((0,i-BFZ/2),100,BFZ,color=TRACK_Z,ec="none",zorder=0.5))
        for gx in tksz: ax_z.vlines(gx,-0.5,n-0.5,colors=(0,0,0,0.16),lw=0.8,zorder=0.75)
        for i,(lab,pct,vs) in enumerate(data[::-1]):
            bw=float(np.clip(pct,0,100))
            ax_z.add_patch(plt.Rectangle((0,i-BFZ/2),bw,BFZ,color=p2rgb_z(bw),ec="none",zorder=1))
            ax_z.text(max(1.0,bw+0.5),i,vs,ha="left",va="center",fontsize=8,color="#0B0B0B",zorder=2,clip_on=False)
        ax_z.axvline(50,color="#000",ls=(0,(4,4)),lw=1.5,alpha=0.6,zorder=3.5)
        for i,(lab,_,_) in enumerate(data[::-1]):
            yfz=(y_top_z-hhz-n*rslz)+((i+0.5)*rslz)
            fig_z.text(LFZ,yfz,lab,ha="left",va="center",fontsize=10,fontweight="bold",color=LABEL_C_Z)
        if is_last_z:
            trz=ax_z.get_xaxis_transform()
            oiz=ScaledTranslation(7/72,0,fig_z.dpi_scale_trans)
            o0z=ScaledTranslation(4/72,0,fig_z.dpi_scale_trans)
            o100z=ScaledTranslation(10/72,0,fig_z.dpi_scale_trans)
            ylz=-0.075
            for gx in tksz:
                ax_z.plot([gx,gx],[-0.03,0],transform=trz,color=(0,0,0,0.6),lw=1.1,clip_on=False,zorder=4)
                ax_z.text(gx,ylz,f"{int(gx)}",transform=trz,ha="center",va="top",fontsize=10,fontweight="700",color="#000",zorder=4,clip_on=False)
                offz=o0z if gx==0 else (o100z if gx==100 else oiz)
                ax_z.text(gx,ylz,"%",transform=trz+offz,ha="left",va="top",fontsize=10,fontweight="700",color="#000",zorder=4,clip_on=False)
        else:
            y0z=y_top_z-hhz-n*rslz-0.008
            fig_z.lines.append(plt.Line2D([LFZ,1-rmz],[y0z,y0z],transform=fig_z.transFigure,color="#000",lw=1.2,alpha=0.3))
        y_top_z=y_top_z-hhz-n*rslz-gpz

    footer_z=fz_footer_text if fz_edit_footer else "Percentile Rank"
    fig_z.text((LFZ+gutz+(1-rmz))/2,bmz*0.35,footer_z,ha="center",va="center",fontsize=11,fontweight="bold",color=LABEL_C_Z)
    st.pyplot(fig_z,use_container_width=True)
    buf_z=io.BytesIO(); fig_z.savefig(buf_z,format="png",dpi=130,bbox_inches="tight",facecolor=PAGE_BG_Z)
    st.download_button("⬇️ Download Feature Z",buf_z.getvalue(),f"{sel_team_z.replace(' ','_')}_featureZ.png","image/png",key="ts_z_dl")
    plt.close(fig_z)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 6 – FEATURE Y (Polar Radar, Light, no default title)
# ══════════════════════════════════════════════════════
st.subheader("🌀 Feature Y — Team Polar Radar")

sel_team_y=st.selectbox("Select team",team_options,key="ts_y_team")
t_row_y=df[df["Team"]==sel_team_y]
if t_row_y.empty:
    st.info("Team not found.")
else:
    t_row_y=t_row_y.iloc[0]; t_league_y=str(t_row_y.get("League",""))
    comp_y=st.multiselect("Comparison pool",sorted(df["League"].dropna().unique()),default=[t_league_y],key="ts_y_comp")
    pool_y=df[df["League"].isin(comp_y)] if comp_y else df[df["League"]==t_league_y]
    show_y_title=st.checkbox("Add custom title",False,key="ts_y_show_title")
    custom_y_title=st.text_input("Custom title",sel_team_y,key="ts_y_custom_title") if show_y_title else ""

    METRICS_Y=[("xG","xG p90",False),("Goals","Goals p90",False),("Touches","Touches in Box p90",False),
               ("PPDA","PPDA",True),("Possession","Possession %",False),("Passes","Passes p90",False),
               ("Long Passes","Long Passes p90",False),("Final 3rd","Passes to Final Third p90",False),
               ("xGA","xG Against p90",True),("Goals vs","Goals Against p90",True),
               ("Points","Points",False),("xPoints","Expected Points",False)]
    metrics_y=[(lab,col,inv) for lab,col,inv in METRICS_Y if col in df.columns]

    def pct_y(col,invert=False):
        if col not in pool_y.columns: return 50
        s=pd.to_numeric(pool_y[col],errors="coerce").dropna()
        v=float(t_row_y[col]) if pd.notna(t_row_y.get(col)) else np.nan
        if pd.isna(v) or s.empty: return 50
        p=(s<v).mean()*100+(s==v).mean()*50
        return float(np.clip((100-p) if invert else p,0,100))

    pcts_y=[pct_y(col,inv) for _,col,inv in metrics_y]; labels_y=[lab for lab,_,_ in metrics_y]
    N_y=len(metrics_y)
    cmap_y=LinearSegmentedColormap.from_list("csy",["#be2a3e","#e25f48","#f88f4d","#f4d166","#90b960","#4b9b5f","#22763f"])
    angles_y=np.linspace(0,2*np.pi,N_y,endpoint=False)[::-1]
    rot_shift_y=np.deg2rad(75)-angles_y[0]
    rot_angles_y=[(a+rot_shift_y)%(2*np.pi) for a in angles_y]
    bar_w_y=(2*np.pi/N_y)*0.85

    fig_y=plt.figure(figsize=(8,6.5)); fig_y.patch.set_facecolor("#f3f4f6")
    ax_y=fig_y.add_axes([0.05,0.05,0.9,0.9],polar=True); ax_y.set_facecolor("#f3f4f6"); ax_y.set_rlim(0,100)
    for i in range(N_y): ax_y.bar(rot_angles_y[i],100,width=bar_w_y,color="#d5d7db",edgecolor="none",zorder=0)
    for i,p in enumerate(pcts_y):
        c=cmap_y(p/100)
        ax_y.bar(rot_angles_y[i],p,width=bar_w_y,color=c,edgecolor="black",linewidth=0.8,zorder=2)
        if p>=12:
            lp=p-10 if p>=30 else p*0.7
            ax_y.text(rot_angles_y[i],lp,f"{int(round(p))}",ha="center",va="center",fontsize=11,weight="bold",color="white",zorder=3)
    for i in range(N_y):
        sep=(rot_angles_y[i]-bar_w_y/2)%(2*np.pi); ax_y.plot([sep,sep],[0,100],color="black",lw=0.9,zorder=4)
    ax_y.plot(np.linspace(0,2*np.pi,500),[100]*500,color="black",lw=2.2,zorder=5)
    for i,lab in enumerate(labels_y):
        ax_y.text(rot_angles_y[i],145,lab,ha="center",va="center",fontsize=9,weight="bold",color="#111827",zorder=5)
    ax_y.set_xticks([]); ax_y.set_yticks([]); ax_y.spines["polar"].set_visible(False); ax_y.grid(False)
    if show_y_title and custom_y_title.strip():
        fig_y.text(0.5,0.97,custom_y_title.strip(),ha="center",fontsize=14,weight="bold",color="#111827")

    st.pyplot(fig_y,use_container_width=True)
    buf_y=io.BytesIO(); fig_y.savefig(buf_y,format="png",dpi=300,bbox_inches="tight",facecolor="#f3f4f6")
    st.download_button("⬇️ Download Feature Y",buf_y.getvalue(),f"{sel_team_y.replace(' ','_')}_featureY.png","image/png",key="ts_y_dl")
    plt.close(fig_y)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 7 – LEADERBOARD (player-style)
# ══════════════════════════════════════════════════════
st.subheader("📉 Leaderboard")

with st.expander("Leaderboard settings",expanded=False):
    lb_metric=st.selectbox("Metric",[c for c in NUMERIC_COLS if c in df.columns],format_func=mlabel,key="ts_lb_metric")
    lb_n=st.slider("Top N",5,40,20,5,key="ts_lb_n")
    lb_theme=st.radio("Theme",["Light","Dark"],horizontal=True,key="ts_lb_theme")
    lb_palette=st.selectbox("Palette",["Red–Gold–Green (diverging)","Light-grey → Black",
        "Light-Red → Dark-Red","Light-Blue → Dark-Blue","Light-Green → Dark-Green",
        "All Black","All Blue","All Green"],index=0,key="ts_lb_pal")
    lb_reverse=st.checkbox("Reverse colours",False,key="ts_lb_reverse")
    show_league_lb=st.checkbox("Show league in label",False,key="ts_lb_names")
    show_lb_title=st.checkbox("Show custom title",False,key="ts_lb_show_title")
    lb_custom_title=st.text_input("Custom title","",key="ts_lb_custom_title")

if lb_metric in df.columns:
    lb_df=df[["Team","League",lb_metric]].dropna(subset=[lb_metric]).copy()
    lb_df[lb_metric]=pd.to_numeric(lb_df[lb_metric],errors="coerce")
    lb_df=lb_df.dropna().sort_values(lb_metric,ascending=(lb_metric in INVERT_METRICS)).reset_index(drop=True).head(lb_n)

    if lb_theme=="Light":
        PBG_L="#ebebeb"; ABG_L="#ebebeb"; TXT_L="#111111"; GRID_L="#d7d7d7"; SPINE_L="#c8c8c8"
    else:
        PBG_L="#0a0f1c"; ABG_L="#0f151f"; TXT_L="#f5f5f5"; GRID_L="#3a4050"; SPINE_L="#6b7280"

    vals_lb=lb_df[lb_metric].values
    vmin_lb,vmax_lb=float(vals_lb.min()),float(vals_lb.max()) if len(vals_lb)>1 else (0.0,1.0)
    if vmin_lb==vmax_lb: vmax_lb=vmin_lb+1e-6
    ts_lb=(vals_lb-vmin_lb)/(vmax_lb-vmin_lb)
    if lb_metric in INVERT_METRICS: ts_lb=1.0-ts_lb
    if lb_reverse: ts_lb=1.0-ts_lb

    TR_L=np.array([199,54,60]); TG_L=np.array([240,197,106]); TGN_L=np.array([61,166,91])
    def lb_color(t):
        t=float(np.clip(t,0,1))
        def interp(a,b,u):
            a=np.array(a,dtype=float); b=np.array(b,dtype=float)
            return tuple(np.clip(a+(b-a)*np.clip(u,0,1),0,255)/255)
        if lb_palette=="Red–Gold–Green (diverging)":
            return interp(TR_L,TG_L,t/0.5) if t<=0.5 else interp(TG_L,TGN_L,(t-0.5)/0.5)
        if lb_palette=="Light-grey → Black":    return interp([210,214,220],[20,23,31],t)
        if lb_palette=="Light-Red → Dark-Red":  return interp([252,190,190],[139,0,0],t)
        if lb_palette=="Light-Blue → Dark-Blue":return interp([191,210,255],[10,42,102],t)
        if lb_palette=="Light-Green → Dark-Green":return interp([196,235,203],[12,92,48],t)
        if lb_palette=="All Black":  return (0,0,0)
        if lb_palette=="All Blue":   return (15/255,70/255,180/255)
        if lb_palette=="All Green":  return (20/255,120/255,60/255)
        return (0,0,0)

    bar_colors_lb=[lb_color(float(t)) for t in ts_lb]
    fig_lb,ax_lb=plt.subplots(figsize=(11.5,6.2))
    fig_lb.patch.set_facecolor(PBG_L); ax_lb.set_facecolor(ABG_L)
    title_lb=(lb_custom_title.strip() if (show_lb_title and lb_custom_title.strip()) else f"Top {lb_n} — {mlabel(lb_metric)}")
    fig_lb.suptitle(title_lb,fontsize=26,fontweight="bold",color=TXT_L,y=0.985)
    plt.subplots_adjust(top=0.90,left=0.30,right=0.965,bottom=0.14)
    ypos=np.arange(len(vals_lb))
    bars=ax_lb.barh(ypos,vals_lb,color=bar_colors_lb,edgecolor="none",zorder=2); ax_lb.invert_yaxis()
    ytlabs=[f"{r['Team']} ({r['League']})" if show_league_lb else r["Team"] for _,r in lb_df.iterrows()]
    ax_lb.set_yticks(ypos); ax_lb.set_yticklabels(ytlabs,fontsize=10.5,color=TXT_L)
    ax_lb.set_xlabel(mlabel(lb_metric),color=TXT_L,labelpad=6,fontsize=10.5,fontweight="semibold")
    ax_lb.grid(axis="x",color=GRID_L,linewidth=0.8,zorder=1)
    for s in ["top","right","left"]: ax_lb.spines[s].set_visible(False)
    ax_lb.spines["bottom"].set_color(SPINE_L); ax_lb.tick_params(axis="y",length=0)
    for tick in ax_lb.get_xticklabels(): tick.set_color(TXT_L)
    xmax_lb=float(vals_lb.max())*1.1 if len(vals_lb) else 1; ax_lb.set_xlim(0,xmax_lb)
    pad_lb=(ax_lb.get_xlim()[1]-ax_lb.get_xlim()[0])*0.012
    for rect,v in zip(bars,vals_lb):
        ax_lb.text(rect.get_width()+pad_lb,rect.get_y()+rect.get_height()/2,
                   f"{v:.2f}".rstrip("0").rstrip("."),va="center",ha="left",fontsize=8.5,color=TXT_L)
    st.pyplot(fig_lb,use_container_width=True); plt.close(fig_lb)

st.markdown("---")

# ══════════════════════════════════════════════════════
# SECTION 8 – SCATTER (player-style)
# ══════════════════════════════════════════════════════
st.subheader("🔵 Scatter Chart")

num_cols_sc=[c for c in NUMERIC_COLS if c in df.columns]
with st.expander("Scatter settings",expanded=False):
    sc_x=st.selectbox("X axis",num_cols_sc,index=num_cols_sc.index("xG p90") if "xG p90" in num_cols_sc else 0,format_func=mlabel,key="ts_sc_x")
    sc_y=st.selectbox("Y axis",num_cols_sc,index=num_cols_sc.index("xG Against p90") if "xG Against p90" in num_cols_sc else 1,format_func=mlabel,key="ts_sc_y")
    sc_colour=st.selectbox("Colour dots by",num_cols_sc,index=0,format_func=mlabel,key="ts_sc_colour")
    sc_palette=st.selectbox("Palette",["Red–Gold–Green (diverging)","Light-grey → Black",
        "Light-Red → Dark-Red","Light-Blue → Dark-Blue","Light-Green → Dark-Green","All Black","All Blue"],index=5,key="ts_sc_palette")
    sc_rev_pal=st.checkbox("Reverse colours",False,key="ts_sc_rev_pal")
    sc_theme=st.radio("Theme",["Light","Dark"],horizontal=True,key="ts_sc_theme")
    sc_labels=st.checkbox("Show labels",True,key="ts_sc_labels")
    sc_medians=st.checkbox("Show median lines",True,key="ts_sc_medians")
    sc_iqr=st.checkbox("Shade IQR (25–75%)",True,key="ts_sc_iqr")
    sc_size=st.slider("Point size",30,300,200,key="ts_sc_size")
    sc_alpha=st.slider("Point opacity",0.2,1.0,0.88,0.02,key="ts_sc_alpha")
    sc_marker=st.selectbox("Marker",["o","s","^","D"],index=0,key="ts_sc_marker")
    sc_label_size=st.slider("Label size",8,20,11,1,key="ts_sc_label_size")
    show_sc_title=st.checkbox("Custom title",False,key="ts_sc_show_title")
    sc_custom_title=st.text_input("Title text",f"{mlabel(sc_x)} vs {mlabel(sc_y)}",key="ts_sc_title_text")
    canvas_sc=st.selectbox("Canvas",["1280×720","1600×900","1920×1080"],index=1,key="ts_sc_canvas")
    top_gap_sc=st.slider("Top gap (px)",0,200,80,5,key="ts_sc_topgap")

if sc_x in df.columns and sc_y in df.columns:
    PBG_SC="#ebebeb" if sc_theme=="Light" else "#0a0f1c"
    PLOT_BG_SC="#f3f3f3" if sc_theme=="Light" else "#0f151f"
    GRID_SC="#d7d7d7" if sc_theme=="Light" else "#3a4050"
    TXT_SC="#111111" if sc_theme=="Light" else "#f5f5f5"
    STROKE_SC="#ffffff" if sc_theme=="Light" else "#1e293b"

    sc_df=df[["Team","League",sc_x,sc_y,sc_colour]].dropna(subset=[sc_x,sc_y]).copy()
    w_sc,h_sc=map(int,canvas_sc.replace("×","x").split("x"))
    fig_sc,ax_sc=plt.subplots(figsize=(w_sc/100,h_sc/100),dpi=100)
    fig_sc.patch.set_facecolor(PBG_SC); ax_sc.set_facecolor(PLOT_BG_SC)

    cvals=pd.to_numeric(sc_df[sc_colour],errors="coerce").fillna(0).values
    cmin_sc,cmax_sc=float(cvals.min()),float(cvals.max())
    if cmin_sc==cmax_sc: cmax_sc=cmin_sc+1e-6
    ts_sc=(cvals-cmin_sc)/(cmax_sc-cmin_sc)
    if sc_rev_pal: ts_sc=1.0-ts_sc

    TR_SC=np.array([199,54,60]); TG_SC=np.array([240,197,106]); TGN_SC=np.array([61,166,91])
    def sc_color_fn(t):
        t=float(np.clip(t,0,1))
        def interp(a,b,u):
            a=np.array(a,dtype=float); b=np.array(b,dtype=float)
            return tuple(np.clip(a+(b-a)*np.clip(u,0,1),0,255)/255)
        if sc_palette=="Red–Gold–Green (diverging)":
            return interp(TR_SC,TG_SC,t/0.5) if t<=0.5 else interp(TG_SC,TGN_SC,(t-0.5)/0.5)
        if sc_palette=="Light-grey → Black":    return interp([210,214,220],[20,23,31],t)
        if sc_palette=="Light-Red → Dark-Red":  return interp([252,190,190],[139,0,0],t)
        if sc_palette=="Light-Blue → Dark-Blue":return interp([191,210,255],[10,42,102],t)
        if sc_palette=="Light-Green → Dark-Green":return interp([196,235,203],[12,92,48],t)
        if sc_palette=="All Black":  return (0,0,0)
        if sc_palette=="All Blue":   return (15/255,70/255,180/255)
        return (0,0,0)

    sc_colors=[sc_color_fn(float(t)) for t in ts_sc]
    ax_sc.scatter(sc_df[sc_x],sc_df[sc_y],s=sc_size,c=sc_colors,alpha=float(sc_alpha),edgecolors="none",marker=sc_marker,zorder=2)

    if sc_iqr:
        xq1,xq3=np.nanpercentile(sc_df[sc_x],[25,75]); yq1,yq3=np.nanpercentile(sc_df[sc_y],[25,75])
        iqrc="#cfd3da" if sc_theme=="Light" else "#9aa4b1"
        ax_sc.axvspan(xq1,xq3,color=iqrc,alpha=0.25,zorder=1); ax_sc.axhspan(yq1,yq3,color=iqrc,alpha=0.25,zorder=1)
    if sc_medians:
        mx=float(sc_df[sc_x].median()); my=float(sc_df[sc_y].median())
        mc="#000000" if sc_theme=="Light" else "#ffffff"
        ax_sc.axvline(mx,color=mc,ls=(0,(4,4)),lw=2.2,zorder=3); ax_sc.axhline(my,color=mc,ls=(0,(4,4)),lw=2.2,zorder=3)
    if sc_labels:
        for _,row in sc_df.iterrows():
            t=ax_sc.annotate(row["Team"],(row[sc_x],row[sc_y]),xytext=(8,10),
                textcoords="offset points",fontsize=sc_label_size,color=TXT_SC,fontweight="semibold",zorder=5)
            t.set_path_effects([pe.withStroke(linewidth=2.0,foreground=STROKE_SC,alpha=0.9)])

    ax_sc.set_xlabel(mlabel(sc_x),color=TXT_SC,fontsize=14,fontweight="semibold")
    ax_sc.set_ylabel(mlabel(sc_y),color=TXT_SC,fontsize=14,fontweight="semibold")
    ax_sc.grid(True,color=GRID_SC,lw=0.9)
    for s in ax_sc.spines.values(): s.set_color("#9ca3af"); s.set_linewidth(0.9)
    for tick in ax_sc.get_xticklabels()+ax_sc.get_yticklabels(): tick.set_color(TXT_SC); tick.set_fontweight("semibold")

    top_frac_sc=1.0-(top_gap_sc/float(h_sc))
    fig_sc.subplots_adjust(left=0.075,right=0.985,bottom=0.105,top=top_frac_sc)
    if show_sc_title and sc_custom_title.strip():
        fig_sc.text(0.5,top_frac_sc+(1-top_frac_sc)*0.44,sc_custom_title.strip(),
                    ha="center",va="center",color=TXT_SC,fontsize=24,fontweight="semibold")

    buf_sc=io.BytesIO()
    fig_sc.savefig(buf_sc,format="png",dpi=100,facecolor=PBG_SC,bbox_inches=None,pad_inches=0)
    buf_sc.seek(0)
    st.image(buf_sc,width=w_sc)
    st.download_button("⬇️ Download Scatter",buf_sc.getvalue(),"team_scatter.png","image/png",key="ts_sc_dl")
    plt.close(fig_sc)

# ══════════════════════════════════════════════════════
# SECTION 9 – COMPARISON RADAR (player-style)
# Paste this after Section 8 in team_hq_part1.py
# Remove the final st.markdown("---") and st.info line from Part 1 first
# ══════════════════════════════════════════════════════
st.subheader("⚡ Team Comparison Radar")

# Metrics: xGA + Goals vs sit between PPDA and Possession
# Points/xPoints are divided by Matches for per-game rate
RADAR_COMP_CLEAN = [
    ("xG",          "xG p90",                   False),
    ("Goals",       "Goals p90",                False),
    ("Touches",     "Touches in Box p90",        False),
    ("PPDA",        "PPDA",                      True),
    ("xGA",         "xG Against p90",             True),
    ("Goals vs",    "Goals Against p90",          True),
    ("Possession",  "Possession %",              False),
    ("Passes",      "Passes p90",                False),
    ("Final 3rd",   "Passes to Final Third p90", False),
    ("Pts/game",    "Points",                    False),
    ("xPts/game",   "Expected Points",           False),
]

with st.expander("Radar settings", expanded=False):
    comp_theme = st.radio("Theme", ["Light","Dark"], horizontal=True, key="ts_comp_theme")
    edit_team_a_name = st.checkbox("Edit Team A display name", False, key="ts_comp_edit_a")
    edit_team_b_name = st.checkbox("Edit Team B display name", False, key="ts_comp_edit_b")

if comp_theme == "Dark":
    PBG_C="#0a0f1c"; AX_C="#0a0f1c"; LABEL_C_R="#f5f5f5"; TICK_C="#e5e7eb"
    RING_IN="#3a4050"; RING_OUT="#cbd5e1"
else:
    PBG_C="#ffffff"; AX_C="#ebebeb"; LABEL_C_R="#0f172a"; TICK_C="#6b7280"
    RING_IN=RING_OUT="#d1d5db"

team_a = st.selectbox("Team A (red)",  team_options, key="ts_comp_a")
team_b = st.selectbox("Team B (blue)", [t for t in team_options if t != team_a], key="ts_comp_b")

team_a_display = st.text_input("Team A display name", team_a, key="ts_comp_a_name") if edit_team_a_name else team_a
team_b_display = st.text_input("Team B display name", team_b, key="ts_comp_b_name") if edit_team_b_name else team_b

row_a = df[df["Team"] == team_a].iloc[0] if not df[df["Team"] == team_a].empty else None
row_b = df[df["Team"] == team_b].iloc[0] if not df[df["Team"] == team_b].empty else None

if row_a is not None and row_b is not None:
    leagues_ab = {str(row_a.get("League","")), str(row_b.get("League",""))}
    pool_ab    = df[df["League"].isin(leagues_ab)]

    radar_comp_avail = [(lab,col,inv) for lab,col,inv in RADAR_COMP_CLEAN if col in df.columns]

    # Points / xPoints → per-game rate
    def adjusted_val(row, col):
        v = row.get(col, np.nan)
        if pd.isna(v): return np.nan
        v = float(v)
        if col in ("Points","Expected Points"):
            m = float(row.get("Matches",1) or 1)
            return v/m if m>0 else v
        return v

    def pct_comp(t_row, col, invert=False):
        if col not in pool_ab.columns: return 50.0
        v = adjusted_val(t_row, col)
        if pd.isna(v): return 50.0
        if col in ("Points","Expected Points"):
            m_series = pd.to_numeric(pool_ab.get("Matches", pd.Series([1]*len(pool_ab))), errors="coerce").replace(0,1)
            s = pd.to_numeric(pool_ab[col], errors="coerce") / m_series
        else:
            s = pd.to_numeric(pool_ab[col], errors="coerce")
        s = s.dropna()
        if s.empty: return 50.0
        p = (s < v).mean()*100 + (s==v).mean()*50
        return float(np.clip((100-p) if invert else p, 0, 100))

    A_r = np.array([pct_comp(row_a, col, inv) for _,col,inv in radar_comp_avail])
    B_r = np.array([pct_comp(row_b, col, inv) for _,col,inv in radar_comp_avail])
    labels_c = [lab for lab,_,_ in radar_comp_avail]

    def fmt_val_comp(row, col):
        v = adjusted_val(row, col)
        if pd.isna(v): return "—"
        return f"{v:.2f}".rstrip("0").rstrip(".")

    N_c = len(radar_comp_avail)
    theta_c  = np.linspace(0, 2*np.pi, N_c, endpoint=False)
    theta_cc = np.concatenate([theta_c, theta_c[:1]])
    Ar_c = np.concatenate([A_r, A_r[:1]])
    Br_c = np.concatenate([B_r, B_r[:1]])

    COL_A="#C81E1E"; COL_B="#1D4ED8"
    FILL_A=(200/255,30/255,30/255,0.55); FILL_B=(29/255,78/255,216/255,0.55)
    INNER_HOLE=10

    fig_c = plt.figure(figsize=(13.2, 8.0), dpi=220)
    fig_c.patch.set_facecolor(PBG_C)
    ax_c = plt.subplot(111, polar=True); ax_c.set_facecolor(AX_C)
    ax_c.set_theta_offset(np.pi/2); ax_c.set_theta_direction(-1)
    ax_c.set_xticks(theta_c); ax_c.set_xticklabels([])
    ax_c.set_yticks([]); ax_c.grid(False)
    for s in ax_c.spines.values(): s.set_visible(False)

    ring_edges = np.linspace(INNER_HOLE, 100, 11)
    for i in range(10):
        r0,r1 = ring_edges[i], ring_edges[i+1]
        if comp_theme == "Dark":
            band = "#162235" if (9-i)%2==0 else AX_C
        else:
            band = "#e5e7eb" if (9-i)%2==0 else AX_C
        ax_c.add_artist(mpatches.Wedge((0,0), r1, 0, 360, width=(r1-r0),
            transform=ax_c.transData._b, facecolor=band, edgecolor="none", zorder=0.8))

    ring_t = np.linspace(0, 2*np.pi, 361)
    for j,r in enumerate(ring_edges):
        col_r = RING_OUT if j==len(ring_edges)-1 else RING_IN
        ax_c.plot(ring_t, np.full_like(ring_t, r), color=col_r, lw=1.0, zorder=0.9)

    for i,ang in enumerate(theta_c):
        for rr in ring_edges[2:]:
            pct_val = int(round((rr-INNER_HOLE)/(100-INNER_HOLE)*100))
            ax_c.text(ang, rr-1.8, f"{pct_val}", ha="center", va="center",
                      fontsize=6, color=TICK_C, zorder=1.1)

    OUTER_R = 107
    for ang,lab in zip(theta_c, labels_c):
        rot = np.degrees(ax_c.get_theta_direction()*ang + ax_c.get_theta_offset()) - 90
        rn  = ((rot+180)%360)-180
        if rn>90 or rn<-90: rot+=180
        ax_c.text(ang, OUTER_R, lab, rotation=rot, rotation_mode="anchor",
                  ha="center", va="center", fontsize=9, color=LABEL_C_R,
                  fontweight=600, clip_on=False, zorder=2.2)

    # Actual values shown in red/blue below the axis label
    for i,(ang,(lab,col,inv)) in enumerate(zip(theta_c, radar_comp_avail)):
        va_str = fmt_val_comp(row_a, col)
        vb_str = fmt_val_comp(row_b, col)
        ax_c.text(ang, OUTER_R+14, va_str, ha="center", va="center",
                  fontsize=7, color=COL_A, fontweight="bold", clip_on=False, zorder=2.3)
        ax_c.text(ang, OUTER_R+22, vb_str, ha="center", va="center",
                  fontsize=7, color=COL_B, fontweight="bold", clip_on=False, zorder=2.3)

    ax_c.add_artist(plt.Circle((0,0), radius=INNER_HOLE-0.6,
                               transform=ax_c.transData._b, color=PBG_C, zorder=1.2, ec="none"))

    ax_c.plot(theta_cc, Ar_c, color=COL_A, lw=2.2, zorder=3)
    ax_c.fill(theta_cc, Ar_c, color=FILL_A, zorder=2.5)
    ax_c.plot(theta_cc, Br_c, color=COL_B, lw=2.2, zorder=3)
    ax_c.fill(theta_cc, Br_c, color=FILL_B, zorder=2.5)
    ax_c.set_rlim(0,100)

    # Headers
    matches_a = int(row_a.get("Matches",0)) if pd.notna(row_a.get("Matches")) else 0
    matches_b = int(row_b.get("Matches",0)) if pd.notna(row_b.get("Matches")) else 0

    fig_c.text(0.12, 0.965, team_a_display, color=COL_A, fontsize=22, fontweight="bold", ha="left")
    fig_c.text(0.12, 0.935, str(row_a.get("League","")), color=COL_A, fontsize=11, ha="left")
    fig_c.text(0.12, 0.910, f"{matches_a} matches", color=COL_A, fontsize=10, ha="left")

    fig_c.text(0.88, 0.965, team_b_display, color=COL_B, fontsize=22, fontweight="bold", ha="right")
    fig_c.text(0.88, 0.935, str(row_b.get("League","")), color=COL_B, fontsize=11, ha="right")
    fig_c.text(0.88, 0.910, f"{matches_b} matches", color=COL_B, fontsize=10, ha="right")

    st.pyplot(fig_c, use_container_width=True)

    buf_c = io.BytesIO()
    fig_c.savefig(buf_c, format="png", dpi=220, facecolor=PBG_C)
    st.download_button(
        "⬇️ Download Comparison Radar", buf_c.getvalue(),
        f"comparison_{team_a.replace(' ','_')}_vs_{team_b.replace(' ','_')}.png",
        "image/png", key="ts_comp_dl"
    )
    plt.close(fig_c)
else:
    st.info("Select two teams above.")

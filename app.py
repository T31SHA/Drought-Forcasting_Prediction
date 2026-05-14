#  SPEI DROUGHT INTELLIGENCE PLATFORM
#  Northwestern Algeria · 35.75°N, 0.75°E · 1950–2026
#  Streamlit App —

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, cohen_kappa_score
)
import io

#  Page config 
st.set_page_config(
    page_title="SPEI Drought Intelligence Platform",
    page_icon="🌵",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Elite CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root & Reset ── */
:root {
    --bg:       #07090f;
    --bg2:      #0e1320;
    --bg3:      #141c2e;
    --border:   rgba(180,160,100,0.14);
    --sand:     #d4b896;
    --sand2:    #a8885a;
    --sand3:    #6b5230;
    --accent:   #c9923a;
    --text:     #e6d9c7;
    --text2:    rgba(230,217,199,0.6);
    --text3:    rgba(230,217,199,0.3);
    --red:      #e05252;
    --orange:   #e07a30;
    --yellow:   #d4b030;
    --blue:     #5baad4;
    --green:    #5fc490;
    --r: DM Serif Display, Georgia, serif;
    --s: DM Sans, sans-serif;
    --m: DM Mono, monospace;
}

html, body, [class*="css"] {
    font-family: var(--s);
    background-color: var(--bg);
    color: var(--text);
}

/* ── Streamlit overrides ── */
.stApp { background-color: var(--bg); }
section[data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
.stButton > button {
    font-family: var(--m);
    font-size: 11px;
    letter-spacing: .08em;
    text-transform: uppercase;
    background: var(--bg3);
    border: 1px solid var(--border);
    color: var(--sand);
    border-radius: 4px;
    padding: .45rem 1.1rem;
    transition: all .2s;
}
.stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(201,146,58,.06);
}
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg3);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 4px;
}
.stSlider [data-baseweb="slider"] { padding: 0 2px; }
div[data-testid="stMetricValue"]  {
    font-family: var(--m);
    font-size: 1.6rem;
    color: var(--sand);
}
div[data-testid="stMetricLabel"]  {
    font-family: var(--m);
    font-size: .68rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text3);
}
div[data-testid="stMetricDelta"]  {
    font-family: var(--m);
    font-size: .78rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--m);
    font-size: 10px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text3);
    background: transparent;
    border: none;
    padding: .6rem 1.1rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
    color: var(--sand) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}
.stDataFrame, .dataframe {
    font-family: var(--m);
    font-size: 12px;
    background: var(--bg2);
}
hr { border-color: var(--border); margin: 1.5rem 0; }

/* ── Custom components ── */
.hero {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-tag {
    font-family: var(--m);
    font-size: 10px;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .6rem;
}
.hero-title {
    font-family: var(--r);
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 400;
    color: var(--text);
    line-height: 1.1;
    margin-bottom: .5rem;
}
.hero-title em { color: var(--accent); font-style: italic; }
.hero-sub {
    font-family: var(--s);
    font-size: .9rem;
    color: var(--text2);
    font-weight: 300;
    letter-spacing: .02em;
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin: 1.5rem 0;
}
.kpi {
    background: var(--bg2);
    padding: 1rem 1.2rem;
}
.kpi-l {
    font-family: var(--m);
    font-size: 9px;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 5px;
}
.kpi-v {
    font-family: var(--m);
    font-size: 1.5rem;
    font-weight: 500;
    color: var(--sand);
    line-height: 1;
}
.kpi-s {
    font-family: var(--s);
    font-size: .7rem;
    color: var(--text2);
    margin-top: 3px;
}
.section-label {
    font-family: var(--m);
    font-size: 9px;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: .8rem;
    padding-bottom: .4rem;
    border-bottom: 1px solid var(--border);
}
.insight-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin-bottom: .75rem;
}
.insight-card .ic-label {
    font-family: var(--m);
    font-size: 9px;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: .4rem;
}
.insight-card .ic-val {
    font-family: var(--m);
    font-size: 1.3rem;
    font-weight: 500;
}
.insight-card .ic-sub {
    font-family: var(--s);
    font-size: .75rem;
    color: var(--text2);
    margin-top: 3px;
}
.badge {
    display: inline-block;
    font-family: var(--m);
    font-size: 10px;
    letter-spacing: .08em;
    padding: 2px 8px;
    border-radius: 3px;
    border: 1px solid;
}
.badge-red    { color: var(--red);    border-color: rgba(224,82,82,.35);    background: rgba(224,82,82,.07);    }
.badge-orange { color: var(--orange); border-color: rgba(224,122,48,.35);   background: rgba(224,122,48,.07);   }
.badge-yellow { color: var(--yellow); border-color: rgba(212,176,48,.35);   background: rgba(212,176,48,.07);   }
.badge-blue   { color: var(--blue);   border-color: rgba(91,170,212,.35);   background: rgba(91,170,212,.07);   }
.badge-green  { color: var(--green);  border-color: rgba(95,196,144,.35);   background: rgba(95,196,144,.07);   }
.timeline-item {
    display: flex;
    gap: 1rem;
    padding: .6rem 0;
    border-bottom: 1px solid var(--border);
    align-items: center;
}
.timeline-date {
    font-family: var(--m);
    font-size: 11px;
    color: var(--text3);
    width: 70px;
    flex-shrink: 0;
}
.timeline-val {
    font-family: var(--m);
    font-size: 12px;
    font-weight: 500;
    width: 55px;
    flex-shrink: 0;
}
.timeline-bar {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,.06);
    border-radius: 3px;
    overflow: hidden;
}
.timeline-fill {
    height: 100%;
    border-radius: 3px;
}
.sidebar-logo {
    font-family: var(--r);
    font-size: 1.15rem;
    color: var(--text);
    margin-bottom: 4px;
}
.sidebar-sub {
    font-family: var(--m);
    font-size: 9px;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--text3);
}
.nav-item {
    font-family: var(--m);
    font-size: 10px;
    letter-spacing: .1em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)



#  HELPERS & CONSTANTS
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,19,32,0.6)",
    font=dict(family="DM Mono, monospace", color="#c8bba8", size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(7,9,15,0.7)", bordercolor="rgba(180,160,100,0.2)",
                borderwidth=1, font=dict(size=10)),
)

# Shared axis style — used via **AXIS_STYLE, never inside PLOTLY_LAYOUT
# (avoids "multiple values for keyword argument" when passing to update_layout)
AXIS_STYLE = dict(
    gridcolor="rgba(180,160,100,0.08)",
    zeroline=False,
    linecolor="rgba(180,160,100,0.15)",
)

DROUGHT_COLORS = {
    "Extreme drought":  "#e05252",
    "Severe drought":   "#e07a30",
    "Moderate drought": "#d4b030",
    "Near normal":      "#6a7a8a",
    "Moderately wet":   "#5baad4",
    "Severely wet":     "#3a85b8",
    "Extremely wet":    "#2469a0",
}

def spei_color(v):
    if pd.isna(v):      return "#1a2235"
    if v < -2.0:        return "#e05252"
    if v < -1.5:        return "#e07a30"
    if v < -1.0:        return "#d4b030"
    if v < +1.0:        return "#3a4a5a"
    if v < +1.5:        return "#5baad4"
    return                     "#3a85b8"

def classify_spei(v):
    if v >= 2.0:  return "Extremely wet"
    if v >= 1.5:  return "Severely wet"
    if v >= 1.0:  return "Moderately wet"
    if v > -1.0:  return "Near normal"
    if v > -1.5:  return "Moderate drought"
    if v > -2.0:  return "Severe drought"
    return               "Extreme drought"

def fmt_spei(v):
    if pd.isna(v): return "N/A"
    return f"{v:+.3f}"

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]



#  DATA LOADING & FEATURE ENGINEERING 
@st.cache_data(show_spinner=False)
def load_and_engineer(raw_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df["DATE"] = pd.to_datetime(df["DATA"], format="%b%Y")
    df = df.drop(columns=["DATA"]).set_index("DATE")
    df.index.freq = "MS"
    df = df.sort_index()

    # Temporal
    df["month"]             = df.index.month
    df["year"]              = df.index.year
    df["decade"]            = (df.index.year // 10) * 10
    df["month_since_start"] = np.arange(len(df))
    df["month_sin"]         = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]         = np.cos(2 * np.pi * df["month"] / 12)

    # Lags
    for lag in [1, 2, 3, 6, 9, 12, 24]:
        df[f"SPEI_12_lag_{lag}"] = df["SPEI_12"].shift(lag)
    for lag in [1, 3]:
        df[f"SPEI_1_lag_{lag}"]  = df["SPEI_1"].shift(lag)
        df[f"SPEI_6_lag_{lag}"]  = df["SPEI_6"].shift(lag)

    # Rolling stats
    for w in [3, 6, 12]:
        df[f"SPEI_12_roll_mean_{w}"] = df["SPEI_12"].shift(1).rolling(w, min_periods=1).mean()
    for w in [3, 6]:
        df[f"SPEI_12_roll_std_{w}"]  = df["SPEI_12"].shift(1).rolling(w, min_periods=1).std()
    df["SPEI_12_roll_min_12"] = df["SPEI_12"].shift(1).rolling(12, min_periods=1).min()
    df["SPEI_12_roll_max_12"] = df["SPEI_12"].shift(1).rolling(12, min_periods=1).max()

    # Interactions
    df["SPEI_1_minus_12"]  = df["SPEI_1"]  - df["SPEI_12"]
    df["SPEI_6_minus_24"]  = df["SPEI_6"]  - df["SPEI_24"]
    df["SPEI_12_times_24"] = df["SPEI_12"] * df["SPEI_24"]

    # Targets
    df["y_t1"] = df["SPEI_12"].shift(-1)
    df["y_t3"] = df["SPEI_12"].shift(-3)
    df["y_t6"] = df["SPEI_12"].shift(-6)

    TARGET_COLS  = ["y_t1", "y_t3", "y_t6"]
    SPEI_RAW     = [c for c in df.columns if c.startswith("SPEI_") and "_lag_" not in c
                    and "roll" not in c and "minus" not in c and "times" not in c
                    and c not in ["SPEI_12"]]
    FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS + SPEI_RAW]
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
    return df


@st.cache_data(show_spinner=False)
def split_and_scale(df_bytes: bytes):
    df    = load_and_engineer(df_bytes)
    TARGS = ["y_t1", "y_t3", "y_t6"]
    SPEI_RAW = [c for c in df.columns if c.startswith("SPEI_") and "_lag_" not in c
                and "roll" not in c and "minus" not in c and "times" not in c
                and c not in ["SPEI_12"]]
    FEATS = [c for c in df.columns if c not in TARGS + SPEI_RAW]

    tr = df.index <= "2005-12-01"
    va = (df.index > "2005-12-01") & (df.index <= "2015-12-01")
    te = df.index > "2015-12-01"

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(df.loc[tr, FEATS])
    X_va_s   = scaler.transform(df.loc[va, FEATS])
    X_te_s   = scaler.transform(df.loc[te, FEATS])

    return dict(
        df=df, feats=FEATS,
        X_tr=X_tr_s, X_va=X_va_s, X_te=X_te_s,
        y_tr=df.loc[tr, TARGS], y_va=df.loc[va, TARGS], y_te=df.loc[te, TARGS],
        idx_tr=df.index[tr], idx_va=df.index[va], idx_te=df.index[te],
        scaler=scaler,
    )


@st.cache_data(show_spinner=False)
def train_models(df_bytes: bytes):
    d     = split_and_scale(df_bytes)
    results = {}

    for col, label in [("y_t1","t+1"), ("y_t3","t+3"), ("y_t6","t+6")]:
        y_tr = d["y_tr"][col].values
        y_va = d["y_va"][col].values
        y_te = d["y_te"][col].values

        # ── XGBoost (fast, no Optuna for UI speed) ──
        try:
            import xgboost as xgb
            m = xgb.XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.8,
                reg_alpha=0.1, random_state=42, verbosity=0, n_jobs=-1
            )
            m.fit(d["X_tr"], y_tr, eval_set=[(d["X_va"], y_va)],
                  verbose=False)
            pte = m.predict(d["X_te"])
            results[(label, "XGBoost")] = {
                "preds": pte, "model": m,
                "metrics": _metrics(y_te, pte),
                "fi": dict(zip(d["feats"], m.feature_importances_)),
            }
        except Exception as e:
            results[(label, "XGBoost")] = {"preds": np.zeros_like(y_te), "metrics": None}

        # ── Ridge ──
        try:
            best_a, best_r = 1.0, np.inf
            for a in [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]:
                rm = Ridge(alpha=a)
                rm.fit(d["X_tr"], y_tr)
                rp = rm.predict(d["X_va"])
                rv = np.sqrt(mean_squared_error(y_va, rp))
                if rv < best_r:
                    best_r, best_a = rv, a
            best_m = Ridge(alpha=best_a)
            best_m.fit(np.vstack([d["X_tr"], d["X_va"]]),
                       np.concatenate([y_tr, y_va]))
            pte = best_m.predict(d["X_te"])
            results[(label, "Ridge")] = {
                "preds": pte, "model": best_m,
                "metrics": _metrics(y_te, pte),
                "fi": dict(zip(d["feats"], np.abs(best_m.coef_))),
            }
        except Exception as e:
            results[(label, "Ridge")] = {"preds": np.zeros_like(y_te), "metrics": None}

        # ── Bootstrap uncertainty (Ridge t+1 only) ──
        if col == "y_t1":
            try:
                X_all = np.vstack([d["X_tr"], d["X_va"]])
                y_all = np.concatenate([y_tr, y_va])
                bp    = []
                rng   = np.random.default_rng(42)
                for _ in range(100):
                    idx  = rng.integers(0, len(X_all), size=len(X_all))
                    bm   = Ridge(alpha=best_a)
                    bm.fit(X_all[idx], y_all[idx])
                    bp.append(bm.predict(d["X_te"]))
                bp = np.array(bp)
                results["bootstrap"] = {
                    "mean": np.mean(bp, axis=0),
                    "p05":  np.percentile(bp, 5,  axis=0),
                    "p25":  np.percentile(bp, 25, axis=0),
                    "p75":  np.percentile(bp, 75, axis=0),
                    "p95":  np.percentile(bp, 95, axis=0),
                }
            except Exception:
                pass

    return results, d


def _metrics(y_true, y_pred):
    rmse     = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae      = float(mean_absolute_error(y_true, y_pred))
    r2       = float(r2_score(y_true, y_pred))
    mape     = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    bias     = float(np.mean(y_pred - y_true))
    hit      = float(np.mean(((y_pred < -1) == (y_true < -1))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2,
            "MAPE": mape, "Bias": bias, "HitRate": hit}


#  SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='padding:.6rem 0 1.2rem'>
      <div class='sidebar-logo'>🌵 Drought Intelligence</div>
      <div class='sidebar-sub'>SPEI Platform · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload SPEI CSV",
        type=["csv"],
        help="CSV with DATA column (MonYYYY format) and SPEI_1 through SPEI_48 columns",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🌍  Overview", "📊  Data Explorer", "🤖  Model Performance",
         "🎯  Forecast & Uncertainty", "🔬  Feature Intelligence"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:var(--m);font-size:9px;letter-spacing:.1em;
                color:rgba(230,217,199,.25);line-height:1.9'>
    35.75°N &nbsp;·&nbsp; 0.75°E<br>
    Jan 1950 → Feb 2026<br>
    914 monthly observations<br>
    SPEI-1 through SPEI-48
    </div>
    """, unsafe_allow_html=True)


#  GATE: require upload


if uploaded is None:
    st.markdown("""
    <div class='hero'>
      <div class='hero-tag'>⬡ SPEI Drought Intelligence Platform</div>
      <div class='hero-title'>Predict drought.<br><em>Before it arrives.</em></div>
      <div class='hero-sub'>Multi-horizon forecasting · Northwestern Algeria · 1950–2026</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='insight-card'>
          <div class='ic-label'>What this platform does</div>
          <div class='ic-val' style='font-size:1rem;color:var(--text)'>
            Ingests raw SPEI data, engineers 30+ features, trains 2 ML models
            across 3 forecast horizons, and quantifies prediction uncertainty.
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='insight-card'>
          <div class='ic-label'>Models trained</div>
          <div class='ic-val' style='color:var(--accent)'>Ridge &amp; XGBoost</div>
          <div class='ic-sub'>t+1 · t+3 · t+6 month horizons · Bootstrap 90% prediction intervals</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='insight-card'>
          <div class='ic-label'>How to start</div>
          <div class='ic-val' style='font-size:1rem;color:var(--text)'>
            Upload your <code style='font-family:var(--m);color:var(--accent)'>SPEI_*.csv</code>
            file in the sidebar. All analysis runs automatically.
          </div>
        </div>""", unsafe_allow_html=True)

    st.info("👈  Upload your SPEI CSV in the sidebar to launch the platform.", icon="📂")
    st.stop()


#  LOAD DATA


raw_bytes = uploaded.read()

with st.spinner("Loading & engineering features…"):
    df = load_and_engineer(raw_bytes)

spei12      = df["SPEI_12"].dropna()
all_spei    = [c for c in df.columns if c.startswith("SPEI_") and len(c) <= 8
               and "_lag" not in c and "roll" not in c
               and "minus" not in c and "times" not in c]
spei_windows = sorted(all_spei, key=lambda x: int(x.split("_")[1]))



#  PAGE: OVERVIEW

if page == "🌍  Overview":

    st.markdown("""
    <div class='hero'>
      <div class='hero-tag'>⬡ Northwestern Algeria · 35.75°N, 0.75°E</div>
      <div class='hero-title'>SPEI Drought<br><em>Observatory</em></div>
      <div class='hero-sub'>Jan 1950 → Feb 2026 &nbsp;·&nbsp; 914 monthly observations &nbsp;·&nbsp; SPEI-1 through SPEI-48</div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ──
    n_extreme  = int((spei12 < -2.0).sum())
    n_severe   = int(((spei12 >= -2) & (spei12 < -1.5)).sum())
    n_moderate = int(((spei12 >= -1.5) & (spei12 < -1.0)).sum())
    worst_date = spei12.idxmin().strftime("%b %Y")
    worst_val  = spei12.min()
    recent     = spei12.iloc[-1]
    decade_20s = spei12[spei12.index.year >= 2020].mean()

    st.markdown(f"""
    <div class='kpi-grid'>
      <div class='kpi'>
        <div class='kpi-l'>Total record</div>
        <div class='kpi-v'>{len(spei12)}</div>
        <div class='kpi-s'>Monthly observations</div>
      </div>
      <div class='kpi'>
        <div class='kpi-l'>Extreme drought months</div>
        <div class='kpi-v' style='color:var(--red)'>{n_extreme}</div>
        <div class='kpi-s'>SPEI-12 &lt; −2.0 &nbsp;({100*n_extreme/len(spei12):.1f}%)</div>
      </div>
      <div class='kpi'>
        <div class='kpi-l'>Severe drought months</div>
        <div class='kpi-v' style='color:var(--orange)'>{n_severe}</div>
        <div class='kpi-s'>SPEI-12 −2.0 to −1.5</div>
      </div>
      <div class='kpi'>
        <div class='kpi-l'>Historical minimum</div>
        <div class='kpi-v' style='color:var(--red)'>{worst_val:+.3f}</div>
        <div class='kpi-s'>{worst_date}</div>
      </div>
      <div class='kpi'>
        <div class='kpi-l'>Latest reading</div>
        <div class='kpi-v' style='color:{"var(--blue)" if recent>0 else "var(--orange)"}'>{recent:+.3f}</div>
        <div class='kpi-s'>{spei12.index[-1].strftime("%b %Y")} · {classify_spei(recent)}</div>
      </div>
      <div class='kpi'>
        <div class='kpi-l'>2020s decade avg</div>
        <div class='kpi-v' style='color:var(--blue)'>{decade_20s:+.3f}</div>
        <div class='kpi-s'>Wettest decade on record</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main time series chart ──
    st.markdown("<p class='section-label'>SPEI-12 · Full Record (1950–2026)</p>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_hrect(y0=-4, y1=-2.0, fillcolor="#e05252", opacity=0.07, line_width=0)
    fig.add_hrect(y0=-2.0, y1=-1.5, fillcolor="#e07a30", opacity=0.07, line_width=0)
    fig.add_hrect(y0=-1.5, y1=-1.0, fillcolor="#d4b030", opacity=0.07, line_width=0)
    fig.add_hrect(y0=1.0, y1=4,    fillcolor="#5baad4", opacity=0.05, line_width=0)

    colors_above = ["#5baad4" if v >= 0 else "#e07a30" for v in spei12.values]
    for i in range(len(spei12) - 1):
        fig.add_trace(go.Scatter(
            x=spei12.index[i:i+2], y=spei12.values[i:i+2],
            mode="lines", line=dict(color=colors_above[i], width=1.2),
            showlegend=False, hoverinfo="skip",
        ))

    # Rolling 10yr
    roll = spei12.rolling(120, center=True, min_periods=60).mean()
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll.values, mode="lines",
        name="10-yr rolling mean",
        line=dict(color="#e6d9c7", width=2.5, dash="solid"),
    ))

    fig.add_hline(y=0,    line_dash="dot", line_color="rgba(230,217,199,0.25)", line_width=1)
    fig.add_hline(y=-1.0, line_dash="dot", line_color="rgba(212,176,48,0.5)",  line_width=1)
    fig.add_hline(y=-2.0, line_dash="dot", line_color="rgba(224,82,82,0.5)",   line_width=1)

    fig.update_layout(**PLOTLY_LAYOUT, height=340,
                      yaxis=dict(**AXIS_STYLE, range=[-3.5, 4.0],
                                 title="SPEI-12 (z-score)"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Decade bar + worst events side by side ──
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("<p class='section-label'>Decade mean SPEI-12</p>", unsafe_allow_html=True)
        df_dec = df[["SPEI_12","decade"]].dropna()
        dec_avg = df_dec.groupby("decade")["SPEI_12"].mean().reset_index()
        dec_avg.columns = ["Decade","Mean SPEI-12"]
        dec_avg["label"] = dec_avg["Decade"].astype(str) + "s"
        dec_avg["color"] = dec_avg["Mean SPEI-12"].apply(
            lambda v: "#e07a30" if v < -0.1 else "#5baad4" if v > 0.2 else "#6a7a8a")
        fig2 = go.Figure(go.Bar(
            x=dec_avg["label"], y=dec_avg["Mean SPEI-12"],
            marker_color=dec_avg["color"], marker_line_width=0,
            text=dec_avg["Mean SPEI-12"].round(3),
            textposition="outside",
            textfont=dict(size=10, family="DM Mono"),
        ))
        fig2.add_hline(y=0, line_dash="dot", line_color="rgba(230,217,199,0.3)", line_width=1)
        fig2.update_layout(**PLOTLY_LAYOUT, height=280,
                           yaxis=dict(**AXIS_STYLE, title="Mean SPEI-12"))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("<p class='section-label'>Most extreme drought months</p>", unsafe_allow_html=True)
        worst10 = spei12.nsmallest(10).reset_index()
        worst10.columns = ["Date", "SPEI-12"]
        worst10["Date_str"] = worst10["Date"].dt.strftime("%b %Y")
        worst10["Category"] = worst10["SPEI-12"].apply(classify_spei)
        for _, row in worst10.iterrows():
            bar_w = min(abs(row["SPEI-12"]) / 3.0, 1.0) * 100
            color = spei_color(row["SPEI-12"])
            st.markdown(f"""
            <div class='timeline-item'>
              <div class='timeline-date'>{row["Date_str"]}</div>
              <div class='timeline-val' style='color:{color}'>{row["SPEI-12"]:+.3f}</div>
              <div class='timeline-bar'>
                <div class='timeline-fill' style='width:{bar_w:.1f}%;background:{color}'></div>
              </div>
            </div>""", unsafe_allow_html=True)



#  PAGE: DATA EXPLORER

elif page == "📊  Data Explorer":

    st.markdown("<h2 style='font-family:var(--r);font-weight:400;color:var(--text);margin-bottom:.3rem'>Data Explorer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text2);font-size:.88rem;margin-bottom:1.5rem'>Interactive exploration of the SPEI record across all accumulation windows.</p>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["  Multi-Scale  ", "  Calendar Heatmap  ", "  Distribution  ", "  Seasonality  "])

    # ── Tab 1: Multi-scale ──
    with tab1:
        sel_windows = st.multiselect(
            "SPEI windows to display",
            options=spei_windows,
            default=["SPEI_1","SPEI_6","SPEI_12","SPEI_24"],
        )
        date_range = st.slider(
            "Year range",
            int(df.index.year.min()), int(df.index.year.max()),
            (int(df.index.year.min()), int(df.index.year.max())),
        )
        mask = (df.index.year >= date_range[0]) & (df.index.year <= date_range[1])
        df_plot = df.loc[mask]

        if sel_windows:
            fig = go.Figure()
            fig.add_hrect(y0=-4, y1=-2.0, fillcolor="#e05252", opacity=0.06, line_width=0)
            fig.add_hrect(y0=-2.0, y1=-1.5, fillcolor="#e07a30", opacity=0.06, line_width=0)
            fig.add_hrect(y0=-1.5, y1=-1.0, fillcolor="#d4b030", opacity=0.06, line_width=0)

            palette = ["#e6d9c7","#5baad4","#e07a30","#5fc490",
                       "#d4b030","#e05252","#a87ad4","#d46060"]
            for i, w in enumerate(sel_windows):
                if w not in df_plot.columns: continue
                s = df_plot[w].dropna()
                fig.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=w, mode="lines",
                    line=dict(color=palette[i % len(palette)],
                              width=2.0 if w == "SPEI_12" else 1.0,
                              dash="solid"),
                    opacity=0.9 if w == "SPEI_12" else 0.6,
                ))
            fig.add_hline(y=0,  line_dash="dot", line_color="rgba(230,217,199,0.25)", line_width=1)
            fig.add_hline(y=-1, line_dash="dot", line_color="rgba(212,176,48,0.5)",   line_width=1)
            fig.add_hline(y=-2, line_dash="dot", line_color="rgba(224,82,82,0.5)",    line_width=1)
            fig.update_layout(**PLOTLY_LAYOUT, height=380,
                              yaxis=dict(**AXIS_STYLE,
                                         title="SPEI (z-score)", range=[-3.5, 4.0]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one SPEI window above.")

    # ── Tab 2: Calendar Heatmap ──
    with tab2:
        w_sel = st.selectbox("SPEI window", options=spei_windows, index=spei_windows.index("SPEI_12") if "SPEI_12" in spei_windows else 0)
        s = df[w_sel].dropna()
        years = sorted(s.index.year.unique())
        pivot = pd.DataFrame(index=years, columns=range(1, 13), dtype=float)
        for dt, val in s.items():
            pivot.loc[dt.year, dt.month] = val

        z    = pivot.values.tolist()
        text = [[fmt_spei(v) for v in row] for row in pivot.values]

        colorscale = [
            [0.0,  "#e05252"],
            [0.2,  "#e07a30"],
            [0.35, "#d4b030"],
            [0.5,  "#2a3d50"],
            [0.65, "#3a6080"],
            [0.8,  "#5baad4"],
            [1.0,  "#3a85b8"],
        ]
        fig = go.Figure(go.Heatmap(
            z=z, x=MONTH_NAMES, y=[str(y) for y in years],
            text=text, texttemplate="%{text}",
            textfont=dict(size=7, family="DM Mono"),
            colorscale=colorscale, zmid=0,
            zmin=-2.5, zmax=2.5,
            colorbar=dict(title=w_sel, tickfont=dict(size=9, family="DM Mono"),
                          tickvals=[-2,-1,0,1,2]),
            hovertemplate="<b>%{y} %{x}</b><br>SPEI: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=max(500, len(years) * 10 + 100),
            yaxis=dict(**AXIS_STYLE, autorange="reversed",
                       title="", tickfont=dict(size=8)),
            xaxis=dict(**AXIS_STYLE, title=""),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Distribution ──
    with tab3:
        w_dist = st.selectbox("Window", spei_windows,
                              index=spei_windows.index("SPEI_12") if "SPEI_12" in spei_windows else 0,
                              key="dist_w")
        vals = df[w_dist].dropna().values
        x_range = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)
        kde  = stats.gaussian_kde(vals)(x_range)
        norm = stats.norm.pdf(x_range, 0, 1)

        stat_ks, p_ks = stats.kstest(vals, "norm", args=(vals.mean(), vals.std()))
        stat_sw, p_sw = stats.shapiro(vals[:5000])

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals, histnorm="probability density", name="Empirical",
            marker_color="#5baad4", opacity=0.45, nbinsx=60,
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=kde, name="KDE",
            line=dict(color="#e6d9c7", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=norm, name="N(0,1) theoretical",
            line=dict(color="#e07a30", width=2, dash="dash"),
        ))
        fig.add_vline(x=0,  line_dash="dot", line_color="rgba(230,217,199,0.3)")
        fig.add_vline(x=-1, line_dash="dot", line_color="rgba(212,176,48,0.5)")
        fig.add_vline(x=-2, line_dash="dot", line_color="rgba(224,82,82,0.5)")
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          title=dict(text=f"{w_dist} Distribution",
                                     font=dict(size=13, family="DM Serif Display")))
        st.plotly_chart(fig, use_container_width=True)

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Mean",     f"{vals.mean():.4f}")
        cB.metric("Std Dev",  f"{vals.std():.4f}")
        cC.metric("KS p-val", f"{p_ks:.4f}", delta="normal ✓" if p_ks > 0.05 else "non-normal")
        cD.metric("Shapiro p",f"{p_sw:.4f}", delta="normal ✓" if p_sw > 0.05 else "non-normal")

    # ── Tab 4: Seasonality ──
    with tab4:
        w_seas = st.selectbox("Window", spei_windows,
                              index=spei_windows.index("SPEI_12") if "SPEI_12" in spei_windows else 0,
                              key="seas_w")
        s_seas = df[w_seas].dropna()
        monthly_means = s_seas.groupby(s_seas.index.month).mean()
        monthly_stds  = s_seas.groupby(s_seas.index.month).std()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=MONTH_NAMES,
            y=(monthly_means + monthly_stds).values,
            mode="lines", line=dict(width=0), showlegend=False,
            fill=None, fillcolor="rgba(91,170,212,0.12)",
        ))
        fig.add_trace(go.Scatter(
            x=MONTH_NAMES,
            y=(monthly_means - monthly_stds).values,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(91,170,212,0.12)",
            name="±1 std dev",
        ))
        fig.add_trace(go.Scatter(
            x=MONTH_NAMES, y=monthly_means.values,
            mode="lines+markers", name="Monthly mean",
            line=dict(color="#5baad4", width=2.5),
            marker=dict(size=7, color="#5baad4"),
        ))
        fig.add_hline(y=0, line_dash="dot",
                      line_color="rgba(230,217,199,0.3)", line_width=1)
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          yaxis=dict(**AXIS_STYLE, title="SPEI (z-score)"))
        st.plotly_chart(fig, use_container_width=True)

        # Decade-boxplot
        st.markdown("<p class='section-label'>SPEI-12 distribution by decade</p>",
                    unsafe_allow_html=True)
        df_box = df[["SPEI_12","decade"]].dropna()
        fig_box = px.box(df_box, x="decade", y="SPEI_12",
                         color_discrete_sequence=["#5baad4"])
        fig_box.add_hline(y=0,  line_dash="dot", line_color="rgba(230,217,199,0.3)")
        fig_box.add_hline(y=-1, line_dash="dot", line_color="rgba(212,176,48,0.5)")
        fig_box.update_layout(**PLOTLY_LAYOUT, height=300,
                              yaxis=dict(**AXIS_STYLE, title="SPEI-12"),
                              xaxis=dict(**AXIS_STYLE, title="Decade"))
        st.plotly_chart(fig_box, use_container_width=True)



#  PAGE: MODEL PERFORMANCE

elif page == "🤖  Model Performance":

    st.markdown("<h2 style='font-family:var(--r);font-weight:400;color:var(--text);margin-bottom:.3rem'>Model Performance</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text2);font-size:.88rem;margin-bottom:1.5rem'>Ridge Regression and XGBoost evaluated across 3 forecast horizons on the held-out test set (Jan 2016 → Feb 2026).</p>", unsafe_allow_html=True)

    with st.spinner("Training models on your data…"):
        results, d = train_models(raw_bytes)

    # ── Metrics table ──
    st.markdown("<p class='section-label'>Metrics comparison — all models × all horizons</p>",
                unsafe_allow_html=True)

    rows = []
    for label in ["t+1","t+3","t+6"]:
        for model in ["Ridge","XGBoost"]:
            key = (label, model)
            if key in results and results[key]["metrics"]:
                m = results[key]["metrics"]
                rows.append({
                    "Horizon": label, "Model": model,
                    "RMSE":    round(m["RMSE"], 4),
                    "MAE":     round(m["MAE"],  4),
                    "R²":      round(m["R2"],   4),
                    "MAPE %":  round(m["MAPE"], 2),
                    "Bias":    round(m["Bias"], 4),
                    "Hit Rate %": round(m["HitRate"], 1),
                })
    metrics_df = pd.DataFrame(rows)

    if len(metrics_df):
        # Highlight best RMSE per horizon
        def color_cells(val):
            if isinstance(val, float):
                return ""
            return ""

        st.dataframe(
            metrics_df.style
                .format({"RMSE":"{:.4f}","MAE":"{:.4f}","R²":"{:.4f}",
                         "MAPE %":"{:.2f}","Bias":"{:+.4f}","Hit Rate %":"{:.1f}"})
                .background_gradient(subset=["RMSE","MAE"],   cmap="YlOrRd_r")
                .background_gradient(subset=["R²","Hit Rate %"], cmap="YlGn"),
            use_container_width=True, hide_index=True,
        )

    # ── Horizon degradation chart ──
    st.markdown("<p class='section-label' style='margin-top:1.5rem'>Performance vs forecast horizon</p>",
                unsafe_allow_html=True)

    fig_h = go.Figure()
    for model, color in [("Ridge","#5baad4"), ("XGBoost","#e07a30")]:
        rmses = [results.get((h, model), {}).get("metrics", {}).get("RMSE", None)
                 for h in ["t+1","t+3","t+6"]]
        r2s   = [results.get((h, model), {}).get("metrics", {}).get("R2", None)
                 for h in ["t+1","t+3","t+6"]]
        fig_h.add_trace(go.Scatter(
            x=["t+1","t+3","t+6"], y=rmses, name=f"{model} RMSE",
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=9, symbol="circle"),
        ))
    fig_h.update_layout(**PLOTLY_LAYOUT, height=280,
                        yaxis=dict(**AXIS_STYLE, title="RMSE"),
                        xaxis=dict(**AXIS_STYLE, title="Forecast horizon"))
    st.plotly_chart(fig_h, use_container_width=True)

    # ── Actual vs Predicted ──
    st.markdown("<p class='section-label'>Actual vs predicted — test set</p>",
                unsafe_allow_html=True)

    sel_h = st.selectbox("Horizon", ["t+1","t+3","t+6"], key="avp_h")
    sel_m = st.selectbox("Model",   ["Ridge","XGBoost"],  key="avp_m")
    key   = (sel_h, sel_m)

    if key in results and results[key].get("preds") is not None:
        y_te_col = {"t+1":"y_t1","t+3":"y_t3","t+6":"y_t6"}[sel_h]
        y_true   = d["y_te"][y_te_col].values
        y_pred   = results[key]["preds"]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=d["idx_te"], y=y_true, name="Actual",
                line=dict(color="#e6d9c7", width=1.8),
            ))
            fig_ts.add_trace(go.Scatter(
                x=d["idx_te"], y=y_pred, name="Predicted",
                line=dict(color="#e07a30", width=1.5, dash="dash"),
            ))
            fig_ts.add_hline(y=-1, line_dash="dot",
                             line_color="rgba(212,176,48,0.5)", line_width=1)
            fig_ts.add_hline(y=-2, line_dash="dot",
                             line_color="rgba(224,82,82,0.5)", line_width=1)
            fig_ts.update_layout(**PLOTLY_LAYOUT, height=300,
                                 title=dict(text=f"{sel_m} · {sel_h} · Test Period",
                                            font=dict(size=12)))
            st.plotly_chart(fig_ts, use_container_width=True)

        with c2:
            # Scatter
            lim = max(abs(y_true).max(), abs(y_pred).max()) + 0.3
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=y_true, y=y_pred, mode="markers",
                marker=dict(color="#5baad4", size=6, opacity=0.6,
                            line=dict(width=0)),
                name="Test points",
            ))
            fig_sc.add_trace(go.Scatter(
                x=[-lim, lim], y=[-lim, lim], mode="lines",
                line=dict(color="rgba(230,217,199,0.3)", dash="dash"),
                name="1:1 line",
            ))
            fig_sc.update_layout(**PLOTLY_LAYOUT, height=300,
                                 xaxis=dict(**AXIS_STYLE,
                                            title="Actual", range=[-lim, lim]),
                                 yaxis=dict(**AXIS_STYLE,
                                            title="Predicted", range=[-lim, lim]),
                                 title=dict(text=f"R² = {r2_score(y_true, y_pred):.3f}",
                                            font=dict(size=12)))
            st.plotly_chart(fig_sc, use_container_width=True)

        # Residual histogram
        resid = y_true - y_pred
        fig_r = go.Figure()
        fig_r.add_trace(go.Histogram(
            x=resid, name="Residuals", nbinsx=40,
            marker_color="#5fc490", opacity=0.7,
            histnorm="probability density",
        ))
        x_r = np.linspace(resid.min(), resid.max(), 200)
        fig_r.add_trace(go.Scatter(
            x=x_r, y=stats.norm.pdf(x_r, resid.mean(), resid.std()),
            mode="lines", name="Normal fit",
            line=dict(color="#e6d9c7", width=2),
        ))
        fig_r.add_vline(x=0, line_dash="dot",
                        line_color="rgba(230,217,199,0.4)")
        fig_r.update_layout(**PLOTLY_LAYOUT, height=240,
                            title=dict(text=f"Residual Distribution · Bias = {resid.mean():+.4f}",
                                       font=dict(size=11)))
        st.plotly_chart(fig_r, use_container_width=True)

    # ── WMO drought classification ──
    st.markdown("<p class='section-label' style='margin-top:1.5rem'>WMO drought classification performance</p>",
                unsafe_allow_html=True)

    best_key = ("t+1", "Ridge")
    if best_key in results and results[best_key].get("preds") is not None:
        y_true_t1 = d["y_te"]["y_t1"].values
        y_pred_t1 = results[best_key]["preds"]
        y_true_cls = [classify_spei(v) for v in y_true_t1]
        y_pred_cls = [classify_spei(v) for v in y_pred_t1]
        kappa = cohen_kappa_score(y_true_cls, y_pred_cls)

        all_classes = sorted(set(y_true_cls) | set(y_pred_cls))
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=all_classes)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        c1, c2 = st.columns([1.6, 1])
        with c1:
            fig_cm = go.Figure(go.Heatmap(
                z=cm_norm,
                x=all_classes, y=all_classes,
                text=[[f"{v:.2f}" for v in row] for row in cm_norm],
                texttemplate="%{text}",
                textfont=dict(size=10, family="DM Mono"),
                colorscale=[[0,"#0e1320"],[0.5,"#5baad4"],[1,"#e05252"]],
                zmin=0, zmax=1,
                colorbar=dict(tickfont=dict(size=9)),
                hovertemplate="True: %{y}<br>Pred: %{x}<br>Fraction: %{z:.3f}<extra></extra>",
            ))
            fig_cm.update_layout(
                **PLOTLY_LAYOUT, height=320,
                xaxis=dict(**AXIS_STYLE,
                           title="Predicted", tickangle=30, tickfont=dict(size=9)),
                yaxis=dict(**AXIS_STYLE,
                           title="Actual", tickfont=dict(size=9), autorange="reversed"),
                title=dict(text=f"Confusion Matrix · Cohen κ = {kappa:.3f}",
                           font=dict(size=12)),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with c2:
            st.metric("Cohen's Kappa", f"{kappa:.3f}",
                      delta="Substantial agreement" if kappa > 0.6 else "Moderate agreement")
            st.metric("Overall Hit Rate",
                      f"{results[best_key]['metrics']['HitRate']:.1f}%",
                      delta="drought/non-drought accuracy")
            st.markdown("<hr>", unsafe_allow_html=True)
            report_dict = classification_report(
                y_true_cls, y_pred_cls, output_dict=True, zero_division=0)
            for cls in ["Moderate drought","Severe drought","Extreme drought"]:
                if cls in report_dict:
                    r = report_dict[cls]
                    color = {"Moderate drought":"var(--yellow)",
                             "Severe drought":"var(--orange)",
                             "Extreme drought":"var(--red)"}[cls]
                    st.markdown(f"""
                    <div class='insight-card' style='padding:.7rem 1rem;margin-bottom:.5rem'>
                      <div class='ic-label'>{cls}</div>
                      <div style='font-family:var(--m);font-size:.75rem;
                                  display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;
                                  color:var(--text2)'>
                        <span>P: <span style='color:{color}'>{r["precision"]:.2f}</span></span>
                        <span>R: <span style='color:{color}'>{r["recall"]:.2f}</span></span>
                        <span>F1: <span style='color:{color}'>{r["f1-score"]:.2f}</span></span>
                      </div>
                    </div>""", unsafe_allow_html=True)


#  PAGE: FORECAST & UNCERTAINTY
elif page == "🎯  Forecast & Uncertainty":

    st.markdown("<h2 style='font-family:var(--r);font-weight:400;color:var(--text);margin-bottom:.3rem'>Forecast & Uncertainty</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text2);font-size:.88rem;margin-bottom:1.5rem'>Bootstrap 90% prediction intervals on the test set. Ridge Regression · Horizon t+1.</p>", unsafe_allow_html=True)

    with st.spinner("Running bootstrap uncertainty quantification…"):
        results, d = train_models(raw_bytes)

    if "bootstrap" not in results:
        st.warning("Bootstrap results not available.")
    else:
        boot      = results["bootstrap"]
        y_true    = d["y_te"]["y_t1"].values
        idx_test  = d["idx_te"]

        coverage_90 = float(np.mean((y_true >= boot["p05"]) & (y_true <= boot["p95"])) * 100)
        coverage_50 = float(np.mean((y_true >= boot["p25"]) & (y_true <= boot["p75"])) * 100)
        mean_width  = float(np.mean(boot["p95"] - boot["p05"]))
        outside_idx = np.where((y_true < boot["p05"]) | (y_true > boot["p95"]))[0]

        # KPIs
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("90% PI Coverage",  f"{coverage_90:.1f}%",
                  delta="target: 90%" )
        c2.metric("50% PI Coverage",  f"{coverage_50:.1f}%",
                  delta="target: 50%")
        c3.metric("Mean PI Width",    f"{mean_width:.3f}")
        c4.metric("Outside-PI points", str(len(outside_idx)),
                  delta=f"{100*len(outside_idx)/len(y_true):.1f}% of test")

        st.markdown("<p class='section-label' style='margin-top:1.5rem'>Bootstrap prediction intervals — test set</p>",
                    unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_hrect(y0=-4,  y1=-2.0, fillcolor="#e05252", opacity=0.05, line_width=0)
        fig.add_hrect(y0=-2.0,y1=-1.5, fillcolor="#e07a30", opacity=0.05, line_width=0)

        fig.add_trace(go.Scatter(
            x=np.concatenate([idx_test, idx_test[::-1]]),
            y=np.concatenate([boot["p95"], boot["p05"][::-1]]),
            fill="toself", fillcolor="rgba(91,170,212,0.12)",
            line=dict(width=0), showlegend=True, name="90% PI",
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([idx_test, idx_test[::-1]]),
            y=np.concatenate([boot["p75"], boot["p25"][::-1]]),
            fill="toself", fillcolor="rgba(91,170,212,0.20)",
            line=dict(width=0), showlegend=True, name="50% PI",
        ))
        fig.add_trace(go.Scatter(
            x=idx_test, y=boot["mean"], mode="lines",
            name="Bootstrap mean",
            line=dict(color="#5baad4", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=idx_test, y=y_true, mode="lines+markers",
            name="Actual SPEI-12",
            line=dict(color="#e6d9c7", width=1.5),
            marker=dict(size=4, color="#e6d9c7"),
        ))
        # Highlight outside-PI points
        if len(outside_idx):
            fig.add_trace(go.Scatter(
                x=idx_test[outside_idx], y=y_true[outside_idx],
                mode="markers", name="Outside 90% PI",
                marker=dict(color="#e05252", size=9, symbol="x",
                            line=dict(width=2, color="#e05252")),
            ))

        fig.add_hline(y=-1, line_dash="dot",
                      line_color="rgba(212,176,48,0.5)", line_width=1)
        fig.add_hline(y=-2, line_dash="dot",
                      line_color="rgba(224,82,82,0.5)", line_width=1)
        fig.add_hline(y=0,  line_dash="dot",
                      line_color="rgba(230,217,199,0.2)", line_width=1)
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          yaxis=dict(**AXIS_STYLE,
                                     title="SPEI-12 (z-score)", range=[-3.5, 4.0]))
        st.plotly_chart(fig, use_container_width=True)

        # ── PI width over time ──
        st.markdown("<p class='section-label'>Prediction interval width over time</p>",
                    unsafe_allow_html=True)
        pi_width = boot["p95"] - boot["p05"]
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(
            x=idx_test, y=pi_width, mode="lines",
            name="PI width",
            line=dict(color="#d4b030", width=1.5),
            fill="tozeroy", fillcolor="rgba(212,176,48,0.08)",
        ))
        fig_w.update_layout(**PLOTLY_LAYOUT, height=220,
                            yaxis=dict(**AXIS_STYLE, title="PI width"))
        st.plotly_chart(fig_w, use_container_width=True)

        # ── Reliability diagram ──
        st.markdown("<p class='section-label'>Coverage reliability by month</p>",
                    unsafe_allow_html=True)
        cov_by_month = {}
        for mo in range(1, 13):
            mask_mo = idx_test.month == mo
            if mask_mo.sum() == 0: continue
            cov_by_month[mo] = float(np.mean(
                (y_true[mask_mo] >= boot["p05"][mask_mo]) &
                (y_true[mask_mo] <= boot["p95"][mask_mo])
            ) * 100)

        fig_cov = go.Figure()
        fig_cov.add_hline(y=90, line_dash="dash",
                          line_color="rgba(91,170,212,0.6)",
                          annotation_text="90% target",
                          annotation_position="top right")
        fig_cov.add_trace(go.Bar(
            x=[MONTH_NAMES[m-1] for m in cov_by_month],
            y=list(cov_by_month.values()),
            marker_color=["#5fc490" if v >= 85 else "#e07a30"
                          for v in cov_by_month.values()],
            marker_line_width=0,
            text=[f"{v:.0f}%" for v in cov_by_month.values()],
            textposition="outside",
            textfont=dict(size=10, family="DM Mono"),
        ))
        fig_cov.update_layout(**PLOTLY_LAYOUT, height=260,
                              yaxis=dict(**AXIS_STYLE,
                                         title="Coverage %", range=[0, 115]),
                              xaxis=dict(**AXIS_STYLE, title=""))
        st.plotly_chart(fig_cov, use_container_width=True)



#  PAGE: FEATURE INTELLIGENCE

elif page == "🔬  Feature Intelligence":

    st.markdown("<h2 style='font-family:var(--r);font-weight:400;color:var(--text);margin-bottom:.3rem'>Feature Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text2);font-size:.88rem;margin-bottom:1.5rem'>Feature importance from Ridge coefficients and XGBoost gain-based scores.</p>", unsafe_allow_html=True)

    with st.spinner("Computing feature importance…"):
        results, d = train_models(raw_bytes)

    tab1, tab2, tab3 = st.tabs(["  Ridge Coefficients  ", "  XGBoost Importance  ", "  Feature Correlation  "])

    with tab1:
        horizon = st.selectbox("Horizon", ["t+1","t+3","t+6"], key="fi_ridge_h")
        key = (horizon, "Ridge")
        if key in results and results[key].get("fi"):
            fi = pd.Series(results[key]["fi"]).sort_values(ascending=False)
            top_n = st.slider("Top N features", 10, min(40, len(fi)), 20, key="fi_ridge_n")
            top = fi.head(top_n)

            fig = go.Figure(go.Bar(
                x=top.values[::-1], y=top.index[::-1],
                orientation="h",
                marker_color=["#5baad4" if v > 0 else "#e07a30" for v in top.values[::-1]],
                marker_line_width=0,
                text=[f"{v:.4f}" for v in top.values[::-1]],
                textposition="outside",
                textfont=dict(size=9, family="DM Mono"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=max(350, top_n * 22),
                              xaxis=dict(**AXIS_STYLE,
                                         title="|Coefficient| (Ridge)"),
                              yaxis=dict(**AXIS_STYLE,
                                         tickfont=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)

            # Feature category breakdown
            st.markdown("<p class='section-label'>Importance by feature family</p>",
                        unsafe_allow_html=True)
            def categorise(name):
                if "lag" in name:     return "Lag features"
                if "roll" in name:    return "Rolling statistics"
                if "month" in name or "year" in name or "decade" in name: return "Temporal"
                if "minus" in name or "times" in name: return "Interactions"
                return "Other"

            fi_df = pd.DataFrame({"feature": fi.index, "importance": fi.abs().values})
            fi_df["category"] = fi_df["feature"].apply(categorise)
            cat_sum = fi_df.groupby("category")["importance"].sum().sort_values(ascending=False)

            fig_cat = go.Figure(go.Bar(
                x=cat_sum.index, y=cat_sum.values,
                marker_color=["#5baad4","#e07a30","#5fc490","#d4b030","#e05252"][:len(cat_sum)],
                marker_line_width=0,
                text=[f"{v:.3f}" for v in cat_sum.values],
                textposition="outside",
                textfont=dict(size=11, family="DM Mono"),
            ))
            fig_cat.update_layout(**PLOTLY_LAYOUT, height=260,
                                  yaxis=dict(**AXIS_STYLE,
                                             title="Total |coefficient|"))
            st.plotly_chart(fig_cat, use_container_width=True)

    with tab2:
        horizon_xgb = st.selectbox("Horizon", ["t+1","t+3","t+6"], key="fi_xgb_h")
        key_xgb = (horizon_xgb, "XGBoost")
        if key_xgb in results and results[key_xgb].get("fi"):
            fi_xgb = pd.Series(results[key_xgb]["fi"]).sort_values(ascending=False)
            top_n_x = st.slider("Top N", 10, min(40, len(fi_xgb)), 20, key="fi_xgb_n")
            top_x = fi_xgb.head(top_n_x)

            fig_x = go.Figure(go.Bar(
                x=top_x.values[::-1], y=top_x.index[::-1],
                orientation="h",
                marker_color="#e07a30",
                marker_line_width=0,
                text=[f"{v:.5f}" for v in top_x.values[::-1]],
                textposition="outside",
                textfont=dict(size=9, family="DM Mono"),
            ))
            fig_x.update_layout(**PLOTLY_LAYOUT, height=max(350, top_n_x * 22),
                                xaxis=dict(**AXIS_STYLE,
                                           title="Feature importance (gain)"),
                                yaxis=dict(**AXIS_STYLE,
                                           tickfont=dict(size=9)))
            st.plotly_chart(fig_x, use_container_width=True)

            # Ridge vs XGBoost agreement
            st.markdown("<p class='section-label'>Ridge vs XGBoost — top-feature agreement</p>",
                        unsafe_allow_html=True)
            key_ridge = (horizon_xgb, "Ridge")
            if key_ridge in results and results[key_ridge].get("fi"):
                fi_r = pd.Series(results[key_ridge]["fi"])
                common = list(set(fi_xgb.head(20).index) & set(fi_r.head(20).index))
                if common:
                    r_rank = fi_r.rank(ascending=False)[common].astype(int)
                    x_rank = fi_xgb.rank(ascending=False)[common].astype(int)
                    agree_df = pd.DataFrame({
                        "Feature":   common,
                        "Ridge Rank":   r_rank.values,
                        "XGBoost Rank": x_rank.values,
                    }).sort_values("Ridge Rank")
                    fig_agree = go.Figure()
                    for feat in agree_df["Feature"]:
                        r = agree_df.loc[agree_df["Feature"]==feat, "Ridge Rank"].values[0]
                        x = agree_df.loc[agree_df["Feature"]==feat, "XGBoost Rank"].values[0]
                        fig_agree.add_trace(go.Scatter(
                            x=["Ridge", "XGBoost"], y=[r, x],
                            mode="lines+markers+text",
                            name=feat,
                            text=[str(r), str(x)],
                            textposition="middle right",
                            line=dict(width=1.2, color="rgba(91,170,212,0.4)"),
                            marker=dict(size=7),
                            showlegend=False,
                        ))
                    fig_agree.update_layout(
                        **PLOTLY_LAYOUT, height=300,
                        yaxis=dict(**AXIS_STYLE,
                                   title="Rank (lower = more important)",
                                   autorange="reversed"),
                        xaxis=dict(**AXIS_STYLE,
                                   categoryorder="array",
                                   categoryarray=["Ridge","XGBoost"]),
                    )
                    st.plotly_chart(fig_agree, use_container_width=True)

    with tab3:
        st.markdown("<p class='section-label'>Pairwise correlation — engineered features</p>",
                    unsafe_allow_html=True)
        n_feat_corr = st.slider("Number of top features to correlate", 10, 30, 20)
        key_corr = ("t+1", "Ridge")
        if key_corr in results and results[key_corr].get("fi"):
            top_feats = list(pd.Series(results[key_corr]["fi"])
                             .abs().sort_values(ascending=False)
                             .head(n_feat_corr).index)
            feat_df   = pd.DataFrame(
                d["X_te"][:, [d["feats"].index(f) for f in top_feats]],
                columns=top_feats,
            )
            corr = feat_df.corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale=[[0,"#e05252"],[0.5,"#1a2235"],[1,"#5baad4"]],
                zmid=0, zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont=dict(size=8, family="DM Mono"),
                colorbar=dict(tickfont=dict(size=9)),
            ))
            fig_corr.update_layout(
                **PLOTLY_LAYOUT,
                height=max(400, n_feat_corr * 18 + 80),
                xaxis=dict(**AXIS_STYLE,
                           tickangle=45, tickfont=dict(size=8)),
                yaxis=dict(**AXIS_STYLE,
                           autorange="reversed", tickfont=dict(size=8)),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

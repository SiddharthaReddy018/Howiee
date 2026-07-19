"""
app.py
======
Implementation Plan §I — frontend. A single Streamlit app covering all six
required views, ordered per §I.2's demo workflow (ingest -> forecast ->
simulate -> insights):

  1. Data-health / ingestion panel (§A) — what was mapped, inferred,
     degraded, or ignored, for the CURRENT run's data directory. Shown
     first so the audience sees what the model is standing on before any
     forecast number appears.
  2. Probabilistic fan chart (p10/p50/p90) per campaign / campaign_type /
     channel / total, at 30/60/90-day horizons.
  3. Budget "what-if" slider — live re-scored forecast as spend is dialed
     up/down, monotonic by construction (§C.3's isotonic post-processing)
     and business-plausibility-clamped (§G.3) at the exact scenario point
     displayed, not just the background curve.
  4. Reconciled breakdown across the hierarchy (§E) — campaign -> type ->
     channel -> total, with the coherence check visible.
  5. Grounded AI causal summary (§H) with its validation status shown, not
     hidden.
  6. Model reliability panel (§D/§G) — walk-forward vs. grouped CV, the
     final-holdout numbers, and the calibration reliability diagram.

Run with:  streamlit run src/app.py
(defaults assume the repo's own ./data, ./pickle/model.pkl, ./output/ paths;
override via the sidebar's "Advanced paths" expander if needed.)
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from schema_mapper import ingest_directory
from feature_engineering import build_latest_snapshot
import modeling as M
from budget_curves import saturation_status, hill_predict, optimize_budget_allocation
from sanity_clamps import check_forecast_plausibility
import llm_insights as L

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="AIgnition — Ad Revenue Forecasting", layout="wide", page_icon="◆")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ─── Base typography ─── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; letter-spacing: -0.02em; }
.stMetric label, .stMetric [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

/* ─── Apple-style keyframe animations ─── */
@keyframes pullUp {
    from { opacity: 0; transform: translateY(40px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.92); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes glow {
    0%, 100% { box-shadow: 0 0 8px rgba(0,210,255,0.15); }
    50%      { box-shadow: 0 0 20px rgba(0,210,255,0.35); }
}
@keyframes countUp {
    from { opacity: 0; transform: translateY(15px); filter: blur(4px); }
    to   { opacity: 1; transform: translateY(0); filter: blur(0); }
}

/* ─── Eyebrow / branding ─── */
.eyebrow {
    font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.14em;
    font-size: 0.72rem; color: #00D2FF; margin-bottom: -0.4rem;
    animation: fadeIn 0.8s ease-out;
}

/* ─── Stacked column animation classes ─── */
[data-testid="column"]:nth-child(1) > div { animation: pullUp 0.6s ease-out both; animation-delay: 0.05s; }
[data-testid="column"]:nth-child(2) > div { animation: pullUp 0.6s ease-out both; animation-delay: 0.15s; }
[data-testid="column"]:nth-child(3) > div { animation: pullUp 0.6s ease-out both; animation-delay: 0.25s; }
[data-testid="column"]:nth-child(4) > div { animation: pullUp 0.6s ease-out both; animation-delay: 0.35s; }

/* ─── Global containers get a soft pull-up ─── */
[data-testid="stVerticalBlock"] > div { animation: pullUp 0.5s ease-out both; }

/* ─── Streamlit tab bar polish ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: rgba(15,23,42,0.6);
    border-radius: 12px; padding: 4px;
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(0,210,255,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 8px 16px; font-weight: 500; font-size: 0.85rem;
    transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(0,210,255,0.12) !important;
    transform: translateY(-1px);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,210,255,0.25), rgba(0,210,255,0.12)) !important;
    border: 1px solid rgba(0,210,255,0.4) !important;
    box-shadow: 0 2px 12px rgba(0,210,255,0.2);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ─── Sidebar polish ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0F172A 50%, #1E293B 100%) !important;
    border-right: 1px solid rgba(0,210,255,0.15);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    animation: slideInLeft 0.5s ease-out both;
}

/* ─── Caveat / OK boxes with animation ─── */
.caveat-box {
    border-left: 3px solid #FF1744; padding: 0.7rem 1.1rem;
    background: rgba(255,23,68,0.08); border-radius: 0 8px 8px 0;
    font-size: 0.9rem; animation: slideInLeft 0.4s ease-out both;
    transition: background 0.3s ease;
}
.caveat-box:hover { background: rgba(255,23,68,0.14); }
.ok-box {
    border-left: 3px solid #00E676; padding: 0.7rem 1.1rem;
    background: rgba(0,230,118,0.08); border-radius: 0 8px 8px 0;
    font-size: 0.9rem; animation: slideInLeft 0.4s ease-out both;
    transition: background 0.3s ease;
}
.ok-box:hover { background: rgba(0,230,118,0.14); }

/* ─── Data tables get a subtle entrance ─── */
[data-testid="stDataFrame"] { animation: scaleIn 0.5s ease-out both; animation-delay: 0.2s; }

/* ─── Plotly charts get a subtle entrance ─── */
.js-plotly-plot { animation: fadeIn 0.6s ease-out both; animation-delay: 0.15s; }

/* ─── Metric cards ─── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(15,23,42,0.8), rgba(30,41,59,0.6));
    border: 1px solid rgba(0,210,255,0.2); border-radius: 12px;
    padding: 14px 18px; backdrop-filter: blur(8px);
    transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    animation: pullUp 0.5s ease-out both;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0,210,255,0.5);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3), 0 0 12px rgba(0,210,255,0.15);
}

/* ─── Containers (border=True) polish ─── */
[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border-color: rgba(0,210,255,0.15) !important;
    transition: all 0.3s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(0,210,255,0.35) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}

/* ─── Buttons ─── */
.stButton > button {
    border-radius: 8px; font-weight: 500;
    transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    border: 1px solid rgba(0,210,255,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,210,255,0.25);
}

/* ─── Slider ─── */
[data-testid="stSlider"] { animation: fadeIn 0.5s ease-out both; animation-delay: 0.1s; }

/* ─── Horizontal rules ─── */
hr { border-color: rgba(0,210,255,0.15); margin: 1.5rem 0; }

/* ─── Selectbox / inputs ─── */
.stSelectbox, .stTextInput {
    transition: all 0.3s ease;
}

/* ─── Page title entrance ─── */
h1 { animation: pullUp 0.7s ease-out both !important; }
h2 { animation: pullUp 0.6s ease-out both !important; animation-delay: 0.1s !important; }
h3 { animation: pullUp 0.5s ease-out both !important; animation-delay: 0.15s !important; }

/* ─── Executive Dashboard Cards — Glassmorphism + Pull-up ─── */
.exec-card {
    background: linear-gradient(145deg, rgba(15,23,42,0.85) 0%, rgba(30,41,59,0.75) 100%);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(0,210,255,0.2); border-radius: 16px;
    padding: 1.3rem 1.6rem; margin-bottom: 0.8rem;
    transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    animation: pullUp 0.6s ease-out both;
    position: relative; overflow: hidden;
}
.exec-card::before {
    content: ''; position: absolute; top: 0; left: -100%; width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,210,255,0.06), transparent);
    transition: left 0.6s ease;
}
.exec-card:hover::before { left: 120%; }
.exec-card:hover {
    transform: translateY(-6px);
    border-color: rgba(0,210,255,0.5);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 20px rgba(0,210,255,0.12);
}
.exec-card h4 {
    font-family: 'Inter', sans-serif; margin: 0 0 0.4rem 0;
    color: #00D2FF; font-size: 1rem; font-weight: 600;
    letter-spacing: -0.01em;
}
.exec-big {
    font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 700;
    color: #F8FAFC; line-height: 1.2;
    animation: countUp 0.7s ease-out both;
    animation-delay: 0.3s;
}
.exec-sub {
    font-family: 'Inter', sans-serif; font-size: 0.82rem;
    color: rgba(148,163,184,0.9); margin-top: 0.3rem; line-height: 1.5;
}

/* ─── Recommendation cards — slide-in with hover lift ─── */
.rec-card {
    background: linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,230,118,0.06));
    border-left: 4px solid #00E676; border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem; margin-bottom: 0.7rem;
    animation: slideInLeft 0.5s ease-out both;
    transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    backdrop-filter: blur(8px);
}
.rec-card:hover {
    transform: translateX(6px);
    background: linear-gradient(135deg, rgba(0,230,118,0.2), rgba(0,230,118,0.1));
    box-shadow: 0 4px 16px rgba(0,230,118,0.15);
}
.rec-card-warn {
    background: linear-gradient(135deg, rgba(0,210,255,0.12), rgba(0,210,255,0.06));
    border-left: 4px solid #00D2FF; border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem; margin-bottom: 0.7rem;
    animation: slideInLeft 0.5s ease-out both;
    transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    backdrop-filter: blur(8px);
}
.rec-card-warn:hover {
    transform: translateX(6px);
    background: linear-gradient(135deg, rgba(0,210,255,0.2), rgba(0,210,255,0.1));
    box-shadow: 0 4px 16px rgba(0,210,255,0.15);
}
.rec-card-risk {
    background: linear-gradient(135deg, rgba(255,23,68,0.12), rgba(255,23,68,0.06));
    border-left: 4px solid #FF1744; border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem; margin-bottom: 0.7rem;
    animation: slideInLeft 0.5s ease-out both;
    transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    backdrop-filter: blur(8px);
}
.rec-card-risk:hover {
    transform: translateX(6px);
    background: linear-gradient(135deg, rgba(255,23,68,0.2), rgba(255,23,68,0.1));
    box-shadow: 0 4px 16px rgba(255,23,68,0.15);
}

/* ─── Channel pills — glow on hover ─── */
.channel-pill {
    display: inline-block; padding: 0.25rem 0.8rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; margin-right: 0.4rem;
    letter-spacing: 0.03em; transition: all 0.3s ease;
}
.channel-pill:hover { transform: scale(1.08); }
.pill-google {
    background: rgba(66,133,244,0.15); color: #4285F4;
    border: 1px solid rgba(66,133,244,0.35);
    box-shadow: 0 0 8px rgba(66,133,244,0.1);
}
.pill-google:hover { box-shadow: 0 0 16px rgba(66,133,244,0.3); }
.pill-meta {
    background: rgba(24,119,242,0.15); color: #1877F2;
    border: 1px solid rgba(24,119,242,0.35);
    box-shadow: 0 0 8px rgba(24,119,242,0.1);
}
.pill-meta:hover { box-shadow: 0 0 16px rgba(24,119,242,0.3); }
.pill-bing {
    background: rgba(0,167,157,0.15); color: #00A79D;
    border: 1px solid rgba(0,167,157,0.35);
    box-shadow: 0 0 8px rgba(0,167,157,0.1);
}
.pill-bing:hover { box-shadow: 0 0 16px rgba(0,167,157,0.3); }

/* ─── Section headers ─── */
.section-header {
    animation: pullUp 0.6s ease-out both;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid rgba(0,210,255,0.15);
    margin-bottom: 1rem;
}

/* ─── Trend colors ─── */
.trend-up { color: #00E676; } .trend-down { color: #FF1744; } .trend-flat { color: #94A3B8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained model bundle...")
def load_bundle(model_path: str):
    return joblib.load(model_path)


@st.cache_data(show_spinner="Ingesting data directory...")
def load_canonical(data_dir: str):
    df, reports = ingest_directory(data_dir)
    return df, {k: v.to_dict() for k, v in reports.items()}


@st.cache_data(show_spinner=False)
def load_output_csv(path: str):
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_output_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Scoring live scenario...")
def score_scenario(_bundle, snapshot_df: pd.DataFrame, horizon: int, budget_multiplier: float) -> pd.DataFrame:
    snap = snapshot_df.copy()
    snap["horizon_days"] = horizon
    baseline = snap["spend_roll_mean_28"].fillna(snap["spend_lag_7"]).fillna(0.0)
    snap["planned_future_daily_budget"] = baseline * budget_multiplier
    X = snap[_bundle["feature_names"]]
    q_preds = M.predict_quantiles(_bundle["quantile_models"], X, quantiles=_bundle["quantiles"])
    out = snap[["campaign_id", "campaign_name", "channel", "campaign_type"]].copy()
    for i, q in enumerate(_bundle["quantiles"]):
        out[f"q{q}"] = q_preds[:, i]
    # needed downstream to apply the same §G.3 plausibility clamp the static
    # predict.py output gets -- the live what-if slider skipped this before
    out["assumed_spend_total"] = snap["planned_future_daily_budget"].to_numpy() * horizon
    return out


def _clamp_median_column(scored: pd.DataFrame, roas_bounds: dict) -> tuple[pd.Series, int]:
    """§G.3 applied to a single live what-if scenario frame: caps each
    campaign's displayed q0.5 to its group's historical ROAS envelope (same
    rule predict.py applies to the static output), leaving the raw q0.5 (and
    every other quantile column) untouched. Returns (display_series,
    n_flagged)."""
    display_vals, flags = [], []
    for _, row in scored.iterrows():
        res = check_forecast_plausibility(
            row["q0.5"], row["assumed_spend_total"], row["channel"], row["campaign_type"], roas_bounds,
        )
        display_vals.append(res["display_value"])
        flags.append(res["violation"])
    return pd.Series(display_vals, index=scored.index), int(sum(flags))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◆ AIgnition")
    st.caption("Probabilistic multi-channel ad revenue forecasting — TechBlazers")
    with st.expander("Paths", expanded=False):
        data_dir = st.text_input("Data directory", value=os.path.join(REPO_ROOT, "data"))
        model_path = st.text_input("Model bundle", value=os.path.join(REPO_ROOT, "pickle", "model.pkl"))
        output_dir = st.text_input("Output directory", value=os.path.join(REPO_ROOT, "output"))

    if not os.path.exists(model_path):
        st.error(f"No trained model at {model_path}. Run `python3 src/train.py` first.")
        st.stop()

    bundle = load_bundle(model_path)
    canonical_df, ingestion_reports = load_canonical(data_dir)
    snapshot = build_latest_snapshot(canonical_df)

    st.markdown("---")
    channels = ["All"] + sorted(canonical_df["channel"].unique().tolist())
    sel_channel = st.selectbox("Channel", channels)
    scoped = canonical_df if sel_channel == "All" else canonical_df[canonical_df["channel"] == sel_channel]
    types = ["All"] + sorted(scoped["campaign_type"].unique().tolist())
    sel_type = st.selectbox("Campaign type", types)
    horizon = st.select_slider("Forecast horizon (days)", options=bundle["horizons"], value=bundle["horizons"][0])

    st.markdown("---")
    st.caption(f"Model trained {bundle['trained_at'][:10]} · {bundle['training_frame_shape'][0]:,} training rows")

predictions_df = load_output_csv(os.path.join(output_dir, "predictions.csv"))
reconciled_df = load_output_csv(os.path.join(output_dir, "reconciled_hierarchy.csv"))
causal_summaries = load_output_json(os.path.join(output_dir, "causal_summary.json"))
hill_curves_list = load_output_json(os.path.join(output_dir, "hill_curves.json"))
mpc_backtest = load_output_json(os.path.join(output_dir, "mpc_reallocation_backtest.json"))
hindsight_regret = load_output_json(os.path.join(output_dir, "hindsight_regret_audit.json"))


def _hierarchy_node_id(channel: str, campaign_type: str) -> str:
    if channel == "All":
        return "total"
    if campaign_type == "All":
        return f"total/{channel}"
    return f"total/{channel}/{campaign_type}"


def _filter_scope(df, channel_col="channel", type_col="campaign_type"):
    out = df
    if sel_channel != "All":
        out = out[out[channel_col] == sel_channel]
    if sel_type != "All" and type_col in out.columns:
        out = out[out[type_col] == sel_type]
    return out


def _build_hierarchy_frame(hsub: pd.DataFrame) -> pd.DataFrame:
    ids = hsub["unique_id"].tolist()
    parents, labels = [], []
    for uid in ids:
        if uid == "total":
            parents.append("")
            labels.append("Total")
        else:
            parent, _, leaf = uid.rpartition("/")
            parents.append(parent)
            labels.append(leaf.split("::")[-1])
    values = hsub["reconciled_median"].clip(lower=0).to_numpy()
    return pd.DataFrame({"id": ids, "parent": parents, "label": labels, "value": values})


def _build_fan_chart(hist_scope_df: pd.DataFrame, sub: pd.DataFrame, horizon: int,
                      lookback_days: int) -> tuple[go.Figure, dict]:
    q_cols = sorted([c for c in sub.columns if c.startswith("revenue_p") and c[-1].isdigit()])
    totals = sub[q_cols].sum()
    percentiles = [int(c.replace("revenue_p", "")) for c in q_cols]
    rate = {p: totals[f"revenue_p{p:02d}"] / horizon for p in percentiles}

    forecast_as_of = pd.to_datetime(sub["forecast_as_of"].iloc[0])
    fc_start, fc_end = forecast_as_of, forecast_as_of + pd.Timedelta(days=horizon)

    hist = hist_scope_df.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
    hist = hist[(hist["date"] >= forecast_as_of - pd.Timedelta(days=lookback_days)) & (hist["date"] <= forecast_as_of)]
    hist["revenue_7d_avg"] = hist["revenue"].rolling(7, min_periods=1).mean()

    fig = go.Figure()

    if len(hist):
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["revenue"], mode="lines", line=dict(color="#64748B", width=1),
            opacity=0.45, name="Daily actual revenue",
            hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["revenue_7d_avg"], mode="lines", line=dict(color="#F8FAFC", width=2),
            name="7-day avg (actual)", hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra></extra>",
        ))

    band_defs = [(5, 95, "rgba(0,210,255,0.12)"), (10, 90, "rgba(0,210,255,0.22)"),
                 (25, 75, "rgba(0,210,255,0.35)")]
    for lo, hi, color in band_defs:
        if lo in rate and hi in rate:
            fig.add_trace(go.Scatter(x=[fc_start, fc_end], y=[rate[hi], rate[hi]], mode="lines",
                                      line=dict(width=0), showlegend=False,
                                      hovertemplate=f"P{hi} implied daily rate: " + "%{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[fc_start, fc_end], y=[rate[lo], rate[lo]], mode="lines",
                                      line=dict(width=0), fill="tonexty", fillcolor=color,
                                      name=f"P{lo}-P{hi} (forecast)",
                                      hovertemplate=f"P{lo} implied daily rate: " + "%{y:,.0f}<extra></extra>"))

    if 50 in rate:
        if len(hist):
            fig.add_trace(go.Scatter(
                x=[hist["date"].iloc[-1], fc_start], y=[hist["revenue_7d_avg"].iloc[-1], rate[50]],
                mode="lines", line=dict(color="#00D2FF", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[fc_start, fc_end], y=[rate[50], rate[50]], mode="lines",
            line=dict(color="#00D2FF", width=3), name="Median (forecast, implied daily rate)",
            hovertemplate="implied daily rate: %{y:,.0f}<extra></extra>",
        ))

    fig.add_vline(x=forecast_as_of, line_dash="dash", line_color="#F8FAFC",
                  annotation_text="Today", annotation_position="top")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date", yaxis_title="Daily revenue", height=380,
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22),
    )
    return fig, {"forecast_as_of": forecast_as_of, "fc_end": fc_end, "rate": rate}


def _build_seasonal_decomposition_chart(hist_full_scope_df: pd.DataFrame, rate: dict,
                                         forecast_as_of: pd.Timestamp, horizon: int) -> go.Figure | None:
    h = hist_full_scope_df.copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h.groupby("date", as_index=False)["revenue"].sum()
    if len(h) < 21:
        return None

    h["dow"] = h["date"].dt.dayofweek
    dow_mean = h.groupby("dow")["revenue"].mean()
    overall_mean = h["revenue"].mean()
    if overall_mean <= 0 or dow_mean.isna().any() or len(dow_mean) < 7:
        return None
    dow_share = (dow_mean / overall_mean).reindex(range(7)).fillna(1.0)

    fc_dates = pd.date_range(forecast_as_of + pd.Timedelta(days=1), periods=horizon, freq="D")
    raw_weight = dow_share.reindex(fc_dates.dayofweek).to_numpy()
    norm_weight = raw_weight / raw_weight.mean()

    percentiles = sorted(rate.keys())
    day_q = {p: rate[p] * norm_weight for p in percentiles}

    fig = go.Figure()
    band_defs = [(5, 95, "rgba(0,210,255,0.12)"), (10, 90, "rgba(0,210,255,0.22)"),
                 (25, 75, "rgba(0,210,255,0.35)")]
    for lo, hi, color in band_defs:
        if lo in day_q and hi in day_q:
            fig.add_trace(go.Scatter(x=fc_dates, y=day_q[hi], mode="lines", line=dict(width=0),
                                      showlegend=False,
                                      hovertemplate=f"P{hi}: " + "%{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=fc_dates, y=day_q[lo], mode="lines", line=dict(width=0),
                                      fill="tonexty", fillcolor=color, name=f"P{lo}-P{hi} (decomposed)",
                                      hovertemplate=f"P{lo}: " + "%{y:,.0f}<extra></extra>"))
    if 50 in day_q:
        fig.add_trace(go.Scatter(
            x=fc_dates, y=day_q[50], mode="lines+markers", line=dict(color="#00D2FF", width=2),
            marker=dict(size=4), name="Median (day-of-week decomposition)",
            hovertemplate="%{x|%a, %Y-%m-%d}<br>Median decomposed estimate: %{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date", yaxis_title="Decomposed daily revenue", height=340,
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.28),
    )
    return fig


st.markdown("<div class='eyebrow'>NetElixir AIgnition 3.0 · Team TechBlazers</div>", unsafe_allow_html=True)
st.title("Multi-Channel Ad Revenue Forecasting")

tabs = st.tabs([
    "📊 Executive Dashboard", "🏥 Data Health", "📈 Forecast", "💰 Budget What-If", "🔀 Breakdown", "🤖 AI Summary", "🔬 Model Reliability",
])

# ─────────────────────────────────────────────────────────────────────────────
# 📊 Executive Dashboard — business-friendly overview for non-technical users
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Executive Overview")
    as_of = pd.to_datetime(canonical_df["date"]).max().strftime("%b %d, %Y")
    st.caption(f"Performance summary for the next {horizon} days · Data as of {as_of} · "
               f"Use the sidebar to filter by channel or campaign type")

    if predictions_df is not None:
        currency = os.environ.get("CURRENCY_SYMBOL", "$")
        pred_h = _filter_scope(predictions_df)
        pred_h = pred_h[pred_h["horizon_days"] == horizon]

        if len(pred_h) > 0:
            total_revenue_p50 = pred_h["revenue_p50"].sum()
            total_revenue_p10 = pred_h["revenue_p10"].sum() if "revenue_p10" in pred_h else 0
            total_revenue_p90 = pred_h["revenue_p90"].sum() if "revenue_p90" in pred_h else 0
            total_spend = pred_h["assumed_spend_total"].sum()
            blended_roas = total_revenue_p50 / total_spend if total_spend > 0 else 0
            n_total_campaigns = len(pred_h)

            # ── One-line headline — the single takeaway for a reader with 5 seconds ──
            st.markdown(f"""<div class='rec-card' style='font-size:1.05rem'>
                Over the next <strong>{horizon} days</strong>, this portfolio is projected to bring in
                <strong>{currency}{total_revenue_p50:,.0f}</strong> (likely between {currency}{total_revenue_p10:,.0f}
                and {currency}{total_revenue_p90:,.0f}) from <strong>{currency}{total_spend:,.0f}</strong> in ad spend
                across <strong>{n_total_campaigns} campaigns</strong> — a blended <strong>{blended_roas:.2f}x</strong> return.
            </div>""", unsafe_allow_html=True)
            st.markdown("")

            # ── Top-line KPI cards ──
            st.markdown("#### Revenue Forecast")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""<div class='exec-card'>
                    <h4>Projected Revenue</h4>
                    <div class='exec-big'>{currency}{total_revenue_p50:,.0f}</div>
                    <div class='exec-sub'>Median estimate · {horizon}-day horizon</div>
                </div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""<div class='exec-card'>
                    <h4>Confidence Interval</h4>
                    <div class='exec-big'>{currency}{total_revenue_p10:,.0f} – {currency}{total_revenue_p90:,.0f}</div>
                    <div class='exec-sub'>80% probability range (P10–P90)</div>
                </div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(f"""<div class='exec-card'>
                    <h4>Return on Ad Spend</h4>
                    <div class='exec-big'>{blended_roas:.2f}x</div>
                    <div class='exec-sub'>Blended ROAS across all channels</div>
                </div>""", unsafe_allow_html=True)
            with k4:
                daily_spend = total_spend / horizon if horizon > 0 else 0
                st.markdown(f"""<div class='exec-card'>
                    <h4>Total Ad Investment</h4>
                    <div class='exec-big'>{currency}{total_spend:,.0f}</div>
                    <div class='exec-sub'>{currency}{daily_spend:,.0f}/day · {n_total_campaigns} active campaigns</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Channel Performance Breakdown ──
            st.markdown("#### Channel Performance")
            st.caption("Revenue contribution, efficiency, and budget capacity by advertising channel")

            channels_in_scope = sorted(pred_h["channel"].unique())
            chan_cols = st.columns(len(channels_in_scope)) if len(channels_in_scope) <= 4 else st.columns(3)

            for idx, ch in enumerate(channels_in_scope):
                ch_data = pred_h[pred_h["channel"] == ch]
                ch_rev = ch_data["revenue_p50"].sum()
                ch_spend = ch_data["assumed_spend_total"].sum()
                ch_roas = ch_rev / ch_spend if ch_spend > 0 else 0
                ch_share = (ch_rev / total_revenue_p50 * 100) if total_revenue_p50 > 0 else 0
                n_campaigns = len(ch_data)

                sat_counts = ch_data["saturation_status"].value_counts()
                dominant_sat = sat_counts.index[0] if len(sat_counts) > 0 else "unknown"

                if "near_saturation" in dominant_sat:
                    sat_indicator = "<span style='color:#FF1744;font-weight:600'>● Near Saturation</span>"
                elif "approaching" in dominant_sat:
                    sat_indicator = "<span style='color:#00D2FF;font-weight:600'>● Approaching Limit</span>"
                elif "room_to_grow" in dominant_sat:
                    sat_indicator = "<span style='color:#00E676;font-weight:600'>● Growth Capacity Available</span>"
                else:
                    sat_indicator = "<span style='color:#94A3B8;font-weight:600'>● Insufficient Data</span>"

                pill_class = f"pill-{ch}" if ch in ("google", "meta", "bing") else "pill-google"
                col_idx = idx % len(chan_cols)
                with chan_cols[col_idx]:
                    st.markdown(f"""<div class='exec-card'>
                        <div><span class='channel-pill {pill_class}'>{ch.upper()}</span></div>
                        <h4 style='margin-top:0.5rem'>{currency}{ch_rev:,.0f}</h4>
                        <div class='exec-sub'>
                            <strong>{ch_share:.1f}%</strong> of portfolio · {n_campaigns} campaigns<br/>
                            ROAS: <strong>{ch_roas:.2f}x</strong> · Investment: {currency}{ch_spend:,.0f}<br/>
                            {sat_indicator}
                        </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Period-over-Period Trends ──
            has_trends = False
            if causal_summaries:
                total_summary = next((s for s in causal_summaries if s.get("scope_label") == "total"), None)
                if total_summary:
                    pop = total_summary.get("grounding_context", {}).get("period_over_period", {})
                    rev_delta = pop.get("revenue_delta_pct")
                    spend_delta = pop.get("spend_delta_pct")
                    if rev_delta is not None or spend_delta is not None:
                        has_trends = True
                        st.markdown("#### Trend Analysis")
                        st.caption("Period-over-period performance compared to the prior equivalent window")
                        t1, t2 = st.columns(2)
                        if rev_delta is not None:
                            rev_direction = "increase" if rev_delta >= 0 else "decline"
                            rev_color = "#00E676" if rev_delta >= 0 else "#FF1744"
                            rev_arrow = "▲" if rev_delta >= 0 else "▼"
                            with t1:
                                st.markdown(f"""<div class='exec-card'>
                                    <h4>Revenue Trend</h4>
                                    <div class='exec-big' style='color:{rev_color}'>{rev_arrow} {abs(rev_delta):.1f}%</div>
                                    <div class='exec-sub'>{abs(rev_delta):.1f}% {rev_direction} vs. prior period</div>
                                </div>""", unsafe_allow_html=True)
                        if spend_delta is not None:
                            spend_direction = "increase" if spend_delta >= 0 else "decrease"
                            spend_color = "#94A3B8"
                            spend_arrow = "▲" if spend_delta >= 0 else "▼"
                            with t2:
                                st.markdown(f"""<div class='exec-card'>
                                    <h4>Spend Trend</h4>
                                    <div class='exec-big' style='color:{spend_color}'>{spend_arrow} {abs(spend_delta):.1f}%</div>
                                    <div class='exec-sub'>{abs(spend_delta):.1f}% {spend_direction} vs. prior period</div>
                                </div>""", unsafe_allow_html=True)

                        st.markdown("---")

            # ── Strategic Recommendations ──
            st.markdown("#### Strategic Recommendations")
            st.caption("Data-driven budget allocation guidance based on channel saturation curves and forecasted returns")

            # Overall assessment
            if has_trends and rev_delta is not None and spend_delta is not None:
                if rev_delta >= 0 and spend_delta <= 0:
                    st.markdown(f"""<div class='rec-card'>
                        <strong>Efficiency Improving</strong> — Revenue is up {abs(rev_delta):.1f}% while ad spend
                        decreased {abs(spend_delta):.1f}%. The current allocation is delivering stronger returns per dollar invested.
                    </div>""", unsafe_allow_html=True)
                elif rev_delta >= 0 and spend_delta > 0:
                    efficiency = rev_delta / spend_delta if spend_delta > 0 else 0
                    if efficiency > 1.5:
                        st.markdown(f"""<div class='rec-card'>
                            <strong>Strong Growth</strong> — Revenue growth ({abs(rev_delta):.1f}%) is significantly
                            outpacing the increase in spend ({abs(spend_delta):.1f}%), indicating efficient scaling.
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class='rec-card-warn'>
                            <strong>Growth with Caution</strong> — Revenue grew {abs(rev_delta):.1f}% but ad spend increased
                            {abs(spend_delta):.1f}%. Returns are scaling, but at a declining marginal rate. Review channel-level
                            saturation status below.
                        </div>""", unsafe_allow_html=True)
                elif rev_delta < 0:
                    st.markdown(f"""<div class='rec-card-risk'>
                        <strong>Revenue Under Pressure</strong> — Revenue declined {abs(rev_delta):.1f}% period-over-period.
                        Recommend reviewing the campaign-level breakdown and reallocating budget from low-ROAS segments
                        toward channels with available growth capacity.
                    </div>""", unsafe_allow_html=True)

            # Per-channel recommendations — 2-col grid, matches the KPI/channel
            # card layout above instead of a single stacked list.
            rec_cols = st.columns(2)
            for idx, ch in enumerate(channels_in_scope):
                ch_data = pred_h[pred_h["channel"] == ch]
                ch_spend = ch_data["assumed_spend_total"].sum()
                ch_rev = ch_data["revenue_p50"].sum()
                ch_roas = ch_rev / ch_spend if ch_spend > 0 else 0
                sat_counts = ch_data["saturation_status"].value_counts()
                dominant_sat = sat_counts.index[0] if len(sat_counts) > 0 else "unknown"

                with rec_cols[idx % 2]:
                    if "room_to_grow" in dominant_sat and ch_roas > blended_roas:
                        st.markdown(f"""<div class='rec-card'>
                            <strong>{ch.capitalize()} — Scale Opportunity</strong><br/>
                            <span class='exec-sub'>Channel ROAS of {ch_roas:.2f}x exceeds portfolio average ({blended_roas:.2f}x)
                            with available budget capacity. Incremental spend is expected to generate positive marginal returns.</span>
                        </div>""", unsafe_allow_html=True)
                    elif "near_saturation" in dominant_sat:
                        st.markdown(f"""<div class='rec-card-risk'>
                            <strong>{ch.capitalize()} — Diminishing Returns Zone</strong><br/>
                            <span class='exec-sub'>Budget saturation analysis indicates this channel is near its efficiency ceiling.
                            Current ROAS: {ch_roas:.2f}x. Further budget increases are unlikely to generate proportional returns.
                            Consider reallocating to channels with growth capacity.</span>
                        </div>""", unsafe_allow_html=True)
                    elif "approaching" in dominant_sat:
                        st.markdown(f"""<div class='rec-card-warn'>
                            <strong>{ch.capitalize()} — Approaching Saturation</strong><br/>
                            <span class='exec-sub'>Channel is performing at {ch_roas:.2f}x ROAS but nearing its marginal efficiency
                            threshold. Maintain current investment levels and monitor weekly for declining incremental returns.</span>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class='rec-card'>
                            <strong>{ch.capitalize()} — Stable</strong><br/>
                            <span class='exec-sub'>ROAS: {ch_roas:.2f}x · No saturation signal strong enough yet to recommend a
                            change; keep monitoring as more data comes in.</span>
                        </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Recommended Budget Moves (plain-language, auto-computed) ──
            # Reuses the same §F.3 Hill-curve allocator that powers the
            # "Budget What-If" tab, but with zero inputs required: the total
            # budget defaults to what this scope already spends per day, so
            # a non-technical reader gets a concrete "move $X here, $Y there,
            # expect +$Z/day" readout with no sliders to configure.
            st.markdown("#### Recommended Budget Moves")
            st.caption("Where to shift today's existing daily budget — no new spend required — based on each "
                       "channel/campaign-type's fitted response curve")

            if not hill_curves_list:
                st.info("No hill_curves.json found yet — run the pipeline once to unlock this section.")
            else:
                move_pairs = [(c["channel"], c["campaign_type"]) for c in hill_curves_list]
                if sel_channel != "All":
                    move_pairs = [p for p in move_pairs if p[0] == sel_channel]
                if sel_type != "All":
                    move_pairs = [p for p in move_pairs if p[1] == sel_type]

                if len(move_pairs) < 2:
                    st.info("Broaden the channel/campaign-type filter above to see cross-channel move "
                             "recommendations — there's only one group in the current scope.")
                else:
                    move_curves = {p: next(c for c in hill_curves_list if (c["channel"], c["campaign_type"]) == p)
                                    for p in move_pairs}
                    move_hist_spend = {}
                    for p in move_pairs:
                        g_snap = snapshot[(snapshot["channel"] == p[0]) & (snapshot["campaign_type"] == p[1])]
                        move_hist_spend[p] = (
                            float(g_snap["spend_roll_mean_28"].fillna(g_snap["spend_lag_7"]).fillna(0.0).sum())
                            if len(g_snap) else 0.0
                        )
                    move_total = sum(move_hist_spend.values())

                    if move_total <= 0:
                        st.info("No recent spend history in this scope to base a reallocation on.")
                    else:
                        move_result = optimize_budget_allocation(move_curves, move_hist_spend,
                                                                    total_daily_budget=move_total)
                        current_daily_rev = sum(
                            hill_predict(move_hist_spend[p], move_curves[p]) for p in move_pairs
                        )
                        moved_daily_rev = move_result["predicted_daily_revenue"]
                        daily_uplift = moved_daily_rev - current_daily_rev
                        period_uplift = daily_uplift * horizon

                        st.markdown(f"""<div class='rec-card'>
                            <strong>Same {currency}{move_total:,.0f}/day budget, reshuffled by channel/campaign-type
                            &rarr; an estimated {'+' if daily_uplift >= 0 else ''}{currency}{daily_uplift:,.0f}/day
                            more revenue</strong><br/>
                            <span class='exec-sub'>That's roughly {'+' if period_uplift >= 0 else ''}{currency}{period_uplift:,.0f}
                            over the next {horizon} days if the reallocation below is followed — no extra ad spend, just
                            moving today's budget to where it works harder.</span>
                        </div>""", unsafe_allow_html=True)

                        move_rows = []
                        for p in move_pairs:
                            cur = move_hist_spend[p]
                            rec = move_result["allocation"].get(p, 0.0)
                            delta = rec - cur
                            if abs(delta) < 1.0:
                                continue
                            move_rows.append((p, cur, rec, delta))
                        move_rows.sort(key=lambda r: -abs(r[3]))

                        if not move_rows:
                            st.caption("Current allocation is already close to optimal — no material moves recommended.")
                        else:
                            move_display_cols = st.columns(2)
                            for i, (p, cur, rec, delta) in enumerate(move_rows[:6]):
                                label = f"{p[0].capitalize()} — {p[1]}"
                                with move_display_cols[i % 2]:
                                    if delta > 0:
                                        st.markdown(f"""<div class='rec-card'>
                                            <strong>&uarr; Increase {label}</strong><br/>
                                            <span class='exec-sub'>{currency}{cur:,.0f}/day &rarr; {currency}{rec:,.0f}/day
                                            (+{currency}{delta:,.0f}/day) — this group's marginal ROAS
                                            ({move_result['marginal_roas'].get(p, 0.0):.2f}x) is above where the budget
                                            is coming from.</span>
                                        </div>""", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""<div class='rec-card-warn'>
                                            <strong>&darr; Decrease {label}</strong><br/>
                                            <span class='exec-sub'>{currency}{cur:,.0f}/day &rarr; {currency}{rec:,.0f}/day
                                            ({currency}{delta:,.0f}/day) — returns here are flattening out; this budget
                                            is predicted to do more elsewhere.</span>
                                        </div>""", unsafe_allow_html=True)
                            st.caption("Full allocator with adjustable total budget and a minimum-ROAS floor is in "
                                       "the \"Budget What-If\" tab.")

            st.markdown("---")

            # ── Campaign Performance Leaders ──
            st.markdown("#### Campaign Performance Report")
            st.caption("Revenue leaders and underperformers across the portfolio")

            camp_perf = pred_h[["campaign_name", "channel", "campaign_type", "revenue_p50", "roas_p50", "assumed_spend_total", "saturation_status"]].copy()
            camp_perf = camp_perf.sort_values("revenue_p50", ascending=False)

            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("**Performance Leaders**")
                top5 = camp_perf.head(5)
                for rank, (_, row) in enumerate(top5.iterrows(), 1):
                    pill_class = f"pill-{row['channel']}" if row['channel'] in ("google", "meta", "bing") else "pill-google"
                    st.markdown(f"""<div class='exec-card'>
                        <span class='channel-pill {pill_class}'>{row['channel'].upper()}</span>
                        <span style='color:#94A3B8;font-size:0.75rem;margin-left:0.3rem'>#{rank}</span><br/>
                        <strong style='font-size:0.95rem'>{row['campaign_name']}</strong><br/>
                        <span class='exec-sub'>Projected: <strong>{currency}{row['revenue_p50']:,.0f}</strong> · ROAS: {row['roas_p50']:.2f}x · {row['campaign_type']}</span>
                    </div>""", unsafe_allow_html=True)

            with tc2:
                st.markdown("**Underperforming — Review Recommended**")
                bottom5 = camp_perf[camp_perf["revenue_p50"] > 0].tail(5).sort_values("revenue_p50", ascending=True)
                if len(bottom5) == 0:
                    bottom5 = camp_perf.tail(5).sort_values("revenue_p50", ascending=True)
                for _, row in bottom5.iterrows():
                    pill_class = f"pill-{row['channel']}" if row['channel'] in ("google", "meta", "bing") else "pill-google"
                    spend_val = row['assumed_spend_total']
                    st.markdown(f"""<div class='exec-card'>
                        <span class='channel-pill {pill_class}'>{row['channel'].upper()}</span><br/>
                        <strong style='font-size:0.95rem'>{row['campaign_name']}</strong><br/>
                        <span class='exec-sub'>Projected: <strong>{currency}{row['revenue_p50']:,.0f}</strong> · ROAS: {row['roas_p50']:.2f}x · Investment: {currency}{spend_val:,.0f}</span>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── AI-Generated Narrative Insights ──
            st.markdown("#### AI-Generated Insights")
            st.caption("Automated narrative analysis with grounded, verified data points — every figure is validated against the underlying model output")

            if causal_summaries:
                for cs in causal_summaries:
                    scope = cs.get("scope_label", "unknown")
                    summary_text = cs.get("summary", "")
                    drivers = cs.get("key_drivers", [])
                    risks = cs.get("risk_flags", [])
                    source = cs.get("source", "unknown")

                    if scope == "total":
                        scope_display = "Overall Portfolio"
                    else:
                        scope_display = f"{scope.capitalize()} Channel"

                    with st.container(border=True):
                        st.markdown(f"**{scope_display}**")
                        st.markdown(summary_text)
                        if drivers:
                            driver_text = ", ".join(
                                d.replace("_", " ").replace("roll ", "rolling ").title()
                                for d in drivers[:5] if isinstance(d, str)
                            )
                            st.caption(f"Key drivers: {driver_text}")
                        if risks and risks[0] != "No material risk flags from the deterministic checks for this scope.":
                            for r in risks:
                                st.caption(f"⚠ {r}")
                        source_label = "AI-generated (LLM)" if source == "llm" else "Rule-based analysis"
                        st.caption(f"Source: {source_label} · All figures verified against model output")

            st.markdown("---")

            # ── Budget Optimization Backtest ──
            if mpc_backtest or hindsight_regret:
                st.markdown("#### Budget Optimization Potential")
                st.caption("Backtested performance of automated budget reallocation against actual historical outcomes")

                if hindsight_regret:
                    ol_up = hindsight_regret.get("open_loop_vs_actual_uplift_pct")
                    mpc_up = hindsight_regret.get("mpc_vs_actual_uplift_pct")
                    b1, b2 = st.columns(2)
                    with b1:
                        if ol_up is not None:
                            st.markdown(f"""<div class='exec-card'>
                                <h4>Static Reallocation</h4>
                                <div class='exec-big' style='color:#00E676'>+{ol_up:.1%}</div>
                                <div class='exec-sub'>Projected revenue uplift vs. actual allocation.
                                Achieved by redistributing budget from saturated segments to higher-capacity channels.</div>
                            </div>""", unsafe_allow_html=True)
                    with b2:
                        if mpc_up is not None:
                            st.markdown(f"""<div class='exec-card'>
                                <h4>Adaptive Reallocation (MPC)</h4>
                                <div class='exec-big' style='color:#00E676'>+{mpc_up:.1%}</div>
                                <div class='exec-sub'>Projected uplift with rolling 30-day re-optimization.
                                The optimizer adapts to changing market conditions each planning cycle.</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown("---")

            # ── Data Coverage Summary ──
            st.markdown("#### Data Coverage")
            total_rows = sum(r.get("n_rows", 0) for r in ingestion_reports.values() if not r.get("errors"))
            total_camps = sum(r.get("n_campaigns", 0) for r in ingestion_reports.values() if not r.get("errors"))
            n_files = len(ingestion_reports)
            n_errors = sum(1 for r in ingestion_reports.values() if r.get("errors"))

            min_date = pd.to_datetime(canonical_df["date"]).min().strftime("%b %d, %Y")
            max_date = pd.to_datetime(canonical_df["date"]).max().strftime("%b %d, %Y")
            days_span = (pd.to_datetime(canonical_df["date"]).max() - pd.to_datetime(canonical_df["date"]).min()).days

            dq1, dq2, dq3, dq4 = st.columns(4)
            with dq1:
                st.markdown(f"""<div class='exec-card'>
                    <h4>Data Sources</h4>
                    <div class='exec-big'>{n_files}</div>
                    <div class='exec-sub'>{"All ingested successfully" if n_errors == 0 else f"{n_errors} source(s) with issues"}</div>
                </div>""", unsafe_allow_html=True)
            with dq2:
                st.markdown(f"""<div class='exec-card'>
                    <h4>Records Analyzed</h4>
                    <div class='exec-big'>{total_rows:,}</div>
                    <div class='exec-sub'>Daily campaign-level observations</div>
                </div>""", unsafe_allow_html=True)
            with dq3:
                st.markdown(f"""<div class='exec-card'>
                    <h4>Campaigns Tracked</h4>
                    <div class='exec-big'>{total_camps}</div>
                    <div class='exec-sub'>Across {len(sorted(canonical_df['channel'].unique()))} channels</div>
                </div>""", unsafe_allow_html=True)
            with dq4:
                st.markdown(f"""<div class='exec-card'>
                    <h4>Date Range</h4>
                    <div class='exec-big'>{days_span:,}d</div>
                    <div class='exec-sub'>{min_date} – {max_date}</div>
                </div>""", unsafe_allow_html=True)

            # ── Export ──
            # Plain-text summary a non-technical reader can forward or print;
            # pulls from variables set earlier in this tab, with safe
            # fallbacks in case a section above didn't render (e.g. fewer
            # than 2 channel/campaign-type groups, so no reallocation ran).
            st.markdown("---")
            export_lines = [
                f"AIgnition 3.0 — Executive Summary (data as of {as_of})",
                f"Horizon: next {horizon} days" + (f" · Scope: {sel_channel}" if sel_channel != "All" else ""),
                "",
                f"Projected revenue: {currency}{total_revenue_p50:,.0f} "
                f"(range {currency}{total_revenue_p10:,.0f} - {currency}{total_revenue_p90:,.0f})",
                f"Ad spend: {currency}{total_spend:,.0f}   Blended ROAS: {blended_roas:.2f}x   "
                f"Campaigns: {n_total_campaigns}",
                "",
                "Per-channel recommendations:",
            ]
            for ch in channels_in_scope:
                ch_data = pred_h[pred_h["channel"] == ch]
                ch_spend = ch_data["assumed_spend_total"].sum()
                ch_rev = ch_data["revenue_p50"].sum()
                ch_roas = ch_rev / ch_spend if ch_spend > 0 else 0
                export_lines.append(f"  - {ch.capitalize()}: ROAS {ch_roas:.2f}x, spend {currency}{ch_spend:,.0f}, "
                                     f"revenue {currency}{ch_rev:,.0f}")

            if "move_rows" in globals() and move_rows:
                export_lines += ["", f"Recommended budget moves (same {currency}{move_total:,.0f}/day total, "
                                      f"est. {'+' if daily_uplift >= 0 else ''}{currency}{daily_uplift:,.0f}/day uplift):"]
                for p, cur, rec, delta in move_rows[:6]:
                    direction = "Increase" if delta > 0 else "Decrease"
                    export_lines.append(f"  - {direction} {p[0].capitalize()} — {p[1]}: "
                                         f"{currency}{cur:,.0f}/day -> {currency}{rec:,.0f}/day "
                                         f"({'+' if delta > 0 else ''}{currency}{delta:,.0f}/day)")

            st.download_button(
                "📄 Download this summary (.txt)",
                data="\n".join(export_lines),
                file_name=f"aignition_exec_summary_{as_of.replace(' ', '_').replace(',', '')}.txt",
                mime="text/plain",
            )

        else:
            st.info("No campaigns match the current filter. Adjust the sidebar selections.")
    else:
        st.warning("Prediction data not available. Execute the pipeline with `./run.sh` to generate forecasts.")

# ─────────────────────────────────────────────────────────────────────────────
# ① Data health
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Data health — this run's ingestion report")
    st.caption("Recomputed live from the current --data-dir on every run, per §A — not cached from training time.")
    for fname, r in ingestion_reports.items():
        with st.container(border=True):
            top = st.columns([2, 1, 1, 1])
            top[0].markdown(f"**{fname}**  ·  channel: `{r['channel']}`")
            if r.get("errors"):
                top[1].markdown(":red[SKIPPED]")
                st.error("; ".join(r["errors"]))
                continue
            top[1].metric("Rows", f"{r['n_rows']:,}")
            top[2].metric("Campaigns", r["n_campaigns"])
            top[3].markdown(
                f":green[direct revenue]" if r["revenue_confidence"] == "direct" else ":orange[proxy revenue]"
            )
            c1, c2 = st.columns(2)
            with c1:
                if r["mapped"]:
                    st.caption("Mapped (exact alias)")
                    st.json(r["mapped"], expanded=False)
                if r["fuzzy_mapped"]:
                    st.caption("Mapped (fuzzy fallback)")
                    st.json(r["fuzzy_mapped"], expanded=False)
            with c2:
                if r["ignored_columns"]:
                    st.caption(f"Ignored columns: {', '.join(r['ignored_columns'])}")
                if r["missing_optional"]:
                    st.caption(f"Missing (left as genuine NaN): {', '.join(r['missing_optional'])}")
                st.caption(f"Campaign type source: {r['campaign_type_source']} "
                           f"(unclassified: {r['campaign_type_unclassified_count']})")
            for w in r.get("warnings", []):
                st.warning(w)


# ─────────────────────────────────────────────────────────────────────────────
# ② Fan chart
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader(f"Probabilistic forecast — {sel_channel} / {sel_type} · {horizon}-day window")
    if predictions_df is None:
        st.warning("No predictions.csv found yet. Run `run.sh` (generate_features.py + predict.py) first.")
    else:
        sub = _filter_scope(predictions_df)
        sub = sub[sub["horizon_days"] == horizon]
        if len(sub) == 0:
            st.info("No campaigns in this scope.")
        else:
            lookback_days = st.select_slider(
                "History shown", options=[60, 90, 120, 180, 270, 365], value=120,
                help="How much trailing daily-actual history to draw before the forecast window.",
            )
            hist_scope = _filter_scope(canonical_df)
            fig, fc_meta = _build_fan_chart(hist_scope, sub, horizon, lookback_days)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Historical daily revenue (raw + 7-day average) through {fc_meta['forecast_as_of'].date()}, "
                f"followed by the {horizon}-day forecast rendered as its own implied average daily rate — the "
                "model predicts one aggregate total for the window (§B.1), not a daily path, so the forward "
                "region is a flat band rather than a widening cone. See the P10/P50/P90 metrics below for the "
                "actual window totals this band is derived from.",
            )

            show_decomp = st.checkbox(
                "Show day-by-day seasonal decomposition of this forecast",
                value=False,
                help="Distributes the SAME aggregate forecast above across individual days using this "
                     "scope's own day-of-week pattern — a decomposition for readability, not a second, "
                     "independently-fit daily model. Off by default; the flat band above is the model's "
                     "actual, honest output.",
            )
            if show_decomp:
                decomp_fig = _build_seasonal_decomposition_chart(
                    hist_scope, fc_meta["rate"], fc_meta["forecast_as_of"], horizon,
                )
                if decomp_fig is None:
                    st.info(
                        "Not enough daily history in this scope (need ~3+ weeks) to estimate a reliable "
                        "day-of-week pattern — showing the flat band above only."
                    )
                else:
                    st.plotly_chart(decomp_fig, use_container_width=True)
                    st.caption(
                        "Each day's share comes from this scope's own historical day-of-week average "
                        "revenue, normalized so the day-by-day median averages back to exactly the same "
                        "flat-band rate above — every quantile is scaled by the identical per-day factor, "
                        "so this reshapes the SAME forecast rather than re-deriving a new one. Genuine "
                        "day-to-day uncertainty is not modeled here (the underlying forecast has none to "
                        "give — §B.1); treat the shape as illustrative, the flat band above as the number."
                    )

            c1, c2, c3, c4 = st.columns(4)
            median_col = "revenue_p50"
            c1.metric("P10 (downside)", f"{sub['revenue_p10'].sum():,.0f}" if "revenue_p10" in sub else "—")
            c2.metric("P50 (median)", f"{sub[median_col].sum():,.0f}")
            c3.metric("P90 (upside)", f"{sub['revenue_p90'].sum():,.0f}" if "revenue_p90" in sub else "—")

            roas_med = roas_lo = roas_hi = None
            if reconciled_df is not None:
                node_id = _hierarchy_node_id(sel_channel, sel_type)
                node_row = reconciled_df[
                    (reconciled_df["horizon_days"] == horizon) & (reconciled_df["unique_id"] == node_id)
                ]
                if len(node_row) and pd.notna(node_row.iloc[0].get("roas_reconciled_median")):
                    row = node_row.iloc[0]
                    roas_med, roas_lo, roas_hi = row["roas_reconciled_median"], row.get("roas_q0.1"), row.get("roas_q0.9")
            c4.metric("Blended ROAS", f"{roas_med:.2f}x" if roas_med is not None else "—")
            if roas_med is not None and pd.notna(roas_lo) and pd.notna(roas_hi):
                st.caption(f"Blended ROAS likely range (P10–P90): {roas_lo:.2f}x – {roas_hi:.2f}x")

            n_flagged = int(sub["plausibility_flag"].sum())
            if n_flagged:
                st.markdown(
                    f"<div class='caveat-box'>{n_flagged} of {len(sub)} campaign forecasts in this scope were "
                    f"outside their group's historical ROAS envelope — see the capped display value and per-row "
                    f"caveat in the table below.</div>", unsafe_allow_html=True,
                )

            sub = sub.copy()
            sub["roas_range"] = "—"
            if reconciled_df is not None:
                camp_roas = reconciled_df[
                    (reconciled_df["horizon_days"] == horizon) & (reconciled_df["level"] == "campaign")
                ].copy()
                if "roas_q0.1" in camp_roas.columns and len(camp_roas):
                    camp_roas["campaign_id"] = camp_roas["unique_id"].str.rsplit("/", n=1).str[-1]
                    camp_roas = camp_roas[["campaign_id", "roas_q0.1", "roas_q0.9"]].dropna()
                    sub = sub.merge(camp_roas, on="campaign_id", how="left", suffixes=("", "_rng"))
                    has_rng = sub["roas_q0.1"].notna() & sub["roas_q0.9"].notna()
                    sub.loc[has_rng, "roas_range"] = sub.loc[has_rng].apply(
                        lambda r: f"{r['roas_q0.1']:.2f}x – {r['roas_q0.9']:.2f}x", axis=1)

            st.dataframe(
                sub[["campaign_id", "campaign_name", "channel", "campaign_type", "revenue_p50",
                     f"{median_col}_display", "roas_p50", "roas_range", "saturation_status", "plausibility_flag"]]
                .rename(columns={"roas_range": "roas_p10_p90_range"})
                .sort_values("revenue_p50", ascending=False),
                use_container_width=True, height=320,
            )

            with st.expander("Percentile distribution at this horizon (detail view)"):
                st.caption(
                    "A complementary view of the same forecast: the summed quantile curve at this horizon, "
                    "percentile on the x-axis rather than time — useful for reading off any specific percentile "
                    "directly, not a second time series.",
                )
                q_cols = sorted([c for c in sub.columns if c.startswith("revenue_p") and c[-1].isdigit()])
                totals = sub[q_cols].sum()
                dist_fig = go.Figure()
                xs = [int(c.replace("revenue_p", "")) for c in q_cols]
                dist_fig.add_trace(go.Scatter(x=xs, y=totals.values, mode="lines+markers",
                                               line=dict(color="#00D2FF", width=3), name="Summed quantile forecast"))
                dist_fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Percentile", yaxis_title="Revenue (summed across scope)", height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(dist_fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ③ Budget what-if
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Budget what-if — live re-scored, monotonic by construction")
    st.caption(
        "Recomputes the pooled quantile model at a hypothetical daily-spend level for every campaign in scope. "
        "This tab sums campaign-level forecasts directly (no hierarchy reconciliation) so it stays instant; "
        "the Breakdown tab shows the fully reconciled figures.",
    )
    scope_snapshot = _filter_scope(snapshot)
    currency = os.environ.get("CURRENCY_SYMBOL", "$")
    if len(scope_snapshot) == 0:
        st.info("No campaigns in this scope.")
    else:
        multiplier = st.slider("Budget multiplier vs. recent trailing pace", 0.25, 2.5, 1.0, 0.05)

        grid = np.round(np.arange(0.25, 2.55, 0.15), 2)

        roas_bounds = bundle["roas_bounds"]
        curve_totals = {q: [] for q in bundle["quantiles"]}
        median_idx = bundle["quantiles"].index(0.5)
        flagged_by_grid = []
        implied_daily_spend_grid = []
        for m in grid:
            scored = score_scenario(bundle, scope_snapshot, horizon, float(m))
            clamped_median, n_flagged = _clamp_median_column(scored, roas_bounds)
            flagged_by_grid.append(n_flagged)
            implied_daily_spend_grid.append(float(scored["assumed_spend_total"].sum()) / horizon)
            for i, q in enumerate(bundle["quantiles"]):
                curve_totals[q].append(clamped_median.sum() if i == median_idx else scored[f"q{q}"].sum())
        curve_matrix = np.column_stack([curve_totals[q] for q in bundle["quantiles"]])
        curve_matrix = M.enforce_monotonic_along_grid(grid, curve_matrix)

        fig = go.Figure()
        lo_idx, hi_idx, mid_idx = 0, len(bundle["quantiles"]) - 1, median_idx
        fig.add_trace(go.Scatter(x=grid, y=curve_matrix[:, hi_idx], mode="lines", line=dict(width=0), showlegend=False,
                                  customdata=implied_daily_spend_grid,
                                  hovertemplate="Budget x%{x:.2f} · implied spend " + currency +
                                                "%{customdata:,.0f}/day<br>P90: " + currency +
                                                "%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=grid, y=curve_matrix[:, lo_idx], mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(0,210,255,0.18)", name="P10–P90 band",
                                  customdata=implied_daily_spend_grid,
                                  hovertemplate="Budget x%{x:.2f} · implied spend " + currency +
                                                "%{customdata:,.0f}/day<br>P10: " + currency +
                                                "%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=grid, y=curve_matrix[:, mid_idx], mode="lines+markers",
                                  line=dict(color="#00D2FF", width=3), name="Median (plausibility-clamped)",
                                  customdata=implied_daily_spend_grid,
                                  hovertemplate="Budget x%{x:.2f} · implied spend " + currency +
                                                "%{customdata:,.0f}/day<br>Median revenue: " + currency +
                                                "%{y:,.0f}<extra></extra>"))
        fig.add_vline(x=multiplier, line_dash="dash", line_color="#F8FAFC")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Budget multiplier (x recent pace)", yaxis_title=f"Revenue over {horizon} days",
            height=380, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        current_median = float(np.interp(multiplier, grid, curve_matrix[:, mid_idx]))
        baseline_median = float(np.interp(1.0, grid, curve_matrix[:, mid_idx]))
        delta = current_median - baseline_median
        c1, c2 = st.columns(2)
        c1.metric(f"Median revenue @ {multiplier}x", f"{current_median:,.0f}", f"{delta:+,.0f} vs 1.0x")
        c2.metric("Implied spend change", f"{(multiplier - 1) * 100:+.0f}%")

        n_flagged_here = int(round(np.interp(multiplier, grid, flagged_by_grid)))
        if n_flagged_here:
            st.markdown(
                f"<div class='caveat-box'>~{n_flagged_here} campaign(s) in this scope imply a ROAS outside "
                f"their group's historical envelope near this budget level — each is capped to that group's "
                f"plausible range before being summed into the median above (§G.3), same rule the static "
                f"Forecast tab applies.</div>", unsafe_allow_html=True,
            )

        observed_scope_daily_spend = _filter_scope(canonical_df).groupby("date")["spend"].sum()
        observed_max_daily_spend = float(observed_scope_daily_spend.max()) if len(observed_scope_daily_spend) else 0.0
        current_implied_daily_spend = float(np.interp(multiplier, grid, implied_daily_spend_grid))
        if observed_max_daily_spend > 0 and current_implied_daily_spend > observed_max_daily_spend:
            overshoot = current_implied_daily_spend / observed_max_daily_spend
            st.markdown(
                f"<div class='caveat-box'>⚠ Extrapolation warning: at {multiplier}x, this scope's implied daily "
                f"spend (~{current_implied_daily_spend:,.0f}/day) is {overshoot:.1f}× its historical daily maximum "
                f"(~{observed_max_daily_spend:,.0f}/day) over the ingested history. Both the quantile model and "
                f"the Hill saturation curve below were fit on spend at or below that historical range — beyond "
                f"it they're extrapolating, not interpolating, so treat this scenario's numbers with more caution."
                f"</div>", unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**Saturation curve — fitted Hill response (§F)**")
        if not hill_curves_list:
            st.info("No hill_curves.json found yet. Run predict.py first.")
        else:
            available_pairs = [(c["channel"], c["campaign_type"]) for c in hill_curves_list]
            if sel_channel != "All":
                available_pairs = [p for p in available_pairs if p[0] == sel_channel]
            if sel_type != "All":
                available_pairs = [p for p in available_pairs if p[1] == sel_type]

            if not available_pairs:
                st.info("No fitted curve for this exact scope.")
            else:
                if len(available_pairs) == 1:
                    curve_channel, curve_type = available_pairs[0]
                else:
                    by_pair = {(c["channel"], c["campaign_type"]): c for c in hill_curves_list}
                    fit_ok_pairs = [p for p in available_pairs if by_pair[p]["fit_ok"]]
                    default_pair = fit_ok_pairs[0] if fit_ok_pairs else available_pairs[0]
                    pair_labels = [f"{c} · {t}" for c, t in available_pairs]
                    label_to_pair = dict(zip(pair_labels, available_pairs))
                    default_label = f"{default_pair[0]} · {default_pair[1]}"
                    chosen_label = st.selectbox(
                        "Channel / campaign type for the saturation curve",
                        options=pair_labels,
                        index=pair_labels.index(default_label),
                    )
                    curve_channel, curve_type = label_to_pair[chosen_label]
                curve_entry = next(
                    c for c in hill_curves_list
                    if (c["channel"], c["campaign_type"]) == (curve_channel, curve_type)
                )

                grp_daily = (
                    canonical_df[(canonical_df["channel"] == curve_channel)
                                 & (canonical_df["campaign_type"] == curve_type)]
                    .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
                    .groupby("date", as_index=False)
                    .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
                )
                grp_daily = grp_daily[grp_daily["spend"] > 0]

                grp_snapshot = snapshot[(snapshot["channel"] == curve_channel)
                                         & (snapshot["campaign_type"] == curve_type)]
                current_daily_spend = (
                    float(grp_snapshot["spend_roll_mean_28"].fillna(grp_snapshot["spend_lag_7"]).fillna(0.0).sum())
                    if len(grp_snapshot) else 0.0
                )

                max_obs_spend = float(grp_daily["spend"].max()) if len(grp_daily) else (curve_entry.get("K") or 1.0)
                x_max = max(max_obs_spend * 1.15, (curve_entry.get("K") or 0.0) * 2.2, current_daily_spend * 1.15, 1.0)
                xs = np.linspace(0, x_max, 200)
                ys = [hill_predict(float(x), curve_entry) for x in xs]

                fig_hill = go.Figure()
                fig_hill.add_trace(go.Scatter(
                    x=grp_daily["spend"], y=grp_daily["revenue"], mode="markers",
                    name="Observed daily (spend, revenue)", marker=dict(size=6, color="#64748B", opacity=0.55),
                    hovertemplate="Spend: " + currency + "%{x:,.0f}/day<br>Revenue: " + currency +
                                  "%{y:,.0f}/day<extra></extra>",
                ))
                fig_hill.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    name="Fitted Hill curve" if curve_entry["fit_ok"] else "Fallback (flat historical ROAS)",
                    line=(dict(color="#00D2FF", width=3) if curve_entry["fit_ok"]
                          else dict(color="#64748B", width=2, dash="dash")),
                    hovertemplate="At spend " + currency + "%{x:,.0f}/day, predicted revenue " + currency +
                                  "%{y:,.0f}/day<extra></extra>",
                ))
                if len(grp_snapshot):
                    fig_hill.add_trace(go.Scatter(
                        x=[current_daily_spend], y=[hill_predict(current_daily_spend, curve_entry)],
                        mode="markers", name="Current operating point",
                        marker=dict(size=14, color="#F8FAFC", symbol="diamond", line=dict(width=2, color="#0F172A")),
                        hovertemplate="Current spend: " + currency + "%{x:,.0f}/day<br>Predicted revenue: " +
                                      currency + "%{y:,.0f}/day<extra></extra>",
                    ))
                fig_hill.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Daily spend (summed across this group's campaigns)", yaxis_title="Daily revenue",
                    height=380, margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_hill, use_container_width=True)

                if curve_entry["fit_ok"]:
                    sat = saturation_status(current_daily_spend, curve_entry)
                    st.caption(
                        f"{curve_channel} · {curve_type}: fitted Hill curve (R²={curve_entry['r_squared']:.2f}, "
                        f"half-saturation K≈{curve_entry['K']:,.0f}/day) — "
                        f"{sat['status'].replace('_', ' ')} at the current trailing pace. This is a secondary "
                        f"sanity curve (§F), not the primary forecast."
                    )
                else:
                    st.caption(
                        f"{curve_channel} · {curve_type}: no statistically reliable saturation curve could be "
                        f"fit ({curve_entry.get('note', 'insufficient data')}) — showing the flat historical-ROAS "
                        f"fallback line instead of a fabricated curve."
                    )

        st.markdown("---")
        st.markdown("**Budget allocator — best split of a fixed total across channels (§F.3)**")
        if not hill_curves_list:
            st.info("No hill_curves.json found yet. Run predict.py first.")
        else:
            opt_pairs = [(c["channel"], c["campaign_type"]) for c in hill_curves_list]
            if sel_channel != "All":
                opt_pairs = [p for p in opt_pairs if p[0] == sel_channel]
            if sel_type != "All":
                opt_pairs = [p for p in opt_pairs if p[1] == sel_type]

            if len(opt_pairs) < 2:
                st.info(
                    "Broaden the channel/campaign-type filter above (e.g. set both to \"All\") to see a "
                    "cross-channel allocation — there's only one group in the current scope to allocate across."
                )
            else:
                opt_curves = {p: next(c for c in hill_curves_list if (c["channel"], c["campaign_type"]) == p)
                              for p in opt_pairs}
                opt_hist_spend = {}
                for p in opt_pairs:
                    g_snap = snapshot[(snapshot["channel"] == p[0]) & (snapshot["campaign_type"] == p[1])]
                    opt_hist_spend[p] = (
                        float(g_snap["spend_roll_mean_28"].fillna(g_snap["spend_lag_7"]).fillna(0.0).sum())
                        if len(g_snap) else 0.0
                    )

                default_total = sum(opt_hist_spend.values())
                budget_col, floor_col = st.columns([2, 1])
                with budget_col:
                    total_budget_input = st.number_input(
                        "Total daily budget to allocate across these groups",
                        min_value=0.0, value=round(default_total, 2), step=max(round(default_total * 0.05, 2), 1.0),
                        help="Defaults to what this scope is already spending per day, summed across groups — "
                             "adjust it to explore a bigger or smaller total.",
                    )
                with floor_col:
                    use_floor = st.checkbox("Set a minimum blended ROAS", value=False)
                    roas_floor_input = (
                        st.number_input("Minimum blended ROAS", min_value=0.0, value=1.0, step=0.1,
                                        help="Real agencies often can't accept pure revenue-max if it tanks "
                                             "blended return. When set, the allocator may recommend spending "
                                             "LESS than the full budget above rather than dip below this floor.")
                        if use_floor else None
                    )

                if total_budget_input <= 0:
                    st.info("Enter a budget above zero to see a recommended allocation.")
                else:
                    opt_result = optimize_budget_allocation(
                        opt_curves, opt_hist_spend, total_daily_budget=total_budget_input,
                        min_blended_roas=roas_floor_input,
                    )
                    current_revenue_at_input_total = sum(
                        hill_predict(opt_hist_spend[p] * (total_budget_input / default_total) if default_total > 0 else 0.0,
                                      opt_curves[p])
                        for p in opt_pairs
                    )
                    optimized_revenue = opt_result["predicted_daily_revenue"]
                    uplift_pct = (
                        (optimized_revenue / current_revenue_at_input_total - 1.0)
                        if current_revenue_at_input_total > 1e-6 else None
                    )

                    if opt_result["roas_floor_binding"]:
                        st.warning(
                            f"The {roas_floor_input:.2f}x floor is binding: spending the full "
                            f"{total_budget_input:,.0f}/day would dilute blended ROAS below your floor, so "
                            f"{opt_result['unallocated_budget']:,.0f}/day is deliberately left unallocated to "
                            f"hold blended ROAS at {opt_result['blended_roas']:.2f}x."
                        )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted daily revenue — current-shaped split", f"{current_revenue_at_input_total:,.0f}")
                    c2.metric(
                        "Predicted daily revenue — optimized split",
                        f"{optimized_revenue:,.0f}",
                        help=f"Approximate range: {opt_result['predicted_daily_revenue_low']:,.0f} – "
                             f"{opt_result['predicted_daily_revenue_high']:,.0f} (see caption below).",
                    )
                    c3.metric("Potential uplift", f"{uplift_pct:+.1%}" if uplift_pct is not None else "—")
                    st.caption(
                        f"Optimized-split range (approximate, see note below): "
                        f"{opt_result['predicted_daily_revenue_low']:,.0f} – {opt_result['predicted_daily_revenue_high']:,.0f} · "
                        f"realized blended ROAS at this allocation: {opt_result['blended_roas']:.2f}x"
                    )

                    rows = []
                    for p in sorted(opt_pairs, key=lambda p: -opt_result["allocation"].get(p, 0.0)):
                        cur = opt_hist_spend[p]
                        rec = opt_result["allocation"].get(p, 0.0)
                        rows.append({
                            "channel": p[0], "campaign_type": p[1],
                            "current daily spend": round(cur, 0),
                            "recommended daily spend": round(rec, 0),
                            "change": round(rec - cur, 0),
                            "marginal ROAS at recommendation": round(opt_result["marginal_roas"].get(p, 0.0), 2),
                            "curve": "fitted Hill" if opt_curves[p]["fit_ok"] else "flat ROAS fallback",
                        })
                    alloc_df = pd.DataFrame(rows)
                    st.dataframe(alloc_df, hide_index=True, use_container_width=True)

                    fig_alloc = go.Figure()
                    labels = [f"{p[0]}·{p[1]}" for p in opt_pairs]
                    fig_alloc.add_trace(go.Bar(
                        x=labels, y=[opt_hist_spend[p] for p in opt_pairs], name="Current daily spend",
                        marker_color="#64748B",
                    ))
                    fig_alloc.add_trace(go.Bar(
                        x=labels, y=[opt_result["allocation"].get(p, 0.0) for p in opt_pairs], name="Recommended daily spend",
                        marker_color="#00D2FF",
                    ))
                    fig_alloc.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        barmode="group", height=380, yaxis_title="Daily spend",
                        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.3),
                    )
                    st.plotly_chart(fig_alloc, use_container_width=True)

                    st.caption(
                        "Built on the same Hill saturation curves shown above (steady-state daily response, "
                        "not the calibrated 30/60/90-day forecast) — treat this as directional budget-planning "
                        "guidance, not a guaranteed outcome. Solved with dynamic programming rather than a "
                        "simple marginal-return walk, since these curves are S-shaped, not concave, and a naive "
                        "greedy split can get stuck. Each group is capped at 4x its own historical daily spend, "
                        "so the recommendation stays within a defensible extrapolation range of the data its "
                        "curve was actually fit on, rather than proposing an implausible all-in bet on a group "
                        "with little history. \"Marginal ROAS\" is the return on the NEXT rupee at the "
                        "recommended spend level (the derivative of the curve), not revenue/spend — at a true "
                        "optimum it should be roughly equal across every group with room left to move, since "
                        "that's the condition the DP is actually solving for. The optimized-revenue range is an "
                        "approximation from each group's own historical residual spread around its curve, "
                        "combined assuming independence — a rough band for a secondary sanity layer, not a "
                        "substitute for §6's properly calibrated CQR intervals on the primary forecast."
                    )

        if mpc_backtest:
            with st.expander(
                "MPC-style rolling-horizon reallocation backtest (§F.4) — reported, not the shipped allocator",
                expanded=False,
            ):
                st.caption(
                    "The what-if allocator above decides once, using whatever curves happen to be fitted "
                    "right now. This backtest asks a different question on this account's own historical "
                    "timeline: does periodically re-fitting the curves and RE-solving the allocation as new "
                    "data arrives (\"MPC\", closed-loop) earn back more revenue than deciding once at the start "
                    "and running that same plan for the whole horizon (\"open-loop\")? Both are scored against "
                    "an identical, purely retrospective ground-truth curve fit on what actually happened in "
                    "each window — only the allocation differs — and each planned allocation is perturbed by "
                    f"execution noise (±{mpc_backtest['spend_execution_noise_std_frac']:.0%}, planned spend "
                    "rarely lands exactly on target) before being scored."
                )
                lift = mpc_backtest.get("mpc_vs_open_loop_relative_lift")
                c1, c2, c3 = st.columns(3)
                c1.metric("Open-loop avg. daily revenue", f"{mpc_backtest['open_loop_avg_daily_revenue']:,.0f}")
                c2.metric("MPC avg. daily revenue", f"{mpc_backtest['mpc_avg_daily_revenue']:,.0f}")
                c3.metric("MPC vs. open-loop lift", f"{lift:+.1%}" if lift is not None else "—")

                win_rows = [{
                    "window": f"{w['window_start']} → {w['window_end']}",
                    "open-loop realized daily revenue": round(w["open_loop_realized_daily_revenue"], 0),
                    "MPC realized daily revenue": round(w["mpc_realized_daily_revenue"], 0),
                    "groups with a fresh curve fit this window": w["n_groups_with_fresh_curve_fit"],
                } for w in mpc_backtest["windows"]]
                st.dataframe(pd.DataFrame(win_rows), hide_index=True, use_container_width=True)
                st.caption(
                    f"Backtest window: {mpc_backtest['horizon_days']} days starting "
                    f"{mpc_backtest['backtest_start']}, replanned every {mpc_backtest['replan_every_days']} days "
                    f"({mpc_backtest['n_windows']} windows) — the same cadence as this report's own 30/60/90-day "
                    "horizons. A lift near zero (or negative) is still a genuine, honest result: it means "
                    "channel effectiveness didn't drift enough over this particular historical window for "
                    "re-solving to earn back more than the noise/estimation cost of refitting on less data per "
                    "window, not that the mechanism is broken — see docs/technical_documentation.md §9a."
                )

        if hindsight_regret and hindsight_regret.get("windows"):
            with st.expander(
                "Hindsight-regret audit (§F.5) — vs. what actually happened, not vs. another algorithm",
                expanded=False,
            ):
                st.caption(
                    "The backtest above compares two algorithms (MPC vs. open-loop) against each other. This "
                    "compares the tool against the one number that carries no estimation at all: what this "
                    "account's real historical spend and revenue actually were, for the identical windows above. "
                    "Positive means the tool's curve-based estimate says it would have beaten the real decision "
                    "that was actually made — read as \"the tool's estimate,\" not as certainty, since we can "
                    "never truly know what revenue an unplayed alternative allocation would have produced."
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Actual avg. daily revenue", f"{hindsight_regret['actual_avg_daily_revenue']:,.0f}")
                ol_up = hindsight_regret.get("open_loop_vs_actual_uplift_pct")
                mpc_up = hindsight_regret.get("mpc_vs_actual_uplift_pct")
                c2.metric("Open-loop vs. actual", f"{ol_up:+.1%}" if ol_up is not None else "—")
                c3.metric("MPC vs. actual", f"{mpc_up:+.1%}" if mpc_up is not None else "—")

                regret_rows = [{
                    "window": f"{w['window_start']} → {w['window_end']}",
                    "actual daily revenue": round(w["actual_daily_revenue"], 0),
                    "open-loop regret": round(w["open_loop_regret_daily_revenue"], 0),
                    "MPC regret": round(w["mpc_regret_daily_revenue"], 0),
                } for w in hindsight_regret["windows"]]
                st.dataframe(pd.DataFrame(regret_rows), hide_index=True, use_container_width=True)
                st.caption(
                    "\"Regret\" = the tool's estimated revenue for its recommended allocation, minus what "
                    "actually happened, for that same window — positive means the tool's estimate says it left "
                    "less on the table than the real historical decision did. A negative regret in any single "
                    "window is a genuine, reportable outcome, not hidden here."
                )


# ─────────────────────────────────────────────────────────────────────────────
# ④ Reconciled breakdown
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader(f"Reconciled breakdown — {horizon}-day window")
    if reconciled_df is None:
        st.warning("No reconciled_hierarchy.csv found yet. Run predict.py first.")
    else:
        hsub = reconciled_df[reconciled_df["horizon_days"] == horizon].copy()
        err = hsub["max_abs_coherence_error"].iloc[0] if len(hsub) else None
        if err is not None:
            st.markdown(
                f"<div class='ok-box'>Coherence check: channel totals sum exactly to the grand total, and "
                f"campaign_type totals sum exactly to their channel (max abs error = {err:.6f}) — enforced by "
                f"MinTrace reconciliation (Wickramasuriya et al. 2019), not just a hopeful sum.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**Full hierarchy — campaign → campaign type → channel → total**")
        chart_kind = st.radio(
            "Chart type", ["Treemap", "Sunburst"], horizontal=True, label_visibility="collapsed",
        )
        currency = os.environ.get("CURRENCY_SYMBOL", "$")
        hier_frame = _build_hierarchy_frame(hsub)
        trace_kwargs = dict(
            ids=hier_frame["id"], labels=hier_frame["label"], parents=hier_frame["parent"],
            values=hier_frame["value"], branchvalues="total", maxdepth=3,
            marker=dict(colorscale=[[0, "#020617"], [1, "#00D2FF"]], line=dict(width=1, color="#0F172A")),
            hovertemplate="<b>%{label}</b><br>Reconciled median: " + currency + "%{value:,.0f}<br>"
                          "Share of parent: %{percentParent:.1%}<extra></extra>",
        )
        fig_h = go.Figure(go.Treemap(**trace_kwargs) if chart_kind == "Treemap" else go.Sunburst(**trace_kwargs))
        fig_h.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             height=480, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption(
            "Click a segment to drill in (starts collapsed at the campaign-type level — 168 individual campaigns "
            "is a lot to show at once). This communicates the hierarchy's proportions directly; the per-level "
            "charts below add the reconciled uncertainty band (P10–P90) a treemap/sunburst can't show.",
        )

        for level, label in [("total", "Total"), ("channel", "By channel"), ("campaign_type", "By campaign type")]:
            level_rows = hsub[hsub["level"] == level].sort_values("reconciled_median", ascending=False)
            if len(level_rows) == 0:
                continue
            st.markdown(f"**{label}**")
            col_rev, col_roas = st.columns(2)
            with col_rev:
                fig = go.Figure(go.Bar(
                    x=level_rows["unique_id"], y=level_rows["reconciled_median"],
                    marker_color="#00D2FF",
                    customdata=np.column_stack([level_rows["q0.1"], level_rows["q0.9"]]),
                    error_y=dict(type="data", array=level_rows["q0.9"] - level_rows["reconciled_median"],
                                 arrayminus=level_rows["reconciled_median"] - level_rows["q0.1"], color="#64748B"),
                    hovertemplate="<b>%{x}</b><br>Median: " + currency + "%{y:,.0f}<br>Range: " + currency +
                                  "%{customdata[0]:,.0f} – " + currency + "%{customdata[1]:,.0f}<extra></extra>",
                ))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Revenue")
                st.plotly_chart(fig, use_container_width=True)
            with col_roas:
                if "roas_reconciled_median" in level_rows.columns and level_rows["roas_reconciled_median"].notna().any():
                    fig_roas = go.Figure(go.Bar(
                        x=level_rows["unique_id"], y=level_rows["roas_reconciled_median"],
                        marker_color="#00E676",
                        customdata=np.column_stack([level_rows["roas_q0.1"], level_rows["roas_q0.9"]]),
                        error_y=dict(
                            type="data",
                            array=level_rows["roas_q0.9"] - level_rows["roas_reconciled_median"],
                            arrayminus=level_rows["roas_reconciled_median"] - level_rows["roas_q0.1"],
                            color="#64748B",
                        ),
                        hovertemplate="<b>%{x}</b><br>Median ROAS: %{y:.2f}x<br>Range: %{customdata[0]:.2f}x – "
                                      "%{customdata[1]:.2f}x<extra></extra>",
                    ))
                    fig_roas.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="ROAS", yaxis_tickformat=".1f",
                    )
                    st.plotly_chart(fig_roas, use_container_width=True)
                else:
                    st.info("No ROAS range available for this level (spend wasn't tracked for this run).")


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ AI Summary
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Grounded AI causal summary")
    scope_label = sel_channel if sel_channel != "All" else "total"
    match = None
    if causal_summaries:
        match = next((s for s in causal_summaries if s["scope_label"] == scope_label), None)
    if match is None:
        st.info(
            "No precomputed summary for this scope (predict.py precomputes total + each channel at the "
            "primary horizon). Generating one live from the same grounded pipeline..."
        )
        daily = canonical_df.assign(date=pd.to_datetime(canonical_df["date"]).dt.normalize())
        anomalies = L.detect_anomalies(daily)
        scope_filter = {} if sel_channel == "All" else {"channel": sel_channel}
        pop = L.compute_period_over_period(daily, scope_filter, window=horizon)
        sub = _filter_scope(predictions_df) if predictions_df is not None else None
        sub = sub[sub["horizon_days"] == horizon] if sub is not None else None
        forecast = {
            "revenue_p10": float(sub["revenue_p10"].sum()) if sub is not None and len(sub) else None,
            "revenue_p50": float(sub["revenue_p50"].sum()) if sub is not None and len(sub) else None,
            "revenue_p90": float(sub["revenue_p90"].sum()) if sub is not None and len(sub) else None,
        }
        shap_imp = bundle.get("shap_importance")
        top_drivers = shap_imp["top_features"] if shap_imp else bundle["feature_importance"]
        ctx = L.build_grounding_context(
            scope={"channel": sel_channel if sel_channel != "All" else None,
                   "campaign_type": sel_type if sel_type != "All" else None, "window_days": horizon},
            forecast=forecast, top_drivers=top_drivers, period_over_period=pop,
            anomalies=[a for a in anomalies if sel_channel == "All" or a.get("channel") == sel_channel][:5],
            saturation_status={"status": "see_budget_whatif_tab"},
        )
        match = {**L.generate_causal_summary(ctx), "grounding_context": ctx}

    badge = "✓ LLM output, numerically validated" if match["source"] == "llm" else "Rule-based narrator (deterministic, always grounded)"
    st.caption(badge)
    st.markdown(f"### {match['summary']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Key drivers**")
        for d in match["key_drivers"]:
            st.markdown(f"- {d}")
    with col2:
        st.markdown("**Risk flags**")
        for r in match["risk_flags"]:
            st.markdown(f"- {r}")
    st.caption(match["confidence_note"])
    with st.expander("Grounding context (the only facts the narrator was allowed to use)"):
        st.json(match["grounding_context"])


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ Model Reliability
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Model reliability")
    cv = bundle["cv_reports"]
    fh = bundle["final_holdout"]

    st.markdown("**Cross-validation protocols (development)**")
    cv_table = pd.DataFrame([
        {"protocol": "Walk-forward (time)", "CRPS": cv["walk_forward"]["crps"], "WAPE": cv["walk_forward"]["wape_median"]},
        {"protocol": "Grouped (held-out campaigns)", "CRPS": cv["grouped_by_campaign"]["crps"],
         "WAPE": cv["grouped_by_campaign"]["wape_median"]},
    ])
    st.dataframe(cv_table, use_container_width=True, hide_index=True)
    st.caption(
        "Grouped CV shows lower error than walk-forward here — expected, since walk-forward's early folds train "
        "on much less data (expanding window) while grouped CV always trains on ~80% of the full dataset "
        "regardless of time position. This is evidence AGAINST identity-driven overfitting (which would show "
        "the opposite pattern), not a sign of a problem."
    )

    st.markdown("**Final holdout (evaluated once, never used for tuning)**")
    naive = fh.get("naive_baseline")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CRPS (CQR-calibrated)", f"{fh['cqr_calibrated']['crps']:.0f}")
    c2.metric("WAPE (median)", f"{fh['cqr_calibrated']['wape_median']:.1%}")
    c3.metric("Holdout rows", f"{fh['n_rows']:,}")
    if naive and naive.get("model_wape_improvement_vs_naive_pct") is not None:
        c4.metric(
            "vs. naive baseline",
            f"{naive['model_wape_improvement_vs_naive_pct']:+.1%} WAPE",
            help="Naive baseline = 'continue at the recent 28-day daily pace', scaled to the "
                 "horizon. Same holdout, same metric, never trained on -- this is the model's "
                 "actual skill over the simplest thing an agency could do with a spreadsheet.",
        )
    if naive:
        with st.expander("What is the naive baseline, exactly?"):
            st.caption(naive["description"])
            st.dataframe(
                pd.DataFrame([
                    {"forecast": "Naive (recent pace)", "WAPE": naive["wape_median"]},
                    {"forecast": "This model (CQR-calibrated)", "WAPE": fh["cqr_calibrated"]["wape_median"]},
                ]),
                hide_index=True, use_container_width=True,
            )

    by_horizon = fh.get("by_horizon")
    if by_horizon:
        st.markdown("**Accuracy by forecast horizon**")
        rows = []
        for h, m in sorted(by_horizon.items(), key=lambda kv: int(kv[0])):
            row = {"horizon (days)": int(h), "n": m["n"], "WAPE": m["wape_median"]}
            if m.get("naive_baseline_wape_median") is not None:
                row["naive baseline WAPE"] = m["naive_baseline_wape_median"]
                row["improvement vs. naive"] = m.get("model_wape_improvement_vs_naive_pct")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("**Reliability diagram — nominal vs. empirical interval coverage**")
    rel = fh["reliability_diagram"]
    labels = list(rel.keys())
    nominal = [rel[k]["nominal"] for k in labels]
    empirical = [rel[k]["empirical"] for k in labels]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#64748B"),
                              name="Perfect calibration", hovertemplate="Perfect calibration line<extra></extra>"))
    fig.add_trace(go.Scatter(x=nominal, y=empirical, mode="markers+text", text=labels, textposition="top center",
                              marker=dict(size=12, color="#00D2FF"), name="This model",
                              hovertemplate="<b>%{text}</b><br>Nominal (claimed) coverage: %{x:.0%}<br>"
                                            "Empirical (actual) coverage: %{y:.0%}<extra></extra>"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_title="Nominal coverage", yaxis_title="Empirical coverage", height=380,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    rcbl = fh.get("reconciled_calibration_by_level")
    if rcbl and rcbl.get("n_snapshots"):
        with st.expander(
            "Reconciled-band calibration by hierarchy level (§E re-audit) — is the band "
            "still calibrated after reconciliation, not just the point coherent?"
        ):
            st.caption(
                "Coherence (§8: channel totals sum exactly to the account total, verified "
                "every run) and calibration are separate properties (Principato et al. 2024). "
                "This checks the second one directly: pooling the reconciled quantile band's "
                "empirical vs. nominal coverage across every holdout snapshot, at the total, "
                "channel, campaign_type, and campaign level — not just the base per-campaign-row "
                "level the chart above already covers."
            )
            level_rows = []
            level_colors = {"total": "#00D2FF", "channel": "#6FA8DC", "campaign_type": "#93C47D", "campaign": "#B48EAD"}
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                       line=dict(dash="dash", color="#64748B"), name="Perfect calibration"))
            for level in ("total", "channel", "campaign_type", "campaign"):
                lv = rcbl.get(level, {})
                bands = [b for b in ("90%", "80%", "50%") if b in lv]
                if not bands:
                    continue
                fig2.add_trace(go.Scatter(
                    x=[lv[b]["nominal"] for b in bands], y=[lv[b]["empirical"] for b in bands],
                    mode="markers+text", text=bands, textposition="top center",
                    marker=dict(size=11, color=level_colors.get(level, "#00D2FF")), name=level,
                    hovertemplate=f"<b>{level}</b> — " + "%{text} band<br>Nominal: %{x:.0%}<br>"
                                  "Empirical: %{y:.0%}<extra></extra>",
                ))
                for b in bands:
                    level_rows.append({
                        "level": level, "band": b, "nominal": lv[b]["nominal"],
                        "empirical": round(lv[b]["empirical"], 3), "n_observations": lv["n_observations"],
                    })
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                xaxis_title="Nominal coverage", yaxis_title="Empirical")
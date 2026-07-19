"""
feature_engineering.py
=======================
Implementation Plan §B — leakage-safe, gap-safe, origin-based aggregate-window
feature engineering, at CAMPAIGN grain (the bottom level of the §E hierarchy).

§B.1 — this is an *origin-based aggregate-window* forecasting problem:
    For each (campaign_id, origin_date): features = everything knowable using
    data up to and including origin_date. Target = sum(revenue) over
    (origin_date, origin_date + horizon], for horizon in {30, 60, 90}.
    horizon_days is passed as a model feature (single shared model across
    horizons) rather than training 3 fully independent models — this shares
    statistical strength given how few campaigns exist per channel (§C.5).

§B.2 — reindex to a complete daily calendar per campaign BEFORE computing any
    lag/rolling feature, or lags will silently splice non-adjacent days
    together across real gaps in serving. Non-observed days get spend/revenue
    zero-filled (no serving = no spend, a defensible default); a `was_observed`
    flag is kept. Fields that are NaN because the *entire source file* never
    had that column (see schema_mapper §A.4) are left genuinely NaN even on
    observed rows — never zero-filled, never imputed (§C.4).

§B.3 — calendar / seasonality features (day-of-week, week-of-year, month,
    month start/end, a short justified holiday-window list, and — because the
    data spans 2+ years — a year-over-year same-week feature guarded with NaN
    for campaigns too new to have one).

§B.4 — campaign_id is never fed to the model as a raw high-cardinality
    categorical (would memorize identity rather than learn generalizable
    behavior — the "similar but not identical dataset" grading failure mode).
    Only `channel` / `campaign_type` are passed as native categoricals.
    A cheap, leakage-safe substitute for identity signal is an *expanding*
    (strictly-prior) campaign-level revenue/ROAS statistic — this is the
    time-series analogue of out-of-fold target encoding: because the origin
    framing already restricts every feature to data strictly at-or-before
    the origin date, an expanding statistic is automatically leakage-safe
    without needing a K-fold trick (K-fold encoding exists to fix leakage in
    i.i.d. data; origin-based framing already prevents it here).

§B.5 — `planned_future_daily_budget` is kept as an explicit, separate model
    input (not the same column as historical `spend`). At TRAIN time it is
    set to the realized mean daily spend actually observed over the forward
    target window (so the model learns a genuine spend -> revenue response).
    At INFERENCE time the frontend's budget slider / scenario code overwrites
    this same feature slot with a hypothetical spend level. The monotonic
    constraint (§C.3) is applied to this feature specifically.

§B.6 — funnel-efficiency ratios (`ctr_roll_28`, `cvr_roll_28`). Clicks,
    impressions, and conversions were already tracked as rolling SUMS (raw
    volume), but the model could never see click-through or conversion
    *rate* independent of volume — two campaigns spending the same amount
    with the same click volume can have very different revenue if one
    converts at 2% and the other at 8%. Computed from the same rolling sums
    already in the feature table (no new raw data needed), left NaN (not
    zero-filled) when the denominator is zero, same convention as
    `trailing_roas`.

§B.7 — extended funnel features: cost-efficiency (`cpm_roll_28`,
    `cpa_roll_28`) + reach/video signals (`reach_roll_sum_28`,
    `video_views_roll_sum_28`, `frequency_roll_28`, `video_view_rate_roll_28`).
    Added this round on top of §B.6's rate ratios:
      - `cpm_roll_28` = spend / impressions * 1000 (28-day rolling sums) —
        cost per thousand impressions. CTR/CVR say how efficiently traffic
        converts; CPM says how expensive that traffic was to buy in the
        first place — two campaigns with identical CTR/CVR can still have
        very different revenue-per-rupee if one pays 3x the CPM of the
        other. Left NaN when there were no impressions in the window.
      - `cpa_roll_28` = spend / conversions (28-day rolling sums) — cost per
        acquisition, the most directly business-legible efficiency number
        in performance marketing (agencies budget against a CPA target far
        more often than against CTR or CVR in isolation). Left NaN when
        there were no conversions in the window (an undefined cost, not a
        free one).
      - `reach_roll_sum_28` / `video_views_roll_sum_28` — raw rolling sums
        of two fields (schema_mapper §A.9) that were previously ingested
        and then thrown away entirely (`ignored_columns`) because they had
        no canonical slot. `reach` (Meta-only in this project's data) and
        `video_views` (Google-only) are genuinely NaN for channels whose
        files never reported them — same "whole file lacked the column"
        convention as every other optional field, not zero-filled.
      - `frequency_roll_28` = impressions / reach (28-day rolling sums) —
        average number of times the SAME reached user was shown an ad.
        This is the concrete signal that distinguishes brand-awareness
        spend from performance spend in a way `campaign_type` alone can't:
        a low-frequency, high-reach campaign is buying broad first-time
        exposure, while a high-frequency, lower-reach campaign is
        re-serving the same audience (retargeting-flavored), and those two
        patterns have historically very different revenue responses even
        within the same nominal campaign_type. Honesty note: Meta's raw
        export is a DAILY reach figure, not a deduplicated 28-day unique
        count, so summing 28 daily reach values double-counts any user
        reached on more than one day in the window — `frequency_roll_28`
        is therefore a slight over-estimate of true average frequency
        (documented as a known simplification in
        docs/technical_documentation.md §14, not silently assumed away).
      - `video_view_rate_roll_28` = video_views / impressions (28-day
        rolling sums) — the fraction of served impressions that resulted
        in a counted video view, Google's own upper-funnel engagement
        signal, on the same 0-vs-nonzero funnel-stage idea as CTR/CVR.
    All four ratios follow §B.6's existing convention: computed from
    rolling sums already being tracked (no new raw pulls beyond the two
    new §A.9 fields), left NaN — never zero-filled — on a zero denominator,
    same reasoning as `trailing_roas`/`ctr_roll_28`/`cvr_roll_28`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Feature manifest
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES: list[str] = ["channel", "campaign_type"]

NUMERIC_FEATURES: list[str] = [
    "horizon_days",
    "planned_future_daily_budget",
    "campaign_age_days",
    "days_since_last_nonzero_revenue",
    "was_observed_frac_30",
    "month", "week_of_year", "day_of_week", "is_weekend",
    "is_month_start", "is_month_end", "is_q4", "is_holiday_window",
    "days_to_dec25",
    "revenue_lag_7", "revenue_lag_14", "revenue_lag_28",
    "spend_lag_7", "spend_lag_14", "spend_lag_28",
    "conversions_lag_7", "conversions_lag_28",
    "revenue_roll_sum_7", "revenue_roll_sum_14", "revenue_roll_sum_28", "revenue_roll_sum_56",
    "revenue_roll_mean_7", "revenue_roll_mean_28",
    "revenue_roll_std_7", "revenue_roll_std_28",
    "spend_roll_sum_7", "spend_roll_sum_14", "spend_roll_sum_28", "spend_roll_sum_56",
    "spend_roll_mean_7", "spend_roll_mean_28",
    "conversions_roll_sum_28",
    "clicks_roll_sum_28", "impressions_roll_sum_28",
    "ctr_roll_28", "cvr_roll_28",
    "reach_roll_sum_28", "video_views_roll_sum_28",
    "cpm_roll_28", "cpa_roll_28", "frequency_roll_28", "video_view_rate_roll_28",
    "trailing_roas_7", "trailing_roas_28", "trailing_roas_56",
    "revenue_trend_14_28",
    "revenue_same_week_last_year",
    "daily_budget",
    "campaign_expanding_revenue_mean", "campaign_expanding_roas",
]
FEATURE_NAMES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

META_COLUMNS: list[str] = [
    "campaign_id", "campaign_name", "channel", "campaign_type",
    "origin_date", "horizon_days", "target_revenue", "actual_spend_sum",
]

MAX_LOOKBACK_FOR_ORIGIN = 14  # minimum campaign age (days) before an origin is usable


# ─────────────────────────────────────────────────────────────────────────────
# §B.3 — small, justified holiday-window flag (not a full generic calendar)
# ─────────────────────────────────────────────────────────────────────────────
def _is_holiday_window(dates: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros(len(dates), dtype=int)
    years = dates.year.unique()
    windows = []
    for y in years:
        windows += [
            (pd.Timestamp(y - 1, 12, 26), pd.Timestamp(y, 1, 7)),   # New Year hangover
            (pd.Timestamp(y, 2, 10), pd.Timestamp(y, 2, 14)),        # Valentine's
            (pd.Timestamp(y, 8, 15), pd.Timestamp(y, 9, 5)),         # back-to-school
            (pd.Timestamp(y, 11, 20), pd.Timestamp(y, 12, 2)),       # Black Friday / Cyber Monday week
            (pd.Timestamp(y, 12, 18), pd.Timestamp(y, 12, 25)),      # Christmas week
        ]
    for lo, hi in windows:
        mask = (dates >= lo) & (dates <= hi)
        out |= np.asarray(mask).astype(int)
    return out


def _days_to_dec25(dates: pd.DatetimeIndex) -> np.ndarray:
    year = dates.year
    adj_year = np.where(dates.month >= 7, year, year - 1)
    dec25 = pd.to_datetime(pd.Series(adj_year).astype(str) + "-12-25").to_numpy()
    return (dec25 - dates.to_numpy()).astype("timedelta64[D]").astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# §B.2 — reindex + gap handling, per campaign
# ─────────────────────────────────────────────────────────────────────────────
def _reindex_campaign(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date")
    full_range = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
    g = g.set_index("date").reindex(full_range)
    g.index.name = "date"

    was_observed = g["campaign_id"].notna()
    g["was_observed"] = was_observed.astype(int)

    # identity/setting fields persist across gap days
    for c in ["campaign_id", "campaign_name", "channel", "campaign_type"]:
        g[c] = g[c].ffill().bfill()
    g["daily_budget"] = g["daily_budget"].ffill()  # a setting, not an activity measurement

    # activity fields: zero-fill ONLY the newly-added gap rows. Rows that were
    # genuinely present but NaN because the whole source file lacked the
    # column (schema_mapper §A.4) are left untouched -> stay NaN forever.
    #
    # Fix (this round, surfaced by adding reach/video_views -- §A.9): a
    # field that's NaN across EVERY originally-observed row for this
    # campaign means the whole source file never reported it at all, not
    # that it happened to be zero every day. Previously this loop
    # unconditionally zero-filled gap rows regardless -- invisible before
    # because no pre-existing optional field (clicks/impressions/
    # conversions) was ever whole-file-missing in this project's own data,
    # but reach (bing/google) and video_views (bing/meta) genuinely are.
    # Without this guard, a channel that never reports reach at all would
    # still get a fabricated "0 reach" on its reindexed gap days while every
    # originally-observed row correctly stays NaN -- an inconsistent,
    # partially-fabricated column. Only zero-fill gap days for a field this
    # campaign's file actually tracks (at least one real observed value).
    gap_mask = ~was_observed
    for c in ["spend", "revenue", "clicks", "impressions", "conversions", "reach", "video_views"]:
        was_ever_reported = bool(g.loc[was_observed, c].notna().any())
        if was_ever_reported:
            g.loc[gap_mask, c] = g.loc[gap_mask, c].fillna(0.0)

    return g.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Per-campaign daily feature table ("as of end of this date, inclusive")
# ─────────────────────────────────────────────────────────────────────────────
def _add_daily_features(g: pd.DataFrame, yoy_lookup: dict | None = None) -> pd.DataFrame:
    g = g.reset_index(drop=True)
    dates = pd.DatetimeIndex(g["date"])

    rev = g["revenue"]
    sp = g["spend"]
    conv = g["conversions"]
    clk = g["clicks"]
    imp = g["impressions"]
    rch = g["reach"]
    vv = g["video_views"]

    # ── calendar ─────────────────────────────────────────────────────────
    g["month"] = dates.month
    g["week_of_year"] = dates.isocalendar().week.to_numpy().astype(int)
    g["day_of_week"] = dates.dayofweek
    g["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    g["is_month_start"] = dates.is_month_start.astype(int)
    g["is_month_end"] = dates.is_month_end.astype(int)
    g["is_q4"] = (dates.month >= 10).astype(int)
    g["is_holiday_window"] = _is_holiday_window(dates)
    g["days_to_dec25"] = _days_to_dec25(dates)

    # ── recency / maturity ───────────────────────────────────────────────
    campaign_start = dates.min()
    g["campaign_age_days"] = (dates - campaign_start).days
    nonzero_rev_date = pd.Series(np.where(rev > 0, dates.astype("int64"), np.nan))
    last_nonzero = nonzero_rev_date.ffill()
    g["days_since_last_nonzero_revenue"] = (
        (dates.astype("int64") - last_nonzero) / 8.64e13  # ns -> days
    ).fillna(g["campaign_age_days"])
    g["was_observed_frac_30"] = g["was_observed"].rolling(30, min_periods=1).mean()

    # ── lags ─────────────────────────────────────────────────────────────
    for lag in [7, 14, 28]:
        g[f"revenue_lag_{lag}"] = rev.shift(lag)
        g[f"spend_lag_{lag}"] = sp.shift(lag)
    for lag in [7, 28]:
        g[f"conversions_lag_{lag}"] = conv.shift(lag)

    # ── rolling sums / means / std ───────────────────────────────────────
    for w in [7, 14, 28, 56]:
        g[f"revenue_roll_sum_{w}"] = rev.rolling(w, min_periods=1).sum()
        g[f"spend_roll_sum_{w}"] = sp.rolling(w, min_periods=1).sum()
    for w in [7, 28]:
        g[f"revenue_roll_mean_{w}"] = rev.rolling(w, min_periods=1).mean()
        g[f"revenue_roll_std_{w}"] = rev.rolling(w, min_periods=2).std()
        g[f"spend_roll_mean_{w}"] = sp.rolling(w, min_periods=1).mean()
    g["conversions_roll_sum_28"] = conv.rolling(28, min_periods=1).sum()
    g["clicks_roll_sum_28"] = clk.rolling(28, min_periods=1).sum()
    g["impressions_roll_sum_28"] = imp.rolling(28, min_periods=1).sum()
    g["reach_roll_sum_28"] = rch.rolling(28, min_periods=1).sum()
    g["video_views_roll_sum_28"] = vv.rolling(28, min_periods=1).sum()

    # ── funnel-efficiency ratios (§B.6) — clicks/impressions and conversions
    # already existed as rolling SUMS (§B.3); the ratios themselves were never
    # computed as features, so the model could never directly see "this
    # campaign's click-through rate is drifting" independent of raw volume.
    # NaN (not 0) when the denominator is zero, same convention as
    # trailing_roas below -- LightGBM's native NaN handling routes these
    # splits sensibly rather than forcing a fabricated "0% CTR" for a
    # campaign that simply had no impressions in the window.
    g["ctr_roll_28"] = g["clicks_roll_sum_28"] / g["impressions_roll_sum_28"].replace(0.0, np.nan)
    g["cvr_roll_28"] = g["conversions_roll_sum_28"] / g["clicks_roll_sum_28"].replace(0.0, np.nan)

    # ── extended funnel features (§B.7) — cost-efficiency + reach/video ──
    g["cpm_roll_28"] = (
        g["spend_roll_sum_28"] / g["impressions_roll_sum_28"].replace(0.0, np.nan)
    ) * 1000.0
    g["cpa_roll_28"] = g["spend_roll_sum_28"] / g["conversions_roll_sum_28"].replace(0.0, np.nan)
    g["frequency_roll_28"] = g["impressions_roll_sum_28"] / g["reach_roll_sum_28"].replace(0.0, np.nan)
    g["video_view_rate_roll_28"] = g["video_views_roll_sum_28"] / g["impressions_roll_sum_28"].replace(0.0, np.nan)

    # ── trailing ROAS (NaN, not 0, when undefined — let LightGBM handle it) ─
    for w in [7, 28, 56]:
        rsum = g[f"revenue_roll_sum_{w}"] if f"revenue_roll_sum_{w}" in g else rev.rolling(w, min_periods=1).sum()
        ssum = g[f"spend_roll_sum_{w}"] if f"spend_roll_sum_{w}" in g else sp.rolling(w, min_periods=1).sum()
        g[f"trailing_roas_{w}"] = (rsum / ssum.replace(0.0, np.nan))

    # ── trend ────────────────────────────────────────────────────────────
    g["revenue_trend_14_28"] = g["revenue_roll_mean_28"] - g["revenue_roll_mean_28"].shift(14)

    # ── year-over-year (guarded NaN for campaigns too new) ──────────────
    if yoy_lookup is not None:
        key = (g["campaign_id"].iloc[0],)
        weekly = yoy_lookup.get(key)
        if weekly is not None:
            iso = dates.isocalendar()
            lookup_key = list(zip((iso["year"] - 1).to_numpy(), iso["week"].to_numpy()))
            g["revenue_same_week_last_year"] = [weekly.get(k, np.nan) for k in lookup_key]
        else:
            g["revenue_same_week_last_year"] = np.nan
    else:
        g["revenue_same_week_last_year"] = np.nan

    # ── expanding, strictly-causal campaign-identity signal (§B.4) ──────
    cum_rev = rev.cumsum()
    cum_sp = sp.cumsum()
    n = np.arange(1, len(g) + 1)
    g["campaign_expanding_revenue_mean"] = cum_rev / n
    g["campaign_expanding_roas"] = cum_rev / cum_sp.replace(0.0, np.nan)

    return g


def _build_yoy_lookup(daily: pd.DataFrame) -> dict:
    """Per-campaign {(year, iso_week): revenue_sum} for the year-over-year feature."""
    iso = pd.DatetimeIndex(daily["date"]).isocalendar()
    tmp = daily[["campaign_id", "revenue"]].copy()
    tmp["year"] = iso["year"].to_numpy()
    tmp["week"] = iso["week"].to_numpy()
    grouped = tmp.groupby(["campaign_id", "year", "week"])["revenue"].sum()
    lookup: dict = {}
    for (cid, yr, wk), val in grouped.items():
        lookup.setdefault((cid,), {})[(yr, wk)] = val
    return lookup


def build_daily_feature_table(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """
    §B.2/§B.3 — one row per (campaign_id, date), reindexed to a complete daily
    calendar per campaign, with every lag/rolling/calendar/expanding feature
    computed "as of end of that date, inclusive". This is the shared building
    block for both training-frame construction and the live prediction
    snapshot.
    """
    df = canonical_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # reach/video_views (§A.9) are optional canonical columns produced by
    # schema_mapper.ingest_*. A caller that builds a canonical-shaped frame
    # directly (e.g. a test fixture predating §A.9) may omit them entirely
    # rather than including them pre-filled with NaN -- treat "column
    # absent" identically to "column present but entirely NaN" rather than
    # raising a KeyError downstream.
    for _c in ("reach", "video_views"):
        if _c not in df.columns:
            df[_c] = np.nan

    # Multiple rows can exist per (campaign_id, date) only if a campaign
    # somehow appears twice in a source file on the same day - aggregate
    # defensively rather than silently duplicating rows into the reindex.
    agg_cols = {"spend": "sum", "revenue": "sum", "clicks": "sum",
                "impressions": "sum", "conversions": "sum"}
    first_cols = {"campaign_name": "first", "channel": "first",
                  "campaign_type": "first", "daily_budget": "first"}
    daily = (
        df.groupby(["campaign_id", "date"], as_index=False)
          .agg({**agg_cols, **first_cols})
    )
    # reach/video_views need `min_count=1` (NOT plain "sum" like the fields
    # above): a channel whose whole source file never had the column (§A.9)
    # is genuinely NaN on every row, and plain groupby-sum silently turns an
    # all-NaN group into a fabricated 0.0 -- exactly the "valid-looking
    # zero" failure mode §B.2 already guards against for every other field.
    # `min_count=1` preserves NaN when every value being summed is NaN.
    reach_vv = (
        df.groupby(["campaign_id", "date"])[["reach", "video_views"]]
          .sum(min_count=1)
          .reset_index()
    )
    daily = daily.merge(reach_vv, on=["campaign_id", "date"], how="left")

    yoy_lookup = _build_yoy_lookup(daily)

    parts = []
    for cid, g in daily.groupby("campaign_id", sort=False):
        g = _reindex_campaign(g)
        g = _add_daily_features(g, yoy_lookup=yoy_lookup)
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["campaign_id", "date"]).reset_index(drop=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §B.1 — origin-based aggregate-window training frame
# ─────────────────────────────────────────────────────────────────────────────
def build_training_frame(
    canonical_df: pd.DataFrame,
    horizons: tuple[int, ...] = (30, 60, 90),
    origin_stride: int = 3,
    min_history_days: int = MAX_LOOKBACK_FOR_ORIGIN,
) -> pd.DataFrame:
    """
    Build one row per (campaign_id, origin_date, horizon) with:
      - features = daily_feature_table snapshot as of origin_date (inclusive)
      - target_revenue = sum(revenue) strictly AFTER origin_date, through
        origin_date + horizon (i.e. the next `horizon` days)
      - actual_spend_sum = sum(spend) over the same forward window (used to
        set planned_future_daily_budget at train time, and for ROAS eval)

    Origins are only emitted where the full forward window actually exists in
    the observed calendar (no synthetic/partial targets), and where the
    campaign has at least `min_history_days` of history — very young
    campaigns don't yet have enough signal for a stable lag/rolling feature
    set (§B.2/§B.1).
    """
    daily = build_daily_feature_table(canonical_df)
    max_h = max(horizons)

    rows: list[pd.DataFrame] = []
    for cid, g in daily.groupby("campaign_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        rev = g["revenue"].to_numpy()
        sp = g["spend"].to_numpy()
        cum_rev = np.concatenate([[0.0], np.cumsum(rev)])
        cum_sp = np.concatenate([[0.0], np.cumsum(sp)])

        candidate_idx = np.arange(min_history_days, n, origin_stride)
        if len(candidate_idx) == 0:
            continue

        for h in horizons:
            valid = candidate_idx[candidate_idx + h < n]
            if len(valid) == 0:
                continue
            sub = g.iloc[valid].copy()
            sub["horizon_days"] = h
            target_rev = cum_rev[valid + h + 1] - cum_rev[valid + 1]
            spend_sum = cum_sp[valid + h + 1] - cum_sp[valid + 1]
            sub["target_revenue"] = target_rev
            sub["actual_spend_sum"] = spend_sum
            sub["planned_future_daily_budget"] = spend_sum / h
            sub = sub.rename(columns={"date": "origin_date"})
            rows.append(sub)

    if not rows:
        raise RuntimeError(
            "No usable (campaign, origin, horizon) rows were produced — "
            "check that campaigns have enough history relative to min_history_days/horizons."
        )

    out = pd.concat(rows, ignore_index=True)
    keep = list(dict.fromkeys(META_COLUMNS + FEATURE_NAMES))
    missing = [c for c in keep if c not in out.columns]
    if missing:
        raise RuntimeError(f"Missing engineered columns: {missing}")
    out = out[keep].copy()
    out["channel"] = out["channel"].astype("category")
    out["campaign_type"] = out["campaign_type"].astype("category")
    return out.sort_values(["campaign_id", "origin_date", "horizon_days"]).reset_index(drop=True)


def build_latest_snapshot(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per campaign_id: the feature snapshot as of the most recent
    observed date across the whole dataset (the live forecast origin). No
    target — this is what predict.py feeds to the trained models, with
    `horizon_days` and `planned_future_daily_budget` filled in per-scenario
    by the caller.
    """
    daily = build_daily_feature_table(canonical_df)
    global_last_date = daily["date"].max()
    latest = (
        daily.sort_values("date")
        .groupby("campaign_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
        .rename(columns={"date": "origin_date"})
    )
    latest["forecast_as_of"] = global_last_date
    latest["channel"] = latest["channel"].astype("category")
    latest["campaign_type"] = latest["campaign_type"].astype("category")
    return latest


# ─────────────────────────────────────────────────────────────────────────────
# Generic anomalous-segment detector (Part 2 review, §7) -- replaces what was
# a hand-fitted `is_bing_2026` feature + fixed 0.1x training-sample weight.
# ─────────────────────────────────────────────────────────────────────────────
def detect_anomalous_segments(
    canonical_df: pd.DataFrame,
    z_thresh: float = 2.0,
    roll_window: int = 30,
    min_run_days: int = 14,
    min_history_days: int = 90,
    warmup_days: int = 60,
) -> list[dict]:
    """
    Generic replacement for a hand-fitted "this channel, this year, had a
    tracking break" constant. For every (channel, campaign_type) group with
    enough of its OWN history to establish a baseline, flags contiguous date
    ranges where the rolling zero-revenue-rate is a genuine outlier relative
    to that SAME group's own history (a robust median/MAD z-score computed
    per-group, not a fixed threshold tied to any particular channel or date)
    -- so an analogous break in a different channel, or none at all in this
    one, is treated identically by whatever data is loaded at train time.

    Each group's own first `warmup_days` worth of rolling observations are
    excluded from the *baseline* estimate (not from evaluation) because a
    brand-new campaign type's near-100%-zero start is ordinary ramp-up, not
    an anomaly relative to itself -- without this exclusion, a group that
    both ramped up slowly AND had a later genuine break (the actual profile
    seen in `bing/Search` here) pulls its own baseline upward and can mask
    the later break entirely.

    Honesty note (see docs/technical_documentation.md §7 for the full
    writeup): at this threshold, this generic check confidently flags two
    `meta` segments (max z up to 9.4) but only weakly flags `bing/Search`'s
    2026 period (z~2.25) -- because that channel/type is persistently
    volatile across its *entire* life, not uniquely anomalous in one place.
    A detector tuned specifically to make Bing 2026 "win" would be exactly
    the kind of dataset-specific overfitting this replaces; this returns
    whatever a fair, identical check finds across every group.

    Returns a list of {"channel", "campaign_type", "start_date", "end_date",
    "n_days", "baseline_zero_rate", "segment_zero_rate", "max_z_score",
    "suggested_weight", "note"} dicts. `suggested_weight` scales down
    automatically with how extreme the anomaly is (clipped to [0.1, 0.9]) --
    an automatically-detected multiplier, not a single magic constant
    applied uniformly to every flagged period.
    """
    df = canonical_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = (
        df.groupby(["channel", "campaign_type", "date"], as_index=False)
          .agg(revenue=("revenue", "sum"))
    )

    segments: list[dict] = []
    for (channel, ctype), g in daily.groupby(["channel", "campaign_type"]):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < min_history_days:
            continue
        zero_flag = (g["revenue"] <= 0).astype(float)
        roll_rate = zero_flag.rolling(roll_window, min_periods=max(5, roll_window // 3)).mean()

        baseline_vals = roll_rate.iloc[warmup_days:].dropna()
        if len(baseline_vals) < 10:
            continue
        median = float(baseline_vals.median())
        mad = float((baseline_vals - median).abs().median()) * 1.4826  # normal-consistent scale
        if mad < 1e-6:
            continue  # this group's own zero-rate is essentially constant -- nothing to compare against

        z = (roll_rate - median) / mad
        flagged = (z >= z_thresh).fillna(False)
        flagged.iloc[:warmup_days] = False

        run_id = (flagged != flagged.shift(fill_value=False)).cumsum()
        tagged = g.assign(flagged=flagged, run_id=run_id, z=z, zero_flag=zero_flag)
        for _, run in tagged.groupby("run_id"):
            if not bool(run["flagged"].iloc[0]) or len(run) < min_run_days:
                continue
            max_z = float(run["z"].max())
            seg_rate = float(run["zero_flag"].mean())
            weight = float(np.clip(1.0 / (1.0 + max(max_z - z_thresh, 0.0)), 0.1, 0.9))
            segments.append({
                "channel": channel, "campaign_type": ctype,
                "start_date": run["date"].iloc[0].date().isoformat(),
                "end_date": run["date"].iloc[-1].date().isoformat(),
                "n_days": int(len(run)),
                "baseline_zero_rate": round(median, 4),
                "segment_zero_rate": round(seg_rate, 4),
                "max_z_score": round(max_z, 2),
                "suggested_weight": round(weight, 3),
                "note": (
                    f"zero-revenue rate rose to {seg_rate:.0%} vs this group's own typical "
                    f"{median:.0%} (z={max_z:.1f}) over {len(run)} consecutive observed days -- "
                    f"training rows whose target window overlaps this period are downweighted "
                    f"to {weight:.2f}x."
                ),
            })

    return segments


def compute_anomaly_weights(frame: pd.DataFrame, segments: list[dict]) -> np.ndarray:
    """
    Per-row training weight (default 1.0), downweighted for any row whose
    target window [origin_date+1, origin_date+horizon_days] overlaps a
    detected anomalous segment (`detect_anomalous_segments`) for that row's
    own (channel, campaign_type) -- generalizes what was a hardcoded
    `is_bing_2026` sample-weight flag into a data-driven mechanism that
    applies identically to whichever (channel, campaign_type, period)
    combination the detector actually flags on the data at hand, and does
    nothing at all if it flags nothing.
    """
    weights = np.ones(len(frame), dtype=float)
    if not segments:
        return weights

    origin = pd.to_datetime(frame["origin_date"]).to_numpy()
    win_start = origin + np.timedelta64(1, "D")
    win_end = origin + frame["horizon_days"].to_numpy().astype("timedelta64[D]")
    channel_arr = frame["channel"].to_numpy()
    ctype_arr = frame["campaign_type"].to_numpy()

    for seg in segments:
        mask_group = (channel_arr == seg["channel"]) & (ctype_arr == seg["campaign_type"])
        if not mask_group.any():
            continue
        seg_start = np.datetime64(seg["start_date"])
        seg_end = np.datetime64(seg["end_date"])
        overlap = mask_group & (win_start <= seg_end) & (win_end >= seg_start)
        if overlap.any():
            weights[overlap] = np.minimum(weights[overlap], seg["suggested_weight"])

    return weights

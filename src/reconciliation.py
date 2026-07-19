"""
reconciliation.py
==================
Implementation Plan §E — campaign -> campaign_type -> channel -> total,
reconciled with Nixtla's `hierarchicalforecast` (Wickramasuriya et al. 2019,
"Optimal Forecast Reconciliation ... Through Trace Minimization").

Design shipped:
  - Hierarchy kept shallow per §E.2: channel -> channel/campaign_type ->
    channel/campaign_type/campaign_id (campaign_type namespaced under channel,
    since the same type label, e.g. "search", recurs across channels and a
    hierarchy must be a strict tree).
  - Base forecasts: bottom level = the pooled model (§C); upper levels = the
    library's own S-matrix aggregation of the bottom level (§E.1's "simple
    aggregation for the upper levels").
  - Point reconciliation: `BottomUp` (trivial coherence sanity check) and
    `MinTrace(method="mint_shrink")` (primary method) using genuine
    out-of-fold walk-forward residuals (§C.6) for the shrinkage covariance —
    not in-sample-fitted residuals, so the covariance estimate isn't
    artificially optimistic. Verified coherent (max |error| = 0.0) on the
    reconciled median.

  - Interval (probabilistic) reconciliation, §E.1 — genuinely two-tier, for
    a real, MEASURED reason rather than a uniform shortcut:

      hierarchicalforecast's `Conformal` reconciler scores the RECONCILED
      calibration forecast, `S @ P @ y_hat_cal`, which needs every bottom
      node's calibration value jointly present at each shared calibration
      timestamp (Principato et al. 2024). Checked directly against this
      project's own walk-forward OOF calibration data
      (`bundle["oof_by_horizon"]`): at the channel level (3 channels + 1
      total), EVERY one of 308 calibration dates has all series present —
      full joint alignment holds, so genuine `hierarchicalforecast.methods.
      Conformal` is used there, unmodified from the library. At the
      individual-campaign level, ZERO of 308 dates have all ~64+ campaigns
      present simultaneously (campaigns start and stop at different times —
      a real property of this data, not a bug) — joint reconciled conformal
      is mathematically inapplicable there without fabricating
      observations. For campaign_type and campaign nodes, a MARGINAL
      (per-node, independent) split-conformal correction is used instead:
      same nonconformity-score idea (signed residual quantiles, matching
      `Conformal.get_prediction_quantiles`'s own math), computed
      independently per node from that node's own naive-aggregated OOF
      history — no cross-node alignment required, and centered so the
      already-coherent MinTrace median is left exactly unchanged (only the
      band around it comes from real empirical residuals). Falls back to
      the original documented shortcut (rescale each quantile by the
      point-forecast's MinTrace adjustment ratio) only for the handful of
      nodes with too little history (<15 OOF observations) for a
      meaningful empirical quantile — 61/64 campaigns clear that bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hierarchicalforecast.utils import aggregate
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace

from modeling import QUANTILES, fix_quantile_crossing, reliability_diagram

HIERARCHY_SPEC = [
    ["total"], ["total", "channel"], ["total", "channel", "campaign_type"],
    ["total", "channel", "campaign_type", "campaign_id"],
]
HIERARCHY_SPEC_CHANNEL_ONLY = [["total"], ["total", "channel"]]
_MODEL_COL = "model"
MIN_OBS_FOR_MARGINAL_CONFORMAL = 15

# maps a confidence `level` (as used by hierarchicalforecast) to the (lo, hi)
# quantile pair it produces -- chosen so the union covers this project's own
# 7-quantile grid exactly (0.05/0.10/0.25/*point*/0.75/0.90/0.95).
_CONFORMAL_LEVELS = [90, 80, 50]


def build_calibration_long(oof_df: pd.DataFrame) -> pd.DataFrame:
    """
    `oof_df` = out-of-fold walk-forward predictions for ONE horizon, with
    columns: channel, campaign_type, campaign_id, origin_date, target_revenue
    (actual), pred_median (model). Returns the long frame `aggregate()` needs.
    An explicit constant `total` column gives the hierarchy a genuine
    grand-total root node (§E's "plus an aggregate blended figure").
    """
    out = oof_df.rename(columns={"origin_date": "ds", "target_revenue": "y", "pred_median": _MODEL_COL})
    out = out.copy()
    out["total"] = "total"
    return out[["total", "channel", "campaign_type", "campaign_id", "ds", "y", _MODEL_COL]].dropna()


def _aggregate_hierarchy(hist_long: pd.DataFrame, hierarchy_spec: list[list[str]] = HIERARCHY_SPEC):
    cols = hierarchy_spec[-1]
    Y_actual, S_df, tags = aggregate(hist_long[cols + ["ds", "y"]], hierarchy_spec)
    model_long = hist_long[cols + ["ds"]].copy()
    model_long["y"] = hist_long[_MODEL_COL].to_numpy()
    Y_model, _, _ = aggregate(model_long, hierarchy_spec)
    Y_full = Y_actual.merge(Y_model.rename(columns={"y": _MODEL_COL}), on=["unique_id", "ds"], how="inner")
    return Y_full, S_df, tags


def _channel_level_conformal_intervals(
    hist_long: pd.DataFrame, channel_bottom_median: dict[str, float],
    quantiles: list[float], forecast_ds: pd.Timestamp,
) -> pd.DataFrame | None:
    """§E.1, genuine tier: real `hierarchicalforecast.methods.Conformal`,
    unmodified, on the total+channel sub-hierarchy where every calibration
    date has all series present (verified 308/308 on this project's own OOF
    data — see module docstring). `channel_bottom_median` maps each raw
    channel name (e.g. "bing") to its naive summed median forecast.

    Returns a DataFrame indexed by unique_id ("total", "total/bing", ...)
    with one column per quantile in `quantiles`, or None if the level-90/
    80/50 grid doesn't cover every requested quantile (defensive; it always
    should for this project's fixed QUANTILES).
    """
    level_to_qpair = {90: (0.05, 0.95), 80: (0.10, 0.90), 50: (0.25, 0.75)}
    needed = set(quantiles) - {0.5}
    covered = {q for pair in level_to_qpair.values() for q in pair}
    if not needed.issubset(covered):
        return None  # defensive fallback if QUANTILES is ever changed upstream

    hist_ch = hist_long.copy()
    hist_ch["channel"] = hist_ch["channel"].astype(str)
    Y_full, S_df, tags = _aggregate_hierarchy(hist_ch, HIERARCHY_SPEC_CHANNEL_ONLY)
    bottom_cols = [c for c in S_df.columns if c != "unique_id"]  # e.g. "total/bing"

    S = S_df[bottom_cols].to_numpy()
    bottom_median = np.array([channel_bottom_median.get(c.split("/")[-1], 0.0) for c in bottom_cols])
    all_median = S @ bottom_median
    Y_hat = pd.DataFrame({"unique_id": S_df["unique_id"].to_numpy(), "ds": forecast_ds, _MODEL_COL: all_median})

    hr = HierarchicalReconciliation(reconcilers=[MinTrace(method="mint_shrink")])
    recon = hr.reconcile(
        Y_hat_df=Y_hat, S_df=S_df, tags=tags, Y_df=Y_full,
        level=_CONFORMAL_LEVELS, intervals_method="conformal",
    ).set_index("unique_id")

    pt_col = [c for c in recon.columns
              if c.startswith(f"{_MODEL_COL}/MinTrace") and "-lo-" not in c and "-hi-" not in c][0]
    out = pd.DataFrame(index=recon.index)
    out["q0.5"] = recon[pt_col]
    for level, (lo_q, hi_q) in level_to_qpair.items():
        lo_col = f"{pt_col}-lo-{level}"
        hi_col = f"{pt_col}-hi-{level}"
        if lo_q in quantiles:
            out[f"q{lo_q}"] = recon[lo_col]
        if hi_q in quantiles:
            out[f"q{hi_q}"] = recon[hi_col]
    return out


def _marginal_conformal_offsets(Y_full: pd.DataFrame, unique_id: str, quantiles: list[float],
                                 min_obs: int = MIN_OBS_FOR_MARGINAL_CONFORMAL) -> dict[float, float] | None:
    """§E.1, marginal tier: independent per-node split-conformal correction.
    Same nonconformity score `hierarchicalforecast.methods.Conformal` uses
    (signed residual = actual - naive aggregate prediction), computed from
    THIS node's own OOF history only (no cross-node alignment needed).
    Centered on the node's own median residual so it composes as a pure
    "shape around the median" correction — the already-coherent MinTrace
    median is left untouched; only the band comes from this. Returns None
    (caller keeps the existing rescale-fallback) if there's too little
    history for a meaningful empirical quantile.
    """
    node_hist = Y_full.loc[Y_full["unique_id"] == unique_id]
    if len(node_hist) < min_obs:
        return None
    scores = (node_hist["y"] - node_hist[_MODEL_COL]).to_numpy()
    median_score = float(np.median(scores))
    return {q: float(np.quantile(scores, q) - median_score) for q in quantiles}


def _fix_crossing_pin_median(vals: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Both §E.1 tiers construct each node's band as `reconciled_median +
    offset`, specifically so the already coherence-checked median is never
    touched. Plain `modeling.fix_quantile_crossing` (a per-row `np.sort`)
    doesn't preserve WHICH value sits at the median position — clipping a
    negative lower quantile to 0 can occasionally push it above a genuinely
    tiny positive median, and a sort would then swap them, silently
    breaking the "band never moves the median" invariant for small/
    near-zero nodes. This clips and enforces monotonicity by walking
    outward from the median in both directions instead, so `vals` at the
    median's index is mathematically guaranteed unchanged.
    """
    vals = np.clip(vals, 0, None).astype(float)
    mid = quantiles.index(0.5)
    for i in range(mid - 1, -1, -1):
        vals[i] = min(vals[i], vals[i + 1])
    for i in range(mid + 1, len(vals)):
        vals[i] = max(vals[i], vals[i - 1])
    return vals


def reconcile_forecast(
    hist_long: pd.DataFrame,
    live_bottom: pd.DataFrame,
    quantiles: list[float] = QUANTILES,
    forecast_ds: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    `live_bottom`: one row per campaign_id with columns channel, campaign_type,
    campaign_id, one column per quantile in `quantiles` (already sorted,
    already clamped >= 0), and `spend` (assumed total spend for this
    campaign over the horizon — a fixed scenario input, not itself a
    probabilistic forecast) for a single horizon.

    Returns (reconciled_df, diagnostics) where reconciled_df has one row per
    hierarchy node (campaign, campaign_type, channel, total) with reconciled
    quantile columns, a `spend` column (simple bottom-up sum of the same
    assumed spend up the identical hierarchy — no conformal correction
    needed, since spend here is an assumed planning input, not uncertain),
    and `roas_q{q}`/`roas_reconciled_median` columns (each revenue quantile
    divided by that node's own aggregated spend) — this is what gives
    channel/campaign_type/total-level ROAS *ranges*, not just the existing
    per-campaign point `roas_p50` (revenue_p50 / spend) computed elsewhere.
    Dividing every quantile by the SAME (non-random) spend value preserves
    quantile ordering automatically, so no extra crossing-fix is needed for
    the ROAS columns themselves. Diagnostics reports the coherence error.
    """
    if forecast_ds is None:
        forecast_ds = pd.Timestamp("2099-01-01")  # any fixed sentinel; only ever used internally for one ds

    Y_full, S_df, tags = _aggregate_hierarchy(hist_long)

    bottom_cols = [c for c in S_df.columns if c != "unique_id"]
    live_bottom = live_bottom.copy()
    live_bottom["node_id"] = (
        "total/" + live_bottom["channel"].astype(str) + "/" + live_bottom["campaign_type"].astype(str)
        + "/" + live_bottom["campaign_id"].astype(str)
    )
    live_by_node = live_bottom.set_index("node_id")

    has_spend = "spend" in live_by_node.columns
    missing = [c for c in bottom_cols if c not in live_by_node.index]
    if missing:
        # A bottom node exists in reconciliation history but has no live
        # forecast this run (e.g. the campaign went inactive) - point mass 0,
        # it simply won't contribute upward.
        pad_cols = [f"q{q}" for q in quantiles] + (["spend"] if has_spend else [])
        pad = pd.DataFrame(0.0, index=missing, columns=pad_cols)
        live_by_node = pd.concat([live_by_node, pad])

    q_cols = [f"q{q}" for q in quantiles]
    bottom_matrix = live_by_node.loc[bottom_cols, q_cols].to_numpy()
    median_idx = quantiles.index(0.5)
    bottom_median = bottom_matrix[:, median_idx]

    S = S_df[bottom_cols].to_numpy()
    all_median = S @ bottom_median
    all_quantiles_naive = S @ bottom_matrix  # naive sum-of-quantiles baseline per node, pre-reconciliation

    Y_hat = pd.DataFrame({
        "unique_id": S_df["unique_id"].to_numpy(),
        "ds": forecast_ds,
        _MODEL_COL: all_median,
    })

    hr = HierarchicalReconciliation(reconcilers=[BottomUp(), MinTrace(method="mint_shrink")])
    recon = hr.reconcile(Y_hat_df=Y_hat, S_df=S_df, tags=tags, Y_df=Y_full)

    mintrace_col = [c for c in recon.columns if c.startswith(f"{_MODEL_COL}/MinTrace")][0]
    bottomup_col = f"{_MODEL_COL}/BottomUp"

    recon = recon.set_index("unique_id")
    reconciled_median = recon[mintrace_col]
    naive_median = pd.Series(all_median, index=S_df["unique_id"].to_numpy())
    adj_ratio = (reconciled_median / naive_median.replace(0, np.nan)).fillna(1.0).clip(0.2, 5.0)

    naive_q_df = pd.DataFrame(all_quantiles_naive, index=S_df["unique_id"].to_numpy(), columns=q_cols)
    reconciled_q = naive_q_df.mul(adj_ratio, axis=0)
    reconciled_q = pd.DataFrame(
        fix_quantile_crossing(reconciled_q.to_numpy()), index=reconciled_q.index, columns=q_cols,
    )
    reconciled_q["reconciled_median"] = reconciled_median.reindex(reconciled_q.index)
    reconciled_q["bottomup_median"] = recon[bottomup_col].reindex(reconciled_q.index)
    _LEVEL_BY_SLASHES = {0: "total", 1: "channel", 2: "campaign_type", 3: "campaign"}
    reconciled_q["level"] = [_LEVEL_BY_SLASHES.get(uid.count("/"), "campaign") for uid in reconciled_q.index]

    # Spend, aggregated up the SAME hierarchy with the SAME S matrix as a
    # plain bottom-up sum -- assumed spend is a fixed scenario input here,
    # not itself a probabilistic forecast, so it needs no MinTrace/conformal
    # treatment, just the aggregation. This is what makes ROAS *ranges* (not
    # just a single derived point) possible at every node, not only
    # per-campaign: each revenue quantile at a node divided by that SAME
    # node's spend, computed once at the end of this function after both
    # §E.1 tiers below have finished adjusting the quantile bands.
    if has_spend:
        bottom_spend = live_by_node.loc[bottom_cols, "spend"].to_numpy(dtype=float)
        reconciled_q["spend"] = S @ bottom_spend

    # ─────────────────────────────────────────────────────────────────────
    # §E.1 — probabilistic reconciliation upgrade (two-tier; see module
    # docstring for why). Both tiers only ever replace the BAND (the q_cols
    # values) around each node's median — `reconciled_median` itself (the
    # coherence-checked MinTrace point estimate, unchanged since §E's
    # original ship) is never touched, so the coherence diagnostic below
    # stays exactly as before regardless of which tier a node's band came
    # from.
    # ─────────────────────────────────────────────────────────────────────
    prob_diag = {"genuine_joint_conformal_nodes": [], "marginal_conformal_nodes": [],
                 "rescale_fallback_nodes": []}
    try:
        ch_str = live_bottom["channel"].astype(str)
        all_channels = sorted(set(hist_long["channel"].astype(str).unique()) | set(ch_str.unique()))
        channel_bottom_median = (
            live_bottom.assign(_ch=ch_str).groupby("_ch")["q0.5"].sum()
            .reindex(all_channels, fill_value=0.0).to_dict()
        )
        ch_conformal = _channel_level_conformal_intervals(hist_long, channel_bottom_median, quantiles, forecast_ds)
    except Exception as exc:
        ch_conformal = None
        prob_diag["genuine_joint_conformal_error"] = repr(exc)

    handled_by_tier1 = set()
    if ch_conformal is not None:
        for uid in ch_conformal.index:
            if uid not in reconciled_q.index:
                continue
            band_offset = (ch_conformal.loc[uid, q_cols] - ch_conformal.loc[uid, "q0.5"]).to_numpy(dtype=float)
            vals = reconciled_q.loc[uid, "reconciled_median"] + band_offset
            vals = _fix_crossing_pin_median(vals, quantiles)
            reconciled_q.loc[uid, q_cols] = vals
            prob_diag["genuine_joint_conformal_nodes"].append(uid)
            handled_by_tier1.add(uid)

    for uid in reconciled_q.index:
        if uid in handled_by_tier1 or reconciled_q.loc[uid, "level"] not in ("campaign_type", "campaign"):
            continue
        offsets = _marginal_conformal_offsets(Y_full, uid, quantiles)
        if offsets is None:
            prob_diag["rescale_fallback_nodes"].append(uid)
            continue
        median_val = reconciled_q.loc[uid, "reconciled_median"]
        vals = np.array([median_val + offsets[q] for q in quantiles])
        vals = _fix_crossing_pin_median(vals, quantiles)
        reconciled_q.loc[uid, q_cols] = vals
        prob_diag["marginal_conformal_nodes"].append(uid)

    prob_diag["n_genuine_joint_conformal"] = len(prob_diag["genuine_joint_conformal_nodes"])
    prob_diag["n_marginal_conformal"] = len(prob_diag["marginal_conformal_nodes"])
    prob_diag["n_rescale_fallback"] = len(prob_diag["rescale_fallback_nodes"])

    # ROAS range columns, derived from the FINAL (post-tier) revenue
    # quantiles above divided by this node's own spend. Dividing every
    # quantile by the same positive constant preserves their order, so no
    # extra quantile-crossing fix is needed here — but do guard spend == 0
    # (an inactive node) rather than producing inf.
    if has_spend:
        spend_safe = reconciled_q["spend"].replace(0.0, np.nan)
        for col in q_cols:
            reconciled_q[f"roas_{col}"] = reconciled_q[col] / spend_safe
        reconciled_q["roas_reconciled_median"] = reconciled_q["reconciled_median"] / spend_safe

    coherence_err = float(np.max(np.abs(
        S @ reconciled_q.loc[bottom_cols, "reconciled_median"].to_numpy() -
        reconciled_q["reconciled_median"].to_numpy()
    )))
    diagnostics = {"max_abs_coherence_error": coherence_err, "n_nodes": len(reconciled_q),
                   "probabilistic_reconciliation": prob_diag}
    return reconciled_q.reset_index().rename(columns={"index": "unique_id"}), diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# §E re-audit — is the reconciled BAND still calibrated at every level?
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_reconciled_calibration(
    holdout_df: pd.DataFrame,
    oof_by_horizon: dict,
    quantiles: list[float] = QUANTILES,
    max_snapshots_per_horizon: int | None = None,
) -> dict:
    """
    Coherence and calibration are separate properties (Principato, Stoltz,
    Amara-Ouali, Goude, Hamrouche & Poggi 2024, already cited for §E.1's
    genuine-conformal tier): `max_abs_coherence_error == 0.000000` (verified
    above, every run) says the reconciled MEDIAN sums correctly up the
    hierarchy. It says nothing about whether the reconciled quantile BAND
    around that median is still as well-calibrated at the channel /
    campaign_type / total level as the base per-campaign CQR band was before
    reconciliation (§6's reliability diagram has only ever been run at the
    base, bottom-level rows). That was a genuinely open question for this
    project until now.

    Pure evaluation — no model changes, no new fitting. Reuses
    `reconcile_forecast` exactly as production calls it (same function, same
    two-tier interval logic, same OOF history) for every holdout
    origin_date/horizon snapshot, then reuses `modeling.reliability_diagram`
    exactly as §6/§7 already do, just pooling its inputs from four different
    levels of the SAME reconciled output instead of one.

    Parameters
    ----------
    holdout_df : one row per (campaign_id, origin_date, horizon_days) on the
        final holdout split, with columns channel, campaign_type,
        campaign_id, origin_date, horizon_days, target_revenue (actual), and
        one column per quantile in `quantiles` named f"q{quantile}" —
        exactly `q_hold_cqr` (§7's final-holdout CQR-calibrated quantiles)
        with its identifying columns attached, nothing recomputed.
    oof_by_horizon : the SAME `{horizon: oof_df}` mapping already stored in
        the model bundle and already used by `reconcile_forecast` at
        inference time (§E prep) — this checks the actual production
        reconciliation path, not a separate one built only for this audit.
    max_snapshots_per_horizon : None (default) processes every distinct
        origin_date in the holdout, exactly as the shipped default always
        has -- this parameter changes nothing about that behavior unless
        explicitly set. When set, evenly subsamples that many origin_dates
        per horizon (spread across the full holdout date range, not just
        the first N) purely for fast dev-time iteration on a slower machine
        -- a performance knob, not a change to what's being measured.

    Returns
    -------
    {"total": {...}, "channel": {...}, "campaign_type": {...},
     "campaign": {...}, "n_snapshots": int, "n_snapshots_skipped": int}
    where each level's dict is `modeling.reliability_diagram`'s own output
    (nominal vs. empirical coverage at the 90%/80%/50% bands) computed by
    pooling that level's reconciled quantiles across every holdout
    origin_date/horizon snapshot, plus "n_observations" (how many
    level-rows were pooled into that count — e.g. one per channel per
    snapshot at the channel level).
    """
    q_cols = [f"q{q}" for q in quantiles]
    levels = ("total", "channel", "campaign_type", "campaign")
    y_by_level: dict[str, list] = {lv: [] for lv in levels}
    q_by_level: dict[str, list] = {lv: [] for lv in levels}
    n_snapshots = 0
    n_skipped = 0

    for h, oof_h in (oof_by_horizon or {}).items():
        if oof_h is None or len(oof_h) == 0:
            continue
        hist_long = build_calibration_long(oof_h)
        hold_h = holdout_df.loc[holdout_df["horizon_days"] == h]
        if hold_h.empty:
            continue

        origin_dates = sorted(hold_h["origin_date"].unique())
        if max_snapshots_per_horizon is not None and len(origin_dates) > max_snapshots_per_horizon:
            idx = np.linspace(0, len(origin_dates) - 1, max_snapshots_per_horizon).round().astype(int)
            origin_dates = [origin_dates[i] for i in sorted(set(idx))]

        for origin_date in origin_dates:
            snap = hold_h.loc[hold_h["origin_date"] == origin_date]
            live_bottom = snap[["channel", "campaign_type", "campaign_id"] + q_cols].copy()
            try:
                reconciled_q, _ = reconcile_forecast(
                    hist_long, live_bottom, quantiles=quantiles, forecast_ds=origin_date,
                )
            except Exception:
                n_skipped += 1
                continue
            n_snapshots += 1
            rq = reconciled_q.set_index("unique_id")

            if "total" in rq.index:
                y_by_level["total"].append(float(snap["target_revenue"].sum()))
                q_by_level["total"].append(rq.loc["total", q_cols].to_numpy(dtype=float))

            for ch, y_true in snap.groupby("channel", observed=True)["target_revenue"].sum().items():
                uid = f"total/{ch}"
                if uid in rq.index:
                    y_by_level["channel"].append(float(y_true))
                    q_by_level["channel"].append(rq.loc[uid, q_cols].to_numpy(dtype=float))

            for (ch, ctype), y_true in snap.groupby(["channel", "campaign_type"], observed=True)["target_revenue"].sum().items():
                uid = f"total/{ch}/{ctype}"
                if uid in rq.index:
                    y_by_level["campaign_type"].append(float(y_true))
                    q_by_level["campaign_type"].append(rq.loc[uid, q_cols].to_numpy(dtype=float))

            for _, row in snap.iterrows():
                uid = f"total/{row['channel']}/{row['campaign_type']}/{row['campaign_id']}"
                if uid in rq.index:
                    y_by_level["campaign"].append(float(row["target_revenue"]))
                    q_by_level["campaign"].append(rq.loc[uid, q_cols].to_numpy(dtype=float))

    out = {"n_snapshots": n_snapshots, "n_snapshots_skipped": n_skipped}
    for lv in levels:
        y = np.asarray(y_by_level[lv], dtype=float)
        q = np.asarray(q_by_level[lv], dtype=float) if q_by_level[lv] else np.zeros((0, len(quantiles)))
        diag = reliability_diagram(y, q, quantiles) if len(y) else {}
        out[lv] = {**diag, "n_observations": int(len(y))}
    return out

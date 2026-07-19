"""
sanity_clamps.py
=================
Implementation Plan §G.3 — business-plausibility clamps, the numeric-model
"hallucination" guardrail. Applied to every forecast before it reaches a user
or the LLM summary layer.

Deliberately NOT a single global ROAS bound: recall the grounded facts from
§1 — Display showed literally 0 ROAS historically, Bing Shopping showed
17.7x. A per-(channel, campaign_type) envelope (with margin) is required.

Per §G.3: violations are never silently clipped in the numbers used for
metrics/logging (that would hide a real model problem) — only the
user-facing display value is bounded, with a visible caveat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_roas_bounds(canonical_df: pd.DataFrame, margin: float = 1.5, min_days: int = 10) -> dict:
    """
    Per (channel, campaign_type): plausible ROAS envelope from that group's
    own historical daily ROAS distribution (5th/95th percentile, widened by
    `margin`), not a single global number.
    """
    df = canonical_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = (
        df.groupby(["channel", "campaign_type", "date"], as_index=False)
          .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
    )
    daily = daily[daily["spend"] > 0].copy()
    daily["roas"] = daily["revenue"] / daily["spend"]

    bounds = {}
    for (channel, ctype), g in daily.groupby(["channel", "campaign_type"]):
        if len(g) < min_days:
            bounds[(channel, ctype)] = {
                "lo": 0.0, "hi": None, "median": float(g["roas"].median()) if len(g) else 0.0,
                "n_days": int(len(g)), "note": "insufficient_history_wide_bound",
            }
            continue
        p5, p50, p95 = np.percentile(g["roas"], [5, 50, 95])
        lo = max(0.0, p5 / margin)
        hi = p95 * margin
        bounds[(channel, ctype)] = {
            "lo": float(lo), "hi": float(hi), "median": float(p50),
            "n_days": int(len(g)), "note": "empirical",
        }
    return bounds


def check_forecast_plausibility(
    revenue_pred: float, spend_used: float, channel: str, campaign_type: str, bounds: dict,
) -> dict:
    """
    Returns a dict: {violation: bool, implied_roas, bound_lo, bound_hi,
    display_value (clipped only if violation, else == revenue_pred),
    caveat: str|None}. Never mutates the raw model output — callers decide
    whether to store `display_value` separately from the true prediction.
    """
    revenue_pred = max(0.0, float(revenue_pred))
    key = (channel, campaign_type)
    b = bounds.get(key)
    if b is None or spend_used <= 0 or b["hi"] is None:
        return {"violation": False, "implied_roas": None, "bound_lo": None, "bound_hi": None,
                "display_value": revenue_pred, "caveat": None}

    implied_roas = revenue_pred / spend_used
    violation = not (b["lo"] <= implied_roas <= b["hi"])
    display_value = revenue_pred
    caveat = None
    if violation:
        clipped_roas = min(max(implied_roas, b["lo"]), b["hi"])
        display_value = clipped_roas * spend_used
        caveat = (
            f"Forecast implies {implied_roas:.2f}x ROAS for {channel}/{campaign_type}, outside "
            f"that group's historical range [{b['lo']:.2f}x, {b['hi']:.2f}x] (n={b['n_days']} days). "
            f"Displayed value is capped to the historical envelope; the uncapped model output is "
            f"logged for review, not hidden."
        )
    return {
        "violation": bool(violation), "implied_roas": float(implied_roas),
        "bound_lo": b["lo"], "bound_hi": b["hi"], "display_value": float(display_value),
        "caveat": caveat,
    }


def apply_clamps_to_frame(
    forecast_df: pd.DataFrame, bounds: dict,
    revenue_col: str = "q0.5", spend_col: str = "planned_future_daily_budget_total",
) -> tuple[pd.DataFrame, list[dict]]:
    """Vectorized wrapper of `check_forecast_plausibility` over a forecast
    dataframe with channel/campaign_type/revenue/spend columns. Returns the
    frame with `<revenue_col>_display` + `plausibility_flag` columns added,
    and a separate violations log (for the technical-documentation appendix
    / data-health panel)."""
    out = forecast_df.copy()
    display_vals, flags, violations_log = [], [], []
    for _, row in out.iterrows():
        res = check_forecast_plausibility(
            row[revenue_col], row.get(spend_col, 0.0), row["channel"], row["campaign_type"], bounds,
        )
        display_vals.append(res["display_value"])
        flags.append(res["violation"])
        if res["violation"]:
            violations_log.append({
                "channel": row["channel"], "campaign_type": row["campaign_type"],
                "implied_roas": res["implied_roas"], "bound_lo": res["bound_lo"],
                "bound_hi": res["bound_hi"], "caveat": res["caveat"],
            })
    out[f"{revenue_col}_display"] = display_vals
    out["plausibility_flag"] = flags
    return out, violations_log

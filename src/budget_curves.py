"""
budget_curves.py
=================
Implementation Plan §F — Hill saturation curve, explicitly NOT a Media Mix
Model / attribution engine. Used only as a secondary sanity/scaling layer on
top of the pooled quantile model's own budget response (§F.2).

    response(spend) = L * spend^n / (K^n + spend^n)

  - K = half-saturation spend (diminishing-returns point).
  - n = curve steepness.
  - L = asymptotic revenue ceiling for that (channel, campaign_type) group.

Fit per (channel, campaign_type) on DAILY AGGREGATED (summed across that
group's campaigns) (spend, revenue) pairs — §F.1: individual campaigns don't
carry enough distinct spend levels to identify a saturation curve on their
own, a channel x type group does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _hill(spend: np.ndarray, L: float, K: float, n: float) -> np.ndarray:
    spend = np.clip(spend, 0, None)
    return L * (spend ** n) / (K ** n + spend ** n + 1e-9)


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted least-squares R², reducing to the ordinary formula when
    `weights` is None. Deliberately scored with the SAME weights the curve
    was fit against (see `fit_hill_curves`'s recency-weighting note) — an
    unweighted gate would penalize a genuinely recency-shifted fit for not
    explaining old points it was intentionally told to trust less, rejecting
    exactly the cases where recency weighting is doing real work."""
    if weights is None:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    else:
        w_mean = float(np.average(y_true, weights=weights))
        ss_res = float(np.sum(weights * (y_true - y_pred) ** 2))
        ss_tot = float(np.sum(weights * (y_true - w_mean) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_hill_curves(canonical_df: pd.DataFrame, min_points: int = 8, min_r_squared: float = 0.10,
                     recency_half_life_days: float | None = 45.0) -> dict:
    """
    Returns {(channel, campaign_type): {"L":..,"K":..,"n":..,"fit_ok":bool,
    "n_points":int, "fallback_roas": float, "r_squared": float|None,
    "n_param_at_bound": bool|None, "note": str|None, "residual_std": float,
    "recency_weighted": bool, "recency_half_life_days": float|None}}
    for every (channel, campaign_type) group observed in the data.
    `residual_std` (population std of observed-minus-predicted daily
    revenue, computed either way -- around the fitted curve if fit_ok, else
    around the flat fallback-ROAS line) is used downstream by
    `optimize_budget_allocation`'s uncertainty band on the recommended
    allocation's predicted revenue -- an approximate, residual-based band,
    not a formally derived predictive interval (this module remains a
    secondary sanity layer, not the primary calibrated CQR forecast).

    §7 finding (Part 2 review): passing the `n`/`n_unique` count checks is not
    the same as the curve being a real, identifiable saturation relationship —
    a zero-inflated/near-constant-ROAS group (e.g. a tracking-break period, or
    just a genuinely flat-ROAS channel/type with little spend variance) can
    still produce a numerically "successful" `curve_fit` call whose curve
    doesn't actually track the data. Two independent, cheap, general (not
    hardcoded to any channel/date) signals catch this:
      1. R² of the fitted curve against the observed daily (spend, revenue)
         points — below `min_r_squared` means the curve explains close to
         nothing beyond the group's own mean.
      2. The fitted steepness `n` landing exactly on its optimizer bound —
         a classic sign `curve_fit` ran to the edge of the search space
         rather than converging on an interior optimum.
    Either signal alone rejects the fit and falls back to the flat
    historical-ROAS line (still a defensible response curve — just linear,
    not saturating) with the reason recorded in `note`, never silently used
    at full confidence.

    §F re-audit (2026, "Learning to Spend: MPC for Budgeting under
    Non-Stationary Returns"): channel effectiveness is not static — a curve
    fit that weights every historical day equally implicitly assumes it is.
    Two responses to that finding now live in this module: this function's
    own `recency_half_life_days` argument (a cheap, single-curve fix — favor
    *recent* channel behavior within one fit, described below), and, since
    the following round of work, `optimize_budget_allocation_mpc` (§F.4,
    later in this file) — the actual closed-loop, rolling-horizon
    re-optimizer the paper describes, backtested honestly against a
    frozen one-shot allocation on this project's own historical data. This
    function's recency weighting remains useful on its own (every single
    non-MPC curve fit in the pipeline benefits from it, not just the MPC
    allocator's per-window refits), so it's kept as a first-class,
    independently-testable option rather than folded away now that §F.4
    exists.
    `scipy.optimize.curve_fit`'s native `sigma` argument (a per-point
    observation uncertainty; weighted least squares is 1/sigma²).
    `recency_half_life_days` turns each daily point's age (relative to the
    group's own most recent observed day) into an exponential-decay weight —
    a point `recency_half_life_days` old carries half the weight of today's —
    then into the `sigma` `curve_fit` expects (`sigma = 1/sqrt(weight)`, so
    high-weight recent points get small sigma and are trusted more). Pass
    `None` to disable and recover the previous equal-weighted fit exactly
    (what tests that build a single, time-invariant synthetic relationship
    rely on, since there's nothing for recency weighting to correct there).
    The R² fit-quality gate is scored with the SAME weights as the fit
    itself (see `_r_squared`), so a genuine recent-regime shift doesn't get
    rejected for disagreeing with de-emphasized old points; `residual_std`
    stays unweighted against ALL points, since it feeds the allocator's
    uncertainty band on overall variability, a separate concern from which
    points the curve was optimized to track.
    """
    df = canonical_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = (
        df.groupby(["channel", "campaign_type", "date"], as_index=False)
          .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
    )

    curves: dict = {}
    for (channel, ctype), g in daily.groupby(["channel", "campaign_type"]):
        spend = g["spend"].to_numpy()
        revenue = g["revenue"].to_numpy()
        dates = g["date"].to_numpy()
        mask = spend > 0
        spend, revenue, dates = spend[mask], revenue[mask], dates[mask]
        n_unique = len(np.unique(np.round(spend, 0)))

        # Recency weight computed ONCE, up front, so it's available to BOTH
        # the curve fit's `sigma` below AND the flat fallback_roas line here
        # -- a group that doesn't clear the real-fit gates (min_points/
        # n_unique/R²) is not a rare edge case in sparse real ad data, and
        # it must not silently lose all recency-awareness just because it
        # fell back. An unweighted fallback average blends a channel's
        # brand-new regime in with months of stale history at equal
        # weight, which is exactly the failure mode the MPC allocator
        # (§F.4) most needs recency weighting to avoid, since a freshly
        # drifted channel is disproportionately likely to be the one still
        # running on too little post-drift data to support a real curve.
        recency_weight = None
        if recency_half_life_days is not None and recency_half_life_days > 0 and len(dates):
            age_days = (dates.max() - dates) / np.timedelta64(1, "D")
            recency_weight = np.clip(0.5 ** (age_days / recency_half_life_days), 1e-3, 1.0)

        if recency_weight is not None and spend.sum() > 0:
            weighted_spend_sum = float(np.sum(spend * recency_weight))
            fallback_roas = (
                float(np.sum(revenue * recency_weight) / weighted_spend_sum) if weighted_spend_sum > 0 else 0.0
            )
        else:
            fallback_roas = float(revenue.sum() / spend.sum()) if spend.sum() > 0 else 0.0

        entry = {
            "fit_ok": False, "n_points": int(len(spend)), "fallback_roas": fallback_roas,
            "L": None, "K": None, "n": None, "r_squared": None,
            "n_param_at_bound": None, "note": "insufficient_data_for_fit_attempt",
            "residual_std": float(revenue.std()) if len(revenue) else 0.0,
            "recency_weighted": False, "recency_half_life_days": recency_half_life_days,
        }

        if len(spend) >= min_points and n_unique >= 5 and revenue.sum() > 0:
            n_lo, n_hi = 0.3, 6.0
            sigma = 1.0 / np.sqrt(recency_weight) if recency_weight is not None else None
            try:
                L0 = float(revenue.max()) * 1.5 + 1.0
                K0 = float(np.median(spend))
                p0 = [L0, K0, 1.3]
                bounds = ([0, 1e-3, n_lo], [L0 * 20 + 1, spend.max() * 50 + 1, n_hi])
                fit_kwargs = dict(p0=p0, bounds=bounds, maxfev=8000)
                if sigma is not None:
                    fit_kwargs.update(sigma=sigma, absolute_sigma=False)
                popt, _ = curve_fit(_hill, spend, revenue, **fit_kwargs)
                L, K, n = [float(x) for x in popt]
                pred = _hill(spend, L, K, n)
                r2 = _r_squared(revenue, pred, weights=recency_weight if sigma is not None else None)
                at_bound = bool(n <= n_lo * 1.001 or n >= n_hi * 0.999)
                resid_std = float(np.std(revenue - pred))
                entry.update({"r_squared": round(r2, 4), "n_param_at_bound": at_bound, "residual_std": resid_std,
                              "recency_weighted": sigma is not None})

                if r2 >= min_r_squared and not at_bound:
                    entry.update({"fit_ok": True, "L": L, "K": K, "n": n, "note": "fit_ok"})
                else:
                    reason = []
                    if r2 < min_r_squared:
                        reason.append(f"r_squared={r2:.3f} < {min_r_squared}")
                    if at_bound:
                        reason.append(f"n={n:.2f} pinned at optimizer bound [{n_lo},{n_hi}]")
                    entry["note"] = (
                        "fit_rejected_low_quality (" + "; ".join(reason) +
                        ") -- falling back to flat historical-ROAS line"
                    )
            except Exception as exc:
                entry["note"] = f"curve_fit_failed: {exc}"

        curves[(channel, ctype)] = entry

    return curves


def hill_predict(spend: float, curve: dict) -> float:
    """Predicted revenue for a given spend level from a fitted (or fallback) curve."""
    if curve.get("fit_ok"):
        return float(_hill(np.array([spend]), curve["L"], curve["K"], curve["n"])[0])
    return float(spend * curve.get("fallback_roas", 0.0))


def marginal_return(spend: float, curve: dict) -> float:
    """
    d(revenue)/d(spend) at this spend level — the return on the *next* dollar,
    not `revenue / spend` (average ROAS). This is the industry-standard
    "marginal ROAS" (mROAS) framing: the optimal cross-channel allocation is
    exactly the one where every group's marginal return is equalized (the
    DP in `optimize_budget_allocation` finds this implicitly; this function
    just makes that number visible rather than leaving it implicit).

    For the flat fallback line (revenue = spend * fallback_roas), marginal
    and average ROAS are identical by construction — a linear function has
    constant slope. For a fitted Hill curve, the derivative of
    L*x^n / (K^n + x^n) is:

        L * n * K^n * x^(n-1) / (K^n + x^n)^2

    which is DECREASING in x once past the curve's inflection region — the
    formal statement of "diminishing returns" this whole module exists to
    quantify.
    """
    if not curve.get("fit_ok"):
        return float(curve.get("fallback_roas", 0.0))
    L, K, n = curve["L"], curve["K"], curve["n"]
    x = max(float(spend), 1e-6)
    num = L * n * (K ** n) * (x ** (n - 1))
    den = (K ** n + x ** n) ** 2 + 1e-9
    return float(num / den)


def saturation_status(current_spend: float, curve: dict) -> dict:
    """§H.2's `saturation_status` field: how close to the curve's diminishing-
    returns point is the current spend level?"""
    if not curve.get("fit_ok"):
        return {"status": "unknown_insufficient_data", "spend_to_K_ratio": None}
    ratio = current_spend / curve["K"] if curve["K"] else None
    if ratio is None:
        status = "unknown"
    elif ratio < 0.5:
        status = "room_to_grow"
    elif ratio < 1.5:
        status = "approaching_saturation"
    else:
        status = "near_saturation"
    return {"status": status, "spend_to_K_ratio": round(ratio, 2) if ratio is not None else None}


def hill_sanity_check(
    curve: dict, historical_avg_daily_spend: float, new_daily_spend: float,
    model_ratio: float, divergence_threshold: float = 1.75,
) -> dict:
    """
    §F.2(2) — compare the Hill-curve-implied revenue ratio for a budget change
    against the ratio the pooled quantile model itself predicted for the same
    change. Large divergence => flag as a plausibility warning, never silently
    override either number.
    """
    if not curve.get("fit_ok") or historical_avg_daily_spend <= 0:
        return {"checked": False, "flag": False, "hill_ratio": None, "model_ratio": model_ratio}

    hill_hist = hill_predict(historical_avg_daily_spend, curve)
    hill_new = hill_predict(new_daily_spend, curve)
    hill_ratio = (hill_new / hill_hist) if hill_hist > 1e-6 else None

    flag = False
    if hill_ratio is not None and hill_ratio > 1e-6:
        rel_divergence = max(model_ratio, hill_ratio) / max(min(model_ratio, hill_ratio), 1e-6)
        flag = rel_divergence >= divergence_threshold

    return {"checked": True, "flag": bool(flag), "hill_ratio": hill_ratio, "model_ratio": model_ratio}


def curves_to_json(curves: dict) -> list[dict]:
    out = []
    for (channel, ctype), c in curves.items():
        out.append({"channel": channel, "campaign_type": ctype, **c})
    return out


def mpc_backtest_to_json(report: dict) -> dict:
    """JSON-safe version of `optimize_budget_allocation_mpc`'s return value
    -- same "tuple key -> {channel, campaign_type, ...}" convention
    `curves_to_json` already uses above, applied to every per-group
    allocation dict nested inside the per-window report (dict keys must be
    strings for `json.dump`; `(channel, campaign_type)` tuples aren't)."""
    def _alloc_to_list(alloc: dict) -> list[dict]:
        return [{"channel": c, "campaign_type": t, "daily_spend": v} for (c, t), v in alloc.items()]

    out = {k: v for k, v in report.items() if k != "windows" and k != "open_loop_allocation_used_throughout"}
    out["windows"] = [
        {
            **{k: v for k, v in w.items() if k not in ("mpc_allocation", "open_loop_allocation")},
            "mpc_allocation": _alloc_to_list(w["mpc_allocation"]),
            "open_loop_allocation": _alloc_to_list(w["open_loop_allocation"]),
        }
        for w in report["windows"]
    ]
    out["open_loop_allocation_used_throughout"] = _alloc_to_list(report["open_loop_allocation_used_throughout"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §F.3 — cross-channel budget allocator (built on the curves above; never
# touches training, never re-fits anything -- pure post-processing over
# already-computed hill_curves.json)
# ─────────────────────────────────────────────────────────────────────────────
def optimize_budget_allocation(
    curves: dict[tuple[str, str], dict],
    historical_daily_spend: dict[tuple[str, str], float],
    total_daily_budget: float,
    n_grid_steps: int = 200,
    max_scale_factor: float = 4.0,
    min_blended_roas: float | None = None,
) -> dict:
    """
    Given a fixed total daily budget, find the split across every
    (channel, campaign_type) group in `curves` that maximizes total daily
    revenue, using each group's already-fitted Hill curve (or its flat
    historical-ROAS fallback where no reliable curve was fit -- §F.1).

    Why dynamic programming, not a greedy "give the next dollar to whichever
    group has the best marginal return" water-filling loop: a Hill curve with
    n > 1 is genuinely S-shaped, not concave, so its marginal return can be
    LOW near zero spend and HIGHER a bit further out (the "ramp-up" region
    before the curve's inflection point). A greedy step-by-step allocator can
    get permanently stuck starving a group of exactly the spend it would need
    to reach its productive region, because the *first* dollar there looks
    worse than the first dollar elsewhere -- even when a larger commitment to
    that group would have paid off. Discretizing the budget into
    `n_grid_steps` units and solving the resulting separable resource-
    allocation problem with DP is correct regardless of each curve's shape;
    the only approximation is the grid's coarseness, which is directly
    controllable via `n_grid_steps` (200 steps => within 0.5% of the true
    optimum for any reasonably smooth curve).

    Every group's own allocation is capped at `max_scale_factor` times its
    own historical average daily spend (falling back to the *entire* budget
    as the cap when a group has no spend history at all, e.g. a brand-new
    campaign type) -- kept within a defensible extrapolation range of the
    data its curve was actually fit on, the same spirit as this codebase's
    existing plausibility clamps (`sanity_clamps.py`) rather than letting the
    optimizer recommend, say, moving an entire six-figure budget into a
    campaign type that has historically spent 200/day.

    Since every curve here is monotonically non-decreasing in spend (by
    construction -- see `_hill`'s docstring), spending the full budget is
    always at least as good as spending less of it, so -- with no ROAS
    floor -- this allocates the ENTIRE `total_daily_budget` (up to the sum
    of all per-group caps) rather than allowing money to go deliberately
    unspent.

    `min_blended_roas`, if given, changes that: real agencies don't always
    want pure revenue-max if it means running the account at an
    unacceptably low blended return. The DP already computes, as a
    byproduct, `dp[b]` = the best achievable revenue at EVERY possible total
    spend level from 0 up to the budget (not just the final one) -- this is
    the account's full revenue-vs-spend efficient frontier. Enforcing a
    floor is therefore just a different choice of where to stop on a
    frontier that's already been fully solved, not a second optimization:
    the largest total spend b (up to the budget) whose blended ROAS
    `dp[b] / (b * step)` still clears the floor. Since marginal return is
    non-increasing far enough out on this frontier, this may recommend
    spending LESS than the full budget when the floor is strict enough that
    spending it all would dilute blended ROAS below the target -- an
    intentional, economically sensible outcome, not a bug.

    Returns
    -------
    {
      "allocation": {(channel, campaign_type): recommended_daily_spend, ...},
      "marginal_roas": {(channel, campaign_type): return on the NEXT dollar
          at the recommended spend level, ...},   # see `marginal_return`
      "predicted_daily_revenue": float,   # at the recommended allocation
      "predicted_daily_revenue_low": float,   # approximate ~P25 band (see note below)
      "predicted_daily_revenue_high": float,  # approximate ~P75 band
      "blended_roas": float,               # realized total_revenue / total_spend
      "roas_floor_binding": bool,           # True if the floor, not the budget cap, decided the stopping point
      "grid_step": float,                   # dollar value of one DP grid unit
      "unallocated_budget": float,          # > 0 if the floor stopped spending before the full budget, or caps were hit first
    }

    The uncertainty band is an approximation, not a formally derived
    predictive interval: each group's `residual_std` (the spread of its own
    historical daily revenue around its fitted curve, or around the flat
    fallback line -- computed in `fit_hill_curves`) is combined across
    groups assuming rough independence (sqrt of the sum of squares), then
    applied as point estimate ∓ 0.6745·combined_std (the normal-distribution
    z-score for the 25th/75th percentiles). This module remains a secondary
    sanity layer on top of the pooled quantile model's own CQR-calibrated
    forecast (§6), which is this project's actual primary, rigorously
    calibrated uncertainty story -- this band exists so the allocator's
    output isn't a bare point estimate either, in the same probabilistic-
    range spirit, not because it carries the same calibration guarantee.
    """
    groups = list(curves.keys())
    if not groups or total_daily_budget <= 0:
        return {
            "allocation": {g: 0.0 for g in groups}, "marginal_roas": {g: 0.0 for g in groups},
            "predicted_daily_revenue": 0.0, "predicted_daily_revenue_low": 0.0,
            "predicted_daily_revenue_high": 0.0, "blended_roas": 0.0, "roas_floor_binding": False,
            "grid_step": 0.0, "unallocated_budget": max(total_daily_budget, 0.0),
        }

    step = total_daily_budget / n_grid_steps
    n_units = n_grid_steps  # budget is discretized into exactly this many units of `step`

    # Per-group cap, in grid units. No spend history at all -> allow the
    # group to absorb up to the entire budget rather than pinning it at zero.
    caps_units = []
    for g in groups:
        hist = historical_daily_spend.get(g, 0.0)
        cap_amount = (hist * max_scale_factor) if hist > 0 else total_daily_budget
        caps_units.append(int(min(n_units, np.floor(cap_amount / step + 1e-9))))

    # Precompute each group's revenue at every grid-unit spend level.
    unit_spends = np.arange(0, n_units + 1) * step
    rev_table = np.zeros((len(groups), n_units + 1))
    for gi, g in enumerate(groups):
        curve = curves[g]
        rev_table[gi] = np.array([hill_predict(float(s), curve) for s in unit_spends])

    NEG_INF = -1e18
    dp_prev = np.full(n_units + 1, NEG_INF)
    dp_prev[0] = 0.0
    choice = np.zeros((len(groups), n_units + 1), dtype=int)

    for gi in range(len(groups)):
        dp_cur = np.full(n_units + 1, NEG_INF)
        cap = caps_units[gi]
        for b in range(n_units + 1):
            if dp_prev[b] <= NEG_INF / 2:
                continue
            max_k = min(cap, n_units - b)
            if max_k < 0:
                continue
            k_range = np.arange(0, max_k + 1)
            candidates = dp_prev[b] + rev_table[gi, k_range]
            targets = b + k_range
            better = candidates > dp_cur[targets]
            better_targets = targets[better]
            dp_cur[better_targets] = candidates[better]
            choice[gi, better_targets] = k_range[better]
        dp_prev = dp_cur

    reachable = np.where(dp_prev > NEG_INF / 2)[0]
    if len(reachable) == 0:
        return {
            "allocation": {g: 0.0 for g in groups}, "marginal_roas": {g: 0.0 for g in groups},
            "predicted_daily_revenue": 0.0, "predicted_daily_revenue_low": 0.0,
            "predicted_daily_revenue_high": 0.0, "blended_roas": 0.0, "roas_floor_binding": False,
            "grid_step": step, "unallocated_budget": total_daily_budget,
        }

    # Unconstrained "spend it all" choice -- the largest reachable b.
    best_b_unconstrained = int(reachable.max())
    best_b = best_b_unconstrained
    floor_binding = False

    if min_blended_roas is not None:
        # Largest b whose blended ROAS on the efficient frontier still
        # clears the floor. b=0 vacuously satisfies any floor (zero spend,
        # zero revenue) so it's always a valid fallback if nothing else does.
        candidate_b = 0
        for b in reachable:
            if b == 0:
                continue
            spend_b = b * step
            roas_b = dp_prev[b] / spend_b if spend_b > 0 else 0.0
            if roas_b >= min_blended_roas and b > candidate_b:
                candidate_b = int(b)
        best_b = candidate_b
        floor_binding = best_b < best_b_unconstrained

    allocation = {}
    b = best_b
    for gi in reversed(range(len(groups))):
        k = int(choice[gi, b])
        allocation[groups[gi]] = float(k * step)
        b -= k

    predicted_revenue = float(dp_prev[best_b])
    total_spend = best_b * step
    blended_roas = (predicted_revenue / total_spend) if total_spend > 0 else 0.0

    marginal = {g: marginal_return(allocation[g], curves[g]) for g in groups}

    combined_variance = sum(float(curves[g].get("residual_std", 0.0)) ** 2 for g in groups)
    combined_std = float(np.sqrt(combined_variance))
    band_z = 0.6745  # normal-approx z-score for the 25th/75th percentiles
    revenue_low = max(0.0, predicted_revenue - band_z * combined_std)
    revenue_high = predicted_revenue + band_z * combined_std

    return {
        "allocation": allocation,
        "marginal_roas": marginal,
        "predicted_daily_revenue": predicted_revenue,
        "predicted_daily_revenue_low": revenue_low,
        "predicted_daily_revenue_high": revenue_high,
        "blended_roas": blended_roas,
        "roas_floor_binding": floor_binding,
        "grid_step": step,
        "unallocated_budget": float(total_daily_budget - total_spend),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §F.4 — MPC-style, non-stationary budget allocation. The closed-loop
# extension `fit_hill_curves`'s own §F re-audit note flagged as out of scope
# at the time; implemented and backtested here.
# ─────────────────────────────────────────────────────────────────────────────
def optimize_budget_allocation_mpc(
    canonical_df: pd.DataFrame,
    total_daily_budget: float,
    horizon_days: int = 90,
    replan_every_days: int = 30,
    backtest_start=None,
    spend_execution_noise_std_frac: float = 0.05,
    eval_lookback_days: int = 15,
    n_grid_steps: int = 200,
    max_scale_factor: float = 4.0,
    min_blended_roas: float | None = None,
    fit_min_points: int = 8,
    fit_min_r_squared: float = 0.10,
    recency_half_life_days: float | None = 45.0,
    seed: int = 0,
) -> dict:
    """§F.4 — rolling-horizon (Model Predictive Control) budget allocation,
    backtested honestly against a frozen one-shot allocation on this
    project's OWN historical data. `optimize_budget_allocation` above finds
    the best split of a budget given ONE already-fitted set of curves; this
    function asks the harder, closed-loop question the "Learning to Spend"
    paper is actually about: as more real data arrives and channel
    effectiveness genuinely drifts, does periodically RE-fitting the curves
    and RE-solving the allocation — rather than deciding once and running
    that same plan for the whole horizon — actually earn back more revenue
    than it costs to compute?

    Honest scope, stated up front: this is a BACKTEST over `canonical_df`'s
    own historical timeline, not a live continuously-arriving-data
    deployment (there's no live environment to deploy into here). The
    closed-loop MPC LOOP ITSELF is fully real and fully implemented — the
    curves genuinely get refit at every replanning point from only the data
    that would genuinely have been available by then, no peeking — what's
    scoped down from "the full version of the paper" is that the ground
    truth used to SCORE each window's decision is estimated retrospectively
    from this project's own historical (channel, campaign_type) spend/
    revenue pairs (see `eval_lookback_days` below), rather than observed
    from a live system responding to the recommended spend in real time.
    That is the honest, available substitute for a live environment, the
    same "walk-forward against real historical data, never letting the
    decision-maker see the future" methodology this whole project already
    uses everywhere else (§G walk-forward CV, the final holdout, etc.) —
    just applied to a control policy instead of a forecast.

    How it works
    ------------
    1. The horizon (`horizon_days`, default 90 — this project's own longest
       forecasting window) is split into `replan_every_days`-long windows
       (default 30 — this project's shortest window, so a 90-day horizon
       becomes exactly 3 replanning points, one per the report's own
       30/60/90-day cadence).
    2. OPEN-LOOP baseline: `optimize_budget_allocation` is solved exactly
       ONCE, using only data strictly before `backtest_start`, on curves
       fit from `fit_hill_curves`. That SAME allocation is then reused,
       unchanged, for every window across the whole horizon — a real
       agency's "set the media plan once at the start of the quarter and
       don't touch it" baseline.
    3. MPC (closed-loop): at the start of EACH window, curves are refit
       from `fit_hill_curves` on ALL data known as of that window's start
       (an expanding window — never including that window's own data or
       anything after it) and the allocation is RE-solved for that window.
       Later windows see channel behavior the open-loop plan's curves never
       could have — this is the entire mechanism under test.
    4. Planned-vs-realized spend noise (this round's addition, directly
       modeled rather than assumed away): neither method's planned
       allocation is assumed to execute exactly as planned. Each group's
       planned spend is perturbed by `spend_execution_noise_std_frac`
       (Gaussian, relative) before being scored — real pacing/delivery
       algorithms rarely hit the exact planned number, and this keeps both
       methods honestly exposed to the same execution slippage rather than
       comparing two idealized, frictionless plans.
    5. Ground-truth evaluation: for each window, a SEPARATE set of curves
       (`eval_curves`) is fit purely retrospectively on the real
       (channel, campaign_type) spend/revenue pairs actually observed
       during that window (extended `eval_lookback_days` backward for
       enough points to fit on — short windows alone often can't clear
       `fit_hill_curves`' own `min_points`/`n_unique` gates, in which case
       it gracefully falls back to that group's real realized-ROAS line for
       the window, same fallback behavior as everywhere else this function
       is used). BOTH methods' (noisy) allocations are scored against these
       IDENTICAL eval_curves — the only thing that differs between MPC and
       open-loop in the final revenue numbers is which allocation each one
       chose, never which yardstick it's judged by. `eval_curves` are never
       seen by either method's own decision-making curve fit.

    Returns
    -------
    {
      "backtest_start": str, "horizon_days": int, "replan_every_days": int,
      "n_windows": int, "spend_execution_noise_std_frac": float,
      "windows": [{  # one entry per replanning window, in order
          "window_start": str, "window_end": str, "n_days": int,
          "mpc_allocation": {...}, "mpc_realized_daily_revenue": float,
          "mpc_realized_daily_spend": float,
          "open_loop_allocation": {...}, "open_loop_realized_daily_revenue": float,
          "open_loop_realized_daily_spend": float,
          "n_groups_with_fresh_curve_fit": int,  # how many groups got a real (non-fallback) MPC refit this window
      }, ...],
      "mpc_avg_daily_revenue": float,          # day-weighted average across all windows
      "open_loop_avg_daily_revenue": float,
      "mpc_vs_open_loop_relative_lift": float | None,
      "open_loop_allocation_used_throughout": {...},
    }

    A negative or ~zero lift is still a genuine, useful, reportable result —
    it means channel effectiveness didn't drift enough over this
    particular historical horizon for re-solving to earn back more than the
    noise/estimation-error cost of refitting on less data per window, not
    that the mechanism is broken (see docs/technical_documentation.md §9a
    for this project's own real backtest result and that honest reading of
    it either way).
    """
    if len(canonical_df) == 0:
        raise ValueError("canonical_df is empty")
    df = canonical_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    data_min, data_max = df["date"].min(), df["date"].max()

    if backtest_start is None:
        backtest_start = data_max - pd.Timedelta(days=horizon_days)
    else:
        backtest_start = pd.Timestamp(backtest_start).normalize()

    if backtest_start <= data_min:
        raise ValueError(
            f"backtest_start={backtest_start.date()} leaves no prior history before it to fit "
            f"decision curves on (data starts {data_min.date()}); shorten horizon_days, move "
            f"backtest_start later, or supply more historical data."
        )
    if backtest_start + pd.Timedelta(days=horizon_days) > data_max + pd.Timedelta(days=1):
        raise ValueError(
            f"not enough real data after backtest_start={backtest_start.date()} to evaluate a "
            f"{horizon_days}-day horizon (data ends {data_max.date()}); shorten horizon_days or "
            f"move backtest_start earlier."
        )
    if total_daily_budget <= 0:
        raise ValueError("total_daily_budget must be > 0")

    n_windows = int(np.ceil(horizon_days / replan_every_days))
    window_bounds = []
    for i in range(n_windows):
        w_start = backtest_start + pd.Timedelta(days=i * replan_every_days)
        w_end = min(
            backtest_start + pd.Timedelta(days=(i + 1) * replan_every_days),
            backtest_start + pd.Timedelta(days=horizon_days),
        )
        if w_end > w_start:
            window_bounds.append((w_start, w_end))

    rng = np.random.default_rng(seed)

    def _fit_and_allocate(history_df: pd.DataFrame) -> tuple[dict, dict]:
        curves = fit_hill_curves(
            history_df, min_points=fit_min_points, min_r_squared=fit_min_r_squared,
            recency_half_life_days=recency_half_life_days,
        )
        hist_daily_spend: dict = {}
        if len(history_df):
            span_days = max(1, (history_df["date"].max() - history_df["date"].min()).days + 1)
            hist_sum = history_df.groupby(["channel", "campaign_type"])["spend"].sum()
            for g in curves:
                hist_daily_spend[g] = float(hist_sum.get(g, 0.0)) / span_days
        decision = optimize_budget_allocation(
            curves, hist_daily_spend, total_daily_budget,
            n_grid_steps=n_grid_steps, max_scale_factor=max_scale_factor, min_blended_roas=min_blended_roas,
        )
        return curves, decision

    def _score(allocation: dict, curves_for_score: dict) -> tuple[float, float]:
        """Apply spend-execution noise, then score the noisy realized spend
        against `curves_for_score` — the SAME eval_curves for both methods
        in a given window, so only the allocation itself differs."""
        realized_revenue, realized_spend = 0.0, 0.0
        for g, planned in allocation.items():
            noise = rng.normal(0.0, spend_execution_noise_std_frac)
            actual = max(0.0, planned * (1.0 + noise))
            curve = curves_for_score.get(g, {"fit_ok": False, "fallback_roas": 0.0})
            realized_revenue += hill_predict(actual, curve)
            realized_spend += actual
        return realized_revenue, realized_spend

    # OPEN-LOOP baseline: one decision, made once, before backtest_start.
    open_loop_history = df[df["date"] < backtest_start]
    _, open_loop_decision = _fit_and_allocate(open_loop_history)
    open_loop_allocation = open_loop_decision["allocation"]

    windows_report = []
    mpc_rev_days = open_loop_rev_days = 0.0
    total_days = 0

    for (w_start, w_end) in window_bounds:
        n_days = int((w_end - w_start).days)

        # MPC decision: refit on the expanding window of data known as of
        # w_start -- never this window's own data, never anything after it.
        mpc_history = df[df["date"] < w_start]
        mpc_curves, mpc_decision = _fit_and_allocate(mpc_history)
        mpc_allocation = mpc_decision["allocation"]

        # Retrospective ground truth: fit purely on what actually happened
        # during (and just before, for enough points) this window. Used to
        # SCORE both methods; used by NEITHER method to DECIDE.
        eval_start = w_start - pd.Timedelta(days=eval_lookback_days)
        eval_df = df[(df["date"] >= eval_start) & (df["date"] < w_end)]
        eval_curves = fit_hill_curves(
            eval_df, min_points=fit_min_points, min_r_squared=fit_min_r_squared,
            recency_half_life_days=None,  # short retrospective window -- nothing to de-emphasize within it
        )

        mpc_rev, mpc_spend = _score(mpc_allocation, eval_curves)
        ol_rev, ol_spend = _score(open_loop_allocation, eval_curves)

        mpc_rev_days += mpc_rev * n_days
        open_loop_rev_days += ol_rev * n_days
        total_days += n_days

        windows_report.append({
            "window_start": str(w_start.date()), "window_end": str(w_end.date()), "n_days": n_days,
            "mpc_allocation": mpc_allocation, "mpc_realized_daily_revenue": mpc_rev,
            "mpc_realized_daily_spend": mpc_spend,
            "open_loop_allocation": open_loop_allocation, "open_loop_realized_daily_revenue": ol_rev,
            "open_loop_realized_daily_spend": ol_spend,
            "n_groups_with_fresh_curve_fit": int(sum(1 for c in mpc_curves.values() if c.get("fit_ok"))),
        })

    mpc_avg = mpc_rev_days / total_days if total_days else 0.0
    open_loop_avg = open_loop_rev_days / total_days if total_days else 0.0
    lift = ((mpc_avg - open_loop_avg) / open_loop_avg) if open_loop_avg > 0 else None

    return {
        "backtest_start": str(backtest_start.date()), "horizon_days": horizon_days,
        "replan_every_days": replan_every_days, "n_windows": len(window_bounds),
        "spend_execution_noise_std_frac": spend_execution_noise_std_frac,
        "windows": windows_report,
        "mpc_avg_daily_revenue": mpc_avg,
        "open_loop_avg_daily_revenue": open_loop_avg,
        "mpc_vs_open_loop_relative_lift": lift,
        "open_loop_allocation_used_throughout": open_loop_allocation,
    }


def compute_hindsight_regret(canonical_df: pd.DataFrame, mpc_backtest_report: dict) -> dict:
    """
    §F.5 — hindsight-regret audit ("Auditing Marketing Budget Allocation
    with Hindsight Regret," 2026). `optimize_budget_allocation_mpc` above
    compares two ALGORITHMS against each other (MPC vs. open-loop) — this
    compares the tool against what ACTUALLY happened, the comparison that
    paper is really about: not "which allocation strategy is better," but
    "would using this tool at all have beaten the real historical decision,
    which we already know the outcome of?"

    Deliberately consumes `optimize_budget_allocation_mpc`'s own OUTPUT
    report rather than re-deriving windows or re-running any optimization —
    same window boundaries by construction, and zero new risk to that
    already-tested code path. "Actual" performance for each window is a
    plain groupby-sum over `canonical_df`'s real (channel, campaign_type)
    spend/revenue for that window's real dates — no curve, no model, no
    estimation, just what happened.

    Honest asymmetry, stated plainly: the tool's side of this comparison
    (`open_loop`/`mpc` realized revenue) is an ESTIMATE — the noisy
    realized spend from each method's chosen allocation, scored through
    that window's retrospective `eval_curves` (§F.4) — because we cannot
    know what revenue the tool's allocation would truly have produced had
    it actually been run instead of the real decision. The "actual" side
    carries no such uncertainty; it is the one real, already-known number
    in this whole comparison. This asymmetry is inherent to any
    counterfactual budget audit, not a shortcut specific to this
    implementation, and it means a positive "regret" here should be read as
    "the tool's curve-based estimate says it would have beaten reality,"
    not as a certainty.

    Returns
    -------
    {
      "windows": [{"window_start", "window_end", "n_days",
                    "actual_daily_revenue", "actual_daily_spend",
                    "open_loop_regret_daily_revenue",  # open-loop estimate MINUS actual
                    "mpc_regret_daily_revenue"}, ...],           # mpc estimate MINUS actual
      "actual_avg_daily_revenue": float,
      "open_loop_avg_daily_revenue": float,   # same numbers as the MPC report, carried through
      "mpc_avg_daily_revenue": float,
      "open_loop_vs_actual_uplift_pct": float | None,
      "mpc_vs_actual_uplift_pct": float | None,
    }
    """
    df = canonical_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    windows_out = []
    total_days = 0
    actual_rev_days = open_loop_rev_days = mpc_rev_days = 0.0

    for w in mpc_backtest_report.get("windows", []):
        w_start = pd.Timestamp(w["window_start"])
        w_end = pd.Timestamp(w["window_end"])
        n_days = int(w["n_days"])
        mask = (df["date"] >= w_start) & (df["date"] < w_end)
        actual_daily_revenue = float(df.loc[mask, "revenue"].sum()) / max(n_days, 1)
        actual_daily_spend = float(df.loc[mask, "spend"].sum()) / max(n_days, 1)

        windows_out.append({
            "window_start": w["window_start"], "window_end": w["window_end"], "n_days": n_days,
            "actual_daily_revenue": actual_daily_revenue, "actual_daily_spend": actual_daily_spend,
            "open_loop_regret_daily_revenue": w["open_loop_realized_daily_revenue"] - actual_daily_revenue,
            "mpc_regret_daily_revenue": w["mpc_realized_daily_revenue"] - actual_daily_revenue,
        })
        total_days += n_days
        actual_rev_days += actual_daily_revenue * n_days
        open_loop_rev_days += w["open_loop_realized_daily_revenue"] * n_days
        mpc_rev_days += w["mpc_realized_daily_revenue"] * n_days

    actual_avg = actual_rev_days / total_days if total_days else 0.0
    open_loop_avg = open_loop_rev_days / total_days if total_days else 0.0
    mpc_avg = mpc_rev_days / total_days if total_days else 0.0

    return {
        "windows": windows_out,
        "actual_avg_daily_revenue": actual_avg,
        "open_loop_avg_daily_revenue": open_loop_avg,
        "mpc_avg_daily_revenue": mpc_avg,
        "open_loop_vs_actual_uplift_pct": (open_loop_avg / actual_avg - 1.0) if actual_avg > 0 else None,
        "mpc_vs_actual_uplift_pct": (mpc_avg / actual_avg - 1.0) if actual_avg > 0 else None,
    }

"""
tests/test_budget_optimizer.py
================================
Unit tests for `budget_curves.optimize_budget_allocation` (§F.3) — the DP-
based cross-channel budget allocator built on top of the already-fitted Hill
curves. Pure small-input unit tests; no training, no real data. The
integration point (does this render correctly against a real hill_curves.json
in the Streamlit app) is covered by a headless AppTest smoke check, not
re-asserted here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from budget_curves import optimize_budget_allocation, hill_predict


def _flat_curve(roas: float) -> dict:
    """A rejected/no-fit curve -- optimize_budget_allocation must handle this
    exactly the same way hill_predict does: revenue = spend * fallback_roas."""
    return {"fit_ok": False, "fallback_roas": roas}


def _hill_curve(L: float, K: float, n: float) -> dict:
    return {"fit_ok": True, "L": L, "K": K, "n": n}


def test_budget_conservation_and_nonnegative_revenue():
    curves = {("google", "Search"): _flat_curve(3.0), ("meta", "Generic"): _flat_curve(2.0)}
    out = optimize_budget_allocation(curves, {}, total_daily_budget=1000.0, n_grid_steps=100)
    total_allocated = sum(out["allocation"].values())
    assert total_allocated <= 1000.0 + 1e-6
    assert total_allocated >= 1000.0 - out["unallocated_budget"] - 1e-6
    assert out["predicted_daily_revenue"] >= 0.0


def test_zero_budget_gives_zero_allocation_and_revenue():
    curves = {("google", "Search"): _flat_curve(3.0)}
    out = optimize_budget_allocation(curves, {}, total_daily_budget=0.0)
    assert out["predicted_daily_revenue"] == 0.0
    assert all(v == 0.0 for v in out["allocation"].values())


def test_single_group_absorbs_the_whole_budget():
    curves = {("google", "Search"): _flat_curve(4.0)}
    out = optimize_budget_allocation(curves, {}, total_daily_budget=500.0, n_grid_steps=100)
    assert out["allocation"][("google", "Search")] == pytest.approx(500.0, abs=5.0)


def test_prefers_the_strictly_better_flat_roas_group_when_unconstrained():
    """Two flat (never-saturating) groups, no spend history to cap either one
    -- marginal return is CONSTANT per group (3.0 vs 1.0), so the global
    optimum is unambiguous: put everything into the higher-ROAS group. This
    is the case a naive/incorrect implementation would most obviously get
    wrong."""
    curves = {("high_roas", "x"): _flat_curve(3.0), ("low_roas", "x"): _flat_curve(1.0)}
    out = optimize_budget_allocation(curves, {}, total_daily_budget=1000.0, n_grid_steps=200)
    assert out["allocation"][("high_roas", "x")] > out["allocation"][("low_roas", "x")]
    # and it should be close to "all of it", not just "more of it"
    assert out["allocation"][("high_roas", "x")] > 900.0


def test_per_group_cap_is_respected_even_when_that_group_is_the_best_option():
    """A group with tiny spend history should NOT be recommended to absorb the
    entire budget, even if it's the mathematically best ROAS -- it's capped
    at max_scale_factor x its own historical spend to stay within a
    defensible extrapolation range of the data its curve was fit on."""
    curves = {("small_but_best", "x"): _flat_curve(10.0), ("normal", "x"): _flat_curve(1.0)}
    hist = {("small_but_best", "x"): 50.0, ("normal", "x"): 5000.0}  # 4x cap = 200
    out = optimize_budget_allocation(curves, hist, total_daily_budget=5000.0, n_grid_steps=200)
    assert out["allocation"][("small_but_best", "x")] <= 200.0 + 30.0  # + one grid step of slack


def test_dp_beats_naive_uniform_split_on_a_genuinely_sigmoidal_curve():
    """The whole reason this is DP and not greedy water-filling: an S-shaped
    (n > 1) curve can make concentrating budget in one group strictly better
    than spreading it evenly. Confirm the optimizer actually finds that,
    rather than defaulting to an even split."""
    curves = {
        ("a", "x"): _hill_curve(L=10_000, K=500, n=3.0),
        ("b", "x"): _hill_curve(L=10_000, K=500, n=3.0),
    }
    budget = 600.0  # enough to push ONE group past its steep region, not both
    out = optimize_budget_allocation(curves, {}, total_daily_budget=budget, n_grid_steps=300)

    naive_split = budget / 2
    naive_revenue = 2 * hill_predict(naive_split, curves[("a", "x")])

    assert out["predicted_daily_revenue"] >= naive_revenue - 1.0  # DP result is never worse
    # and in this specific constructed case, concentrating spend should
    # actually be BETTER than splitting evenly (the point of the test)
    assert out["predicted_daily_revenue"] > naive_revenue + 1.0


def test_more_total_budget_never_reduces_predicted_revenue():
    curves = {("a", "x"): _flat_curve(2.0), ("b", "x"): _hill_curve(L=5000, K=800, n=2.0)}
    out_small = optimize_budget_allocation(curves, {}, total_daily_budget=1000.0, n_grid_steps=200)
    out_big = optimize_budget_allocation(curves, {}, total_daily_budget=2000.0, n_grid_steps=200)
    assert out_big["predicted_daily_revenue"] >= out_small["predicted_daily_revenue"] - 1e-6


def test_deterministic_across_repeated_calls():
    curves = {("a", "x"): _flat_curve(2.5), ("b", "x"): _hill_curve(L=5000, K=800, n=2.0)}
    out1 = optimize_budget_allocation(curves, {}, total_daily_budget=1234.0, n_grid_steps=150)
    out2 = optimize_budget_allocation(curves, {}, total_daily_budget=1234.0, n_grid_steps=150)
    assert out1["allocation"] == out2["allocation"]
    assert out1["predicted_daily_revenue"] == out2["predicted_daily_revenue"]


def test_empty_curves_dict_does_not_crash():
    out = optimize_budget_allocation({}, {}, total_daily_budget=1000.0)
    assert out["allocation"] == {}
    assert out["predicted_daily_revenue"] == 0.0


def test_marginal_roas_equals_average_for_flat_fallback_curve():
    """A never-saturating flat line has constant slope -- marginal and
    average ROAS must be identical, by construction."""
    from budget_curves import marginal_return
    curve = _flat_curve(3.5)
    assert marginal_return(1000.0, curve) == pytest.approx(3.5)
    assert marginal_return(1.0, curve) == pytest.approx(3.5)


def test_marginal_roas_decreases_past_the_hill_curves_inflection_point():
    """The whole point of mROAS: it should be strictly decreasing well past
    a Hill curve's half-saturation point K, unlike average ROAS which
    changes far more slowly."""
    from budget_curves import marginal_return
    curve = _hill_curve(L=10_000, K=500, n=2.5)
    m_early = marginal_return(100.0, curve)   # well before K -- ramping up
    m_at_k = marginal_return(500.0, curve)    # at half-saturation
    m_late = marginal_return(5000.0, curve)   # deep into diminishing returns
    assert m_at_k < m_early or m_at_k > 0  # sanity: finite, positive
    assert m_late < m_at_k  # strictly diminishing past saturation


def test_optimizer_reports_marginal_roas_per_group():
    curves = {("a", "x"): _flat_curve(3.0), ("b", "x"): _hill_curve(L=5000, K=800, n=2.0)}
    out = optimize_budget_allocation(curves, {}, total_daily_budget=2000.0, n_grid_steps=200)
    assert set(out["marginal_roas"].keys()) == set(curves.keys())
    assert all(v >= 0 for v in out["marginal_roas"].values())


def test_unconstrained_allocation_has_no_floor_binding():
    curves = {("a", "x"): _flat_curve(3.0), ("b", "x"): _flat_curve(1.0)}
    out = optimize_budget_allocation(curves, {}, total_daily_budget=1000.0, n_grid_steps=100)
    assert out["roas_floor_binding"] is False
    assert out["unallocated_budget"] == pytest.approx(0.0, abs=1e-6)


def test_roas_floor_below_the_natural_optimum_is_not_binding():
    """Setting a floor comfortably below what the unconstrained optimum
    already achieves should change nothing -- same allocation, same spend."""
    curves = {("a", "x"): _flat_curve(3.0), ("b", "x"): _flat_curve(2.0)}
    unconstrained = optimize_budget_allocation(curves, {}, total_daily_budget=1000.0, n_grid_steps=200)
    floored = optimize_budget_allocation(
        curves, {}, total_daily_budget=1000.0, n_grid_steps=200, min_blended_roas=1.0,
    )
    assert floored["roas_floor_binding"] is False
    assert floored["predicted_daily_revenue"] == pytest.approx(unconstrained["predicted_daily_revenue"], rel=0.02)


def test_roas_floor_above_the_natural_optimum_forces_less_spend():
    """A floor set ABOVE what spending the full budget can achieve must be
    binding -- the allocator should recommend spending less than the full
    budget rather than blowing through the floor, since every group's
    blended-in marginal return only falls as more is spent."""
    # A single group whose average ROAS falls well below 5x once its
    # near-linear low-spend region is exhausted -- forces the constrained
    # optimum to stop early.
    curves = {("a", "x"): _hill_curve(L=2000, K=100, n=1.0)}
    out_unconstrained = optimize_budget_allocation(curves, {}, total_daily_budget=2000.0, n_grid_steps=400)
    # blended ROAS at full spend in the unconstrained case:
    full_roas = out_unconstrained["predicted_daily_revenue"] / 2000.0
    floor = full_roas * 1.5  # deliberately set the floor above what full spend achieves
    out = optimize_budget_allocation(
        curves, {}, total_daily_budget=2000.0, n_grid_steps=400, min_blended_roas=floor,
    )
    assert out["roas_floor_binding"] is True
    assert out["unallocated_budget"] > 0.0
    assert out["blended_roas"] >= floor - 0.05  # respects the floor (small tol for grid discretization)


def test_uncertainty_band_widens_with_larger_residual_std():
    curves_tight = {("a", "x"): {**_flat_curve(3.0), "residual_std": 10.0}}
    curves_wide = {("a", "x"): {**_flat_curve(3.0), "residual_std": 500.0}}
    out_tight = optimize_budget_allocation(curves_tight, {}, total_daily_budget=1000.0, n_grid_steps=100)
    out_wide = optimize_budget_allocation(curves_wide, {}, total_daily_budget=1000.0, n_grid_steps=100)
    width_tight = out_tight["predicted_daily_revenue_high"] - out_tight["predicted_daily_revenue_low"]
    width_wide = out_wide["predicted_daily_revenue_high"] - out_wide["predicted_daily_revenue_low"]
    assert width_wide > width_tight
    assert out_tight["predicted_daily_revenue_low"] <= out_tight["predicted_daily_revenue"] <= out_tight["predicted_daily_revenue_high"]


def test_residual_std_is_populated_by_fit_hill_curves():
    """Integration point with the actual fitting function, not just the
    allocator's own handling of a hand-built curve dict."""
    from budget_curves import fit_hill_curves
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=60)
    spend = rng.uniform(50, 2000, size=60)
    revenue = 3.0 * spend + rng.normal(0, 50, size=60)  # clean-ish linear relationship
    df = pd.DataFrame({
        "channel": "test_channel", "campaign_type": "test_type", "date": dates,
        "spend": spend, "revenue": revenue,
    })
    curves = fit_hill_curves(df)
    entry = curves[("test_channel", "test_type")]
    assert "residual_std" in entry
    assert entry["residual_std"] >= 0.0


def test_recency_half_life_none_disables_weighting():
    """`recency_half_life_days=None` must turn weighting off (the
    `recency_weighted` flag is False, no `sigma` passed to `curve_fit`)
    without breaking the fit itself on an ordinary single-regime series --
    this is the explicit opt-out contract callers/tests can rely on."""
    from budget_curves import fit_hill_curves
    rng = np.random.default_rng(1)
    dates = pd.date_range("2026-01-01", periods=60)
    spend = rng.uniform(50, 2000, size=60)
    revenue = 3.0 * spend + rng.normal(0, 50, size=60)
    df = pd.DataFrame({
        "channel": "c", "campaign_type": "t", "date": dates, "spend": spend, "revenue": revenue,
    })
    entry = fit_hill_curves(df, recency_half_life_days=None)[("c", "t")]
    assert entry["recency_weighted"] is False
    assert entry["recency_half_life_days"] is None
    assert entry["fit_ok"] is True


def test_recency_weighted_fit_tracks_the_recent_regime_over_the_old_one():
    """Construct a channel whose true ROAS steps from 2.0 (older 60 days) to
    5.0 (most recent 60 days) -- real channel-effectiveness drift, the exact
    scenario recency weighting (§F re-audit) exists to address. A
    short-half-life fit should predict close to the RECENT regime; the
    unweighted fit blends both and lands well short of it."""
    from budget_curves import fit_hill_curves, hill_predict
    rng = np.random.default_rng(42)
    n_old, n_new = 60, 60
    dates_old = pd.date_range("2026-01-01", periods=n_old)
    dates_new = pd.date_range(dates_old[-1] + pd.Timedelta(days=1), periods=n_new)
    spend_old = rng.uniform(100, 2000, size=n_old)
    spend_new = rng.uniform(100, 2000, size=n_new)
    revenue_old = 2.0 * spend_old + rng.normal(0, 30, size=n_old)
    revenue_new = 5.0 * spend_new + rng.normal(0, 30, size=n_new)
    df = pd.DataFrame({
        "channel": "c", "campaign_type": "t",
        "date": dates_old.append(dates_new),
        "spend": np.concatenate([spend_old, spend_new]),
        "revenue": np.concatenate([revenue_old, revenue_new]),
    })

    weighted = fit_hill_curves(df, recency_half_life_days=10.0)[("c", "t")]
    unweighted = fit_hill_curves(df, recency_half_life_days=None)[("c", "t")]
    assert weighted["recency_weighted"] is True
    assert weighted["fit_ok"] is True  # the weighted R^2 gate must not reject a real recency shift

    test_spend = 800.0
    recent_target = 5.0 * test_spend
    pred_weighted = hill_predict(test_spend, weighted)
    pred_unweighted = hill_predict(test_spend, unweighted)

    assert abs(pred_weighted - recent_target) < abs(pred_unweighted - recent_target)
    assert abs(pred_weighted - recent_target) / recent_target < 0.10  # within 10% of the true recent ROAS

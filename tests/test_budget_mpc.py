"""
tests/test_budget_mpc.py
=========================
Unit tests for `budget_curves.optimize_budget_allocation_mpc` (§F.4) — the
closed-loop, rolling-horizon extension of `optimize_budget_allocation`,
backtested against a frozen one-shot ("open-loop") allocation on historical
data.

Same spirit as `test_budget_optimizer.py`: small, direct, real-arithmetic
checks, plus one synthetic regime-shift scenario built specifically to
exercise the mechanism this function exists for (does re-solving with fresh
data actually notice and react to drifting channel effectiveness).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from budget_curves import optimize_budget_allocation_mpc


def _daily_rows(channel, ctype, start, n_days, spend_fn, roas_fn, rng, spend_noise=0.03):
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        base_spend = spend_fn(i)
        spend = max(1.0, base_spend * (1.0 + rng.normal(0, spend_noise)))
        revenue = spend * roas_fn(i)
        rows.append({
            "date": d, "channel": channel, "campaign_type": ctype,
            "campaign_id": f"{channel}_{ctype}_1", "spend": spend, "revenue": revenue,
        })
    return rows


def _regime_shift_df(pre_days=200, horizon_days=90, seed=0):
    """Two groups: A has a constant ROAS=3 throughout. B has LOW ROAS=1
    for all history strictly before `backtest_start`, then HIGH ROAS=6 for
    the entire backtest horizon onward -- a clean, unambiguous regime
    shift that only MPC (which keeps refitting) should ever notice."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    backtest_start = start + pd.Timedelta(days=pre_days)

    rows = []
    # Group A: constant ROAS=3 across the ENTIRE range (pre-history + horizon)
    rows += _daily_rows("google", "Search", start, pre_days + horizon_days,
                         spend_fn=lambda i: 80 + 5 * np.sin(i / 7), roas_fn=lambda i: 3.0, rng=rng)
    # Group B: ROAS=1 pre-backtest, ROAS=6 from backtest_start onward
    rows += _daily_rows("meta", "Generic", start, pre_days,
                         spend_fn=lambda i: 60 + 5 * np.sin(i / 5), roas_fn=lambda i: 1.0, rng=rng)
    rows += _daily_rows("meta", "Generic", backtest_start, horizon_days,
                         spend_fn=lambda i: 60 + 5 * np.sin(i / 5), roas_fn=lambda i: 6.0, rng=rng)

    df = pd.DataFrame(rows)
    return df, backtest_start


def _stable_df(n_days=250, seed=1):
    """No regime shift at all -- both groups have constant ROAS throughout.
    Used for sanity/schema tests where the drift mechanism itself isn't
    what's being tested."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    rows = []
    rows += _daily_rows("google", "Search", start, n_days,
                         spend_fn=lambda i: 100 + 10 * np.sin(i / 6), roas_fn=lambda i: 3.0, rng=rng)
    rows += _daily_rows("meta", "Generic", start, n_days,
                         spend_fn=lambda i: 80 + 8 * np.cos(i / 6), roas_fn=lambda i: 2.0, rng=rng)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Schema / validation
# ─────────────────────────────────────────────────────────────────────────────
def test_basic_schema_and_window_count():
    df = _stable_df()
    result = optimize_budget_allocation_mpc(
        df, total_daily_budget=200.0, horizon_days=90, replan_every_days=30, seed=0,
    )
    assert result["n_windows"] == 3
    assert len(result["windows"]) == 3
    for w in result["windows"]:
        assert set(w["mpc_allocation"].keys()) == set(w["open_loop_allocation"].keys())
        assert w["mpc_realized_daily_revenue"] >= 0
        assert w["open_loop_realized_daily_revenue"] >= 0
    assert result["mpc_avg_daily_revenue"] >= 0
    assert result["open_loop_avg_daily_revenue"] >= 0


def test_uneven_horizon_gives_a_shorter_final_window():
    df = _stable_df(n_days=250)
    result = optimize_budget_allocation_mpc(
        df, total_daily_budget=150.0, horizon_days=100, replan_every_days=30, seed=0,
    )
    assert result["n_windows"] == 4  # ceil(100/30)
    n_days_per_window = [w["n_days"] for w in result["windows"]]
    assert sum(n_days_per_window) == 100
    assert n_days_per_window[-1] == 10  # 100 - 3*30


def test_rejects_non_positive_budget():
    df = _stable_df()
    with pytest.raises(ValueError):
        optimize_budget_allocation_mpc(df, total_daily_budget=0.0, horizon_days=90, replan_every_days=30)


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        optimize_budget_allocation_mpc(pd.DataFrame(), total_daily_budget=100.0)


def test_rejects_backtest_start_with_no_prior_history():
    df = _stable_df(n_days=250)
    too_early = df["date"].min()
    with pytest.raises(ValueError):
        optimize_budget_allocation_mpc(df, total_daily_budget=100.0, horizon_days=30,
                                        replan_every_days=30, backtest_start=too_early)


def test_rejects_horizon_longer_than_available_future_data():
    df = _stable_df(n_days=250)
    late_start = df["date"].max() - pd.Timedelta(days=10)  # only 10 days of "future" data left
    with pytest.raises(ValueError):
        optimize_budget_allocation_mpc(df, total_daily_budget=100.0, horizon_days=90,
                                        replan_every_days=30, backtest_start=late_start)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism / noise behavior
# ─────────────────────────────────────────────────────────────────────────────
def test_deterministic_given_same_seed():
    df = _stable_df()
    r1 = optimize_budget_allocation_mpc(df, total_daily_budget=200.0, horizon_days=90,
                                         replan_every_days=30, seed=42)
    r2 = optimize_budget_allocation_mpc(df, total_daily_budget=200.0, horizon_days=90,
                                         replan_every_days=30, seed=42)
    assert r1["mpc_avg_daily_revenue"] == r2["mpc_avg_daily_revenue"]
    assert r1["open_loop_avg_daily_revenue"] == r2["open_loop_avg_daily_revenue"]


def test_zero_execution_noise_is_exactly_reproducible_regardless_of_seed():
    df = _stable_df()
    r1 = optimize_budget_allocation_mpc(df, total_daily_budget=200.0, horizon_days=90,
                                         replan_every_days=30, spend_execution_noise_std_frac=0.0, seed=1)
    r2 = optimize_budget_allocation_mpc(df, total_daily_budget=200.0, horizon_days=90,
                                         replan_every_days=30, spend_execution_noise_std_frac=0.0, seed=99)
    assert r1["mpc_avg_daily_revenue"] == pytest.approx(r2["mpc_avg_daily_revenue"])


def test_open_loop_allocation_is_frozen_across_every_window():
    """The whole point of the open-loop baseline: it decides once and never
    updates -- every window's `open_loop_allocation` must be identical."""
    df = _stable_df()
    result = optimize_budget_allocation_mpc(df, total_daily_budget=200.0, horizon_days=90,
                                             replan_every_days=30, seed=0)
    allocations = [w["open_loop_allocation"] for w in result["windows"]]
    for a in allocations[1:]:
        assert a == allocations[0]
    assert allocations[0] == result["open_loop_allocation_used_throughout"]


# ─────────────────────────────────────────────────────────────────────────────
# The actual mechanism: does MPC notice and react to a real regime shift?
# ─────────────────────────────────────────────────────────────────────────────
def test_mpc_first_window_matches_open_loop_before_any_new_information():
    """MPC's window-1 decision is made from EXACTLY the same information the
    open-loop baseline had (nothing past `backtest_start` has been observed
    yet) -- the two allocations should be identical (or extremely close;
    both call the identical DP allocator on nearly-identical fitted curves)."""
    df, backtest_start = _regime_shift_df(pre_days=200, horizon_days=90, seed=3)
    total_budget = 150.0
    result = optimize_budget_allocation_mpc(
        df, total_daily_budget=total_budget, horizon_days=90, replan_every_days=30,
        backtest_start=backtest_start, spend_execution_noise_std_frac=0.0, seed=0,
    )
    w1 = result["windows"][0]
    meta_key = ("meta", "Generic")
    assert w1["mpc_allocation"][meta_key] == pytest.approx(w1["open_loop_allocation"][meta_key], rel=0.05)


def test_mpc_shifts_budget_toward_the_now_better_channel_after_regime_shift():
    """The core value proposition: once MPC has observed a full window of
    the NEW high-ROAS regime for group B, its LATER-window allocation to B
    should be meaningfully higher than the open-loop baseline's (which never
    updates and is stuck reacting to B's old, low-ROAS history)."""
    df, backtest_start = _regime_shift_df(pre_days=200, horizon_days=90, seed=3)
    total_budget = 150.0
    result = optimize_budget_allocation_mpc(
        df, total_daily_budget=total_budget, horizon_days=90, replan_every_days=30,
        backtest_start=backtest_start, spend_execution_noise_std_frac=0.0, seed=0,
    )
    last_window = result["windows"][-1]
    meta_key = ("meta", "Generic")
    mpc_b_spend = last_window["mpc_allocation"][meta_key]
    ol_b_spend = last_window["open_loop_allocation"][meta_key]
    assert mpc_b_spend > ol_b_spend, (
        f"expected MPC to shift more budget toward the now-high-ROAS group B "
        f"than the frozen open-loop plan: mpc={mpc_b_spend:.1f} open_loop={ol_b_spend:.1f}"
    )


def test_mpc_beats_open_loop_on_realized_revenue_after_regime_shift():
    """The bottom-line business claim: MPC's realized revenue (scored
    against the SAME retrospective ground truth as open-loop) should beat
    open-loop's, at least in the later windows where MPC has had a chance
    to react to the new regime."""
    df, backtest_start = _regime_shift_df(pre_days=200, horizon_days=90, seed=3)
    total_budget = 150.0
    result = optimize_budget_allocation_mpc(
        df, total_daily_budget=total_budget, horizon_days=90, replan_every_days=30,
        backtest_start=backtest_start, spend_execution_noise_std_frac=0.0, seed=0,
    )
    last_window = result["windows"][-1]
    assert last_window["mpc_realized_daily_revenue"] > last_window["open_loop_realized_daily_revenue"]
    assert result["mpc_avg_daily_revenue"] > result["open_loop_avg_daily_revenue"]
    assert result["mpc_vs_open_loop_relative_lift"] > 0


def test_mpc_allocation_progressively_shifts_toward_b_across_windows():
    """A softer, monotonic-flavored check on the same mechanism: as MPC
    accumulates more evidence of the regime shift across successive
    windows, its allocation to the now-better group should not go
    backward."""
    df, backtest_start = _regime_shift_df(pre_days=200, horizon_days=90, seed=7)
    total_budget = 150.0
    result = optimize_budget_allocation_mpc(
        df, total_daily_budget=total_budget, horizon_days=90, replan_every_days=30,
        backtest_start=backtest_start, spend_execution_noise_std_frac=0.0, seed=0,
    )
    meta_key = ("meta", "Generic")
    b_spend_by_window = [w["mpc_allocation"][meta_key] for w in result["windows"]]
    assert b_spend_by_window[-1] >= b_spend_by_window[0] - 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

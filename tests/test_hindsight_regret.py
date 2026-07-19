"""
tests/test_hindsight_regret.py
================================
Unit tests for `budget_curves.compute_hindsight_regret` (§F.5) — compares
the tool's recommendation against what ACTUALLY happened historically, not
against another algorithm (that's what `optimize_budget_allocation_mpc`'s
own open-loop-vs-MPC comparison already does).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from budget_curves import optimize_budget_allocation_mpc, compute_hindsight_regret
from test_budget_mpc import _daily_rows  # reuse the existing synthetic-data helper


def _two_channel_df(rng, n_days=280):
    rows = []
    # A channel spending steadily at a mediocre, flat ROAS the whole time --
    # real historical decisions never adapt here, so a smarter allocator
    # should be able to find a better split.
    rows += _daily_rows("chan_a", "x", "2026-01-01", n_days, lambda i: 400.0, lambda i: 1.5, rng)
    rows += _daily_rows("chan_b", "x", "2026-01-01", n_days, lambda i: 400.0, lambda i: 4.0, rng)
    return pd.DataFrame(rows)


def test_regret_windows_match_mpc_report_windows_exactly():
    rng = np.random.default_rng(0)
    df = _two_channel_df(rng)
    mpc_report = optimize_budget_allocation_mpc(
        df, total_daily_budget=800.0, horizon_days=90, replan_every_days=30,
        backtest_start="2026-06-01", fit_min_points=8, seed=1,
    )
    regret = compute_hindsight_regret(df, mpc_report)
    assert [w["window_start"] for w in regret["windows"]] == [w["window_start"] for w in mpc_report["windows"]]
    assert [w["window_end"] for w in regret["windows"]] == [w["window_end"] for w in mpc_report["windows"]]
    assert [w["n_days"] for w in regret["windows"]] == [w["n_days"] for w in mpc_report["windows"]]


def test_actual_revenue_matches_a_plain_groupby_sum():
    """No estimation on the 'actual' side -- confirm it's a literal sum over
    canonical_df for the window, not routed through any curve."""
    rng = np.random.default_rng(2)
    df = _two_channel_df(rng)
    mpc_report = optimize_budget_allocation_mpc(
        df, total_daily_budget=800.0, horizon_days=30, replan_every_days=30,
        backtest_start="2026-06-01", fit_min_points=8, seed=1,
    )
    regret = compute_hindsight_regret(df, mpc_report)
    w = regret["windows"][0]
    w_start, w_end = pd.Timestamp(w["window_start"]), pd.Timestamp(w["window_end"])
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= w_start) & (df["date"] < w_end)
    expected_daily_revenue = df.loc[mask, "revenue"].sum() / w["n_days"]
    assert w["actual_daily_revenue"] == pytest.approx(expected_daily_revenue, rel=1e-9)


def test_uplift_direction_is_sane_when_one_channel_is_clearly_better():
    """chan_b's ROAS (4.0) dominates chan_a's (1.5) the whole time and
    'actual' spend here is a naive fixed 50/50 split (see _daily_rows) --
    an allocator with room to shift budget toward chan_b should show a
    positive regret (it beats the naive historical split)."""
    rng = np.random.default_rng(3)
    df = _two_channel_df(rng)
    mpc_report = optimize_budget_allocation_mpc(
        df, total_daily_budget=800.0, horizon_days=90, replan_every_days=30,
        backtest_start="2026-06-01", fit_min_points=8, seed=1,
    )
    regret = compute_hindsight_regret(df, mpc_report)
    assert regret["open_loop_vs_actual_uplift_pct"] > 0
    assert regret["mpc_vs_actual_uplift_pct"] > 0


def test_empty_windows_does_not_crash():
    empty_report = {"windows": []}
    out = compute_hindsight_regret(pd.DataFrame({"date": [], "channel": [], "revenue": [], "spend": []}), empty_report)
    assert out["windows"] == []
    assert out["actual_avg_daily_revenue"] == 0.0
    assert out["open_loop_vs_actual_uplift_pct"] is None

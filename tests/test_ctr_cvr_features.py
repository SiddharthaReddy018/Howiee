"""
tests/test_ctr_cvr_features.py
================================
Focused unit tests for `ctr_roll_28`/`cvr_roll_28` (§B.6) — the highest-risk
change in this round since it touches the shared model's own feature set.
Small, direct arithmetic checks rather than a full pipeline run (that's
covered by the existing dynamic-FEATURE_NAMES tests across the rest of the
suite, which all pass unchanged with these two columns added).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import build_daily_feature_table, NUMERIC_FEATURES


def _toy_canonical_df(n_days=40):
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "campaign_id": "camp_1",
        "channel": "google",
        "campaign_type": "Search",
        "campaign_name": "Test Campaign",
        "spend": np.full(n_days, 100.0),
        "revenue": np.full(n_days, 400.0),
        "clicks": np.full(n_days, 50.0),
        "impressions": np.full(n_days, 1000.0),
        "conversions": np.full(n_days, 5.0),
        "daily_budget": np.full(n_days, 100.0),
    })


def test_ctr_cvr_are_in_the_feature_manifest():
    assert "ctr_roll_28" in NUMERIC_FEATURES
    assert "cvr_roll_28" in NUMERIC_FEATURES


def test_ctr_cvr_computed_correctly_on_a_constant_series():
    df = _toy_canonical_df()
    daily = build_daily_feature_table(df)
    last_row = daily.iloc[-1]
    # constant 50 clicks / 1000 impressions / 5 conversions every day ->
    # rolling sums are just count * daily value, ratio should reduce to
    # the same daily ratio regardless of window length once warmed up
    assert last_row["ctr_roll_28"] == pytest_approx(50.0 / 1000.0)
    assert last_row["cvr_roll_28"] == pytest_approx(5.0 / 50.0)


def test_ctr_is_nan_not_zero_when_no_impressions():
    df = _toy_canonical_df()
    df["impressions"] = 0.0
    df["clicks"] = 0.0
    daily = build_daily_feature_table(df)
    last_row = daily.iloc[-1]
    assert pd.isna(last_row["ctr_roll_28"])  # NaN, not a fabricated 0.0


def test_cvr_is_nan_not_zero_when_no_clicks():
    df = _toy_canonical_df()
    df["clicks"] = 0.0
    df["conversions"] = 0.0
    daily = build_daily_feature_table(df)
    last_row = daily.iloc[-1]
    assert pd.isna(last_row["cvr_roll_28"])


def test_ctr_cvr_never_infinite():
    """A zero denominator must produce NaN, never +/-inf, which would
    otherwise silently poison downstream LightGBM splits or SHAP values."""
    df = _toy_canonical_df()
    df.loc[df.index[:10], "impressions"] = 0.0
    df.loc[df.index[:10], "clicks"] = 0.0
    daily = build_daily_feature_table(df)
    assert not np.isinf(daily["ctr_roll_28"].to_numpy(dtype=float)).any()
    assert not np.isinf(daily["cvr_roll_28"].to_numpy(dtype=float)).any()


def pytest_approx(x, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) <= tol
    return _Approx()

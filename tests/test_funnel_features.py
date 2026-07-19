"""
tests/test_funnel_features.py
==============================
Focused unit tests for the §B.7 extended funnel features added this round:
`cpm_roll_28`, `cpa_roll_28`, `reach_roll_sum_28`, `video_views_roll_sum_28`,
`frequency_roll_28`, `video_view_rate_roll_28` -- plus the schema_mapper §A.9
ingestion path (`reach`, `video_views`) they're built on.

Same spirit as `test_ctr_cvr_features.py`: small, direct arithmetic checks
on a toy campaign, not a full pipeline run.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import build_daily_feature_table, NUMERIC_FEATURES
from schema_mapper import ingest_dataframe, CANONICAL_COLUMNS


def _toy_canonical_df(n_days=40, with_reach=True, with_video_views=True):
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "campaign_id": "camp_1",
        "channel": "meta",
        "campaign_type": "Generic",
        "campaign_name": "Test Campaign",
        "spend": np.full(n_days, 100.0),
        "revenue": np.full(n_days, 400.0),
        "clicks": np.full(n_days, 50.0),
        "impressions": np.full(n_days, 1000.0),
        "conversions": np.full(n_days, 5.0),
        "daily_budget": np.full(n_days, 100.0),
    })
    if with_reach:
        df["reach"] = np.full(n_days, 400.0)
    if with_video_views:
        df["video_views"] = np.full(n_days, 200.0)
    return df


def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ─────────────────────────────────────────────────────────────────────────────
# schema_mapper §A.9 — reach/video_views actually get mapped, not ignored
# ─────────────────────────────────────────────────────────────────────────────
def test_reach_and_video_views_are_canonical_columns():
    assert "reach" in CANONICAL_COLUMNS
    assert "video_views" in CANONICAL_COLUMNS


def test_meta_native_reach_column_is_mapped_exactly():
    raw = pd.DataFrame({
        "campaign_id": [1, 1], "date_start": ["2026-01-01", "2026-01-02"],
        "spend": [10.0, 20.0], "conversion": [1, 2], "reach": [100.0, 150.0],
    })
    df, report = ingest_dataframe(raw, source="meta_test.csv", channel="meta")
    assert "reach" in report.mapped
    assert list(df["reach"]) == [100.0, 150.0]


def test_google_native_video_views_column_is_mapped_exactly():
    raw = pd.DataFrame({
        "campaign_id": [1, 1], "segments_date": ["2026-01-01", "2026-01-02"],
        "metrics_cost_micros": [10_000_000, 20_000_000],
        "metrics_conversions_value": [30.0, 40.0],
        "metrics_video_views": [5, 9],
    })
    df, report = ingest_dataframe(raw, source="google_test.csv", channel="google")
    assert "video_views" in report.mapped
    assert list(df["video_views"]) == [5.0, 9.0]


def test_missing_reach_column_is_nan_not_ignored_error():
    """A channel whose file never had `reach` at all (e.g. bing/google) must
    ingest cleanly with a genuinely-NaN reach column, not raise or silently
    zero it."""
    raw = pd.DataFrame({
        "campaign_id": [1, 1], "TimePeriod": ["2026-01-01", "2026-01-02"],
        "Spend": [10.0, 20.0], "Revenue": [30.0, 40.0],
    })
    df, report = ingest_dataframe(raw, source="bing_test.csv", channel="bing")
    assert "reach" in report.missing_optional
    assert df["reach"].isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# §B.7 feature manifest
# ─────────────────────────────────────────────────────────────────────────────
def test_funnel_features_are_in_the_manifest():
    for f in ("cpm_roll_28", "cpa_roll_28", "reach_roll_sum_28",
              "video_views_roll_sum_28", "frequency_roll_28", "video_view_rate_roll_28"):
        assert f in NUMERIC_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Correct arithmetic on a constant series
# ─────────────────────────────────────────────────────────────────────────────
def test_cpm_computed_correctly():
    df = _toy_canonical_df()
    daily = build_daily_feature_table(df)
    last_row = daily.iloc[-1]
    # 100 spend / 1000 impressions * 1000 = 100.0 CPM, constant regardless of window
    assert _approx(last_row["cpm_roll_28"], 100.0)


def test_cpa_computed_correctly():
    df = _toy_canonical_df()
    daily = build_daily_feature_table(df)
    last_row = daily.iloc[-1]
    # 100 spend / 5 conversions = 20.0 CPA
    assert _approx(last_row["cpa_roll_28"], 20.0)


def test_frequency_computed_correctly():
    df = _toy_canonical_df()
    daily = build_daily_feature_table(df)
    last_row = daily.iloc[-1]
    # 1000 impressions / 400 reach = 2.5x average frequency
    assert _approx(last_row["frequency_roll_28"], 2.5)


def test_video_view_rate_computed_correctly():
    df = _toy_canonical_df()
    daily = build_daily_feature_table(df)
    last_row = daily.iloc[-1]
    # 200 video views / 1000 impressions = 0.2
    assert _approx(last_row["video_view_rate_roll_28"], 0.2)


# ─────────────────────────────────────────────────────────────────────────────
# NaN (not 0, not inf) on a zero/missing denominator
# ─────────────────────────────────────────────────────────────────────────────
def test_cpm_is_nan_not_zero_when_no_impressions():
    df = _toy_canonical_df()
    df["impressions"] = 0.0
    daily = build_daily_feature_table(df)
    assert pd.isna(daily.iloc[-1]["cpm_roll_28"])


def test_cpa_is_nan_not_zero_when_no_conversions():
    df = _toy_canonical_df()
    df["conversions"] = 0.0
    daily = build_daily_feature_table(df)
    assert pd.isna(daily.iloc[-1]["cpa_roll_28"])


def test_frequency_is_nan_when_no_reach_column_at_all():
    """A channel (e.g. bing/google) that never reports reach at all must
    have a genuinely-NaN frequency, on every row -- not a fabricated value
    from an implicit reach=0."""
    df = _toy_canonical_df(with_reach=False)
    daily = build_daily_feature_table(df)
    assert daily["reach_roll_sum_28"].isna().all()
    assert daily["frequency_roll_28"].isna().all()


def test_video_view_rate_is_nan_when_no_video_views_column_at_all():
    df = _toy_canonical_df(with_video_views=False)
    daily = build_daily_feature_table(df)
    assert daily["video_views_roll_sum_28"].isna().all()
    assert daily["video_view_rate_roll_28"].isna().all()


def test_funnel_ratios_never_infinite():
    df = _toy_canonical_df()
    df.loc[df.index[:10], "impressions"] = 0.0
    df.loc[df.index[:10], "conversions"] = 0.0
    df.loc[df.index[:10], "reach"] = 0.0
    daily = build_daily_feature_table(df)
    for col in ("cpm_roll_28", "cpa_roll_28", "frequency_roll_28", "video_view_rate_roll_28"):
        assert not np.isinf(daily[col].to_numpy(dtype=float)).any(), col


# ─────────────────────────────────────────────────────────────────────────────
# Gap-fill regression (the bug this round's reindex fix addresses)
# ─────────────────────────────────────────────────────────────────────────────
def test_gap_days_do_not_fabricate_reach_for_a_channel_that_never_reports_it():
    """A campaign with a real calendar gap (e.g. paused for a few days),
    on a channel that never reports reach at all, must have NaN reach on
    the gap days too -- not a fabricated 0 that a plain "zero-fill every
    gap day" rule would otherwise introduce."""
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-10", "2026-01-11"])
    df = pd.DataFrame({
        "date": dates, "campaign_id": "camp_gap", "channel": "bing",
        "campaign_type": "Search", "campaign_name": "Gappy Campaign",
        "spend": [50.0, 60.0, 70.0, 80.0], "revenue": [10.0, 20.0, 30.0, 40.0],
        "clicks": [5.0, 6.0, 7.0, 8.0], "impressions": [100.0, 110.0, 120.0, 130.0],
        "conversions": [1.0, 1.0, 2.0, 2.0], "daily_budget": [50.0] * 4,
    })
    daily = build_daily_feature_table(df)
    # the reindexed calendar should include the gap days (Jan 3-9)
    assert len(daily) == 11
    gap_rows = daily[(daily["date"] > "2026-01-02") & (daily["date"] < "2026-01-10")]
    assert len(gap_rows) == 7
    # spend/revenue (genuinely tracked by this channel) ARE zero-filled on gap days
    assert (gap_rows["spend"] == 0.0).all()
    # reach (never tracked by this channel at all) stays NaN, even on gap days
    assert gap_rows["reach"].isna().all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

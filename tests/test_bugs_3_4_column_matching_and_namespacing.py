"""
tests/test_bugs_3_4_column_matching_and_namespacing.py
========================================================
Regression tests for the two robustness bugs found in the follow-up review
(see the reviewer notes / this round's changelog), both silent-failure modes
in the schema layer rather than crashes:

Bug 3 -- greedy fuzzy-column-matching was order-dependent, not best-match.
    `map_columns()`'s fuzzy-fallback pass used to score and assign ONE raw
    column at a time, in column order. Whichever column got processed first
    "won" a contested canonical field even if a column processed later
    scored strictly higher against it -- e.g. `Cost_Local` (score 80.8
    against `spend`) would claim the slot before `Ad_Spend_USD` (score 91.2
    against the same field) was even considered, purely because of position
    in the source CSV. Fixed by scoring every (raw column x canonical field)
    pair into a matrix up front and solving it with
    `scipy.optimize.linear_sum_assignment` (maximizing total match quality),
    with sub-threshold cells masked to a flat sentinel first so a pairing
    that could never be accepted can't outbid one that could (see the
    detailed comment in `schema_mapper.map_columns` for the concrete
    almost-regression this guards against).

Bug 4 -- campaign_id wasn't namespaced by channel before feature engineering.
    `feature_engineering.py` groups lag/rolling/expanding-statistic features
    by `campaign_id` ALONE in several places. Two different channels reusing
    the same raw platform campaign_id (plausible with small sequential
    platform IDs) would silently have their histories spliced together
    before reconciliation.py's channel-aware hierarchy keys ever got a
    chance to tell them apart. Fixed by namespacing campaign_id with its
    channel (`f"{channel}::{campaign_id}"`) once, at the schema-mapping
    source, so every downstream `groupby("campaign_id")` is automatically
    collision-safe.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema_mapper import ingest_dataframe, ingest_directory, map_columns
from feature_engineering import build_daily_feature_table


# ─────────────────────────────────────────────────────────────────────────────
# Bug 3 -- order-independent, globally-best-match fuzzy column assignment
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "column_order",
    [
        ["Cost_Local", "Ad_Spend_USD", "CampaignId", "TimePeriod", "Revenue"],
        ["Ad_Spend_USD", "Cost_Local", "CampaignId", "TimePeriod", "Revenue"],
    ],
    ids=["Cost_Local-first", "Ad_Spend_USD-first"],
)
def test_spend_slot_goes_to_the_better_scoring_column_regardless_of_order(column_order):
    """`Ad_Spend_USD` (fuzzy score 91.2 against `spend`) must win the `spend`
    slot over `Cost_Local` (score 80.8) no matter which one appears first in
    the column list -- the previous greedy, order-dependent loop let column
    position decide instead of match quality."""
    rename_map, report = map_columns(column_order)

    assert rename_map.get("Ad_Spend_USD") == "spend"
    assert "Cost_Local" not in rename_map  # the weaker match loses the contested slot
    assert report.fuzzy_mapped["spend"][0] == "Ad_Spend_USD"
    assert any(col == "Cost_Local" for col, _, _ in report.fuzzy_rejected)


def test_mangled_bing_fixture_still_recovers_campaign_name_not_campaign_type():
    """Regression guard for a real near-miss hit while building the Bug 3
    fix: a stray pandas-index-like column ('Unnamed: 0', not in first
    position in this fixture so the usual index-column drop doesn't catch
    it) scores 63.5 against `campaign_name` -- just under its 65-point
    optional threshold. A naive maximize-total-score assignment can "spend"
    campaign_name on that guaranteed-to-be-rejected pairing and bump the
    column that actually means campaign_name (`Camp_Nm`, score 88.3) down to
    `campaign_type` (67.8) instead, since it nets a higher total sum. Sub-
    threshold cells must be masked out of the assignment objective entirely,
    not just down-weighted, so this can't happen."""
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "mangled_bing.csv")
    raw = pd.read_csv(fixture, low_memory=False)
    out, report = ingest_dataframe(raw, source="mangled_bing.csv", channel="bing")

    assert report.fuzzy_mapped["campaign_name"][0] == "Camp_Nm"
    assert "campaign_type" not in report.fuzzy_mapped  # falls through to regex inference instead
    assert report.campaign_type_source == "inferred"
    # and the actual campaign_name values must be the human-readable ones, not junk
    assert out["campaign_name"].iloc[0] == "Search_TM_Campaign_02"


# ─────────────────────────────────────────────────────────────────────────────
# Bug 4 -- campaign_id namespaced by channel before it ever reaches
# feature_engineering.py's groupby("campaign_id") calls
# ─────────────────────────────────────────────────────────────────────────────
def _make_channel_frame(raw_campaign_id: str, spend: float, revenue: float, n_days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D").astype(str)
    return pd.DataFrame({
        "campaign_id": [raw_campaign_id] * n_days,
        "campaign_name": ["Some_Campaign"] * n_days,
        "date": dates,
        "spend": [spend] * n_days,
        "revenue": [revenue] * n_days,
    })


def test_same_raw_campaign_id_across_channels_gets_namespaced():
    alpha_raw = _make_channel_frame("1024", spend=10.0, revenue=100.0)
    beta_raw = _make_channel_frame("1024", spend=1000.0, revenue=9000.0)  # same raw id, different channel

    alpha, _ = ingest_dataframe(alpha_raw, source="alpha.csv", channel="alpha")
    beta, _ = ingest_dataframe(beta_raw, source="beta.csv", channel="beta")

    assert alpha["campaign_id"].iloc[0] == "alpha::1024"
    assert beta["campaign_id"].iloc[0] == "beta::1024"
    assert alpha["campaign_id"].iloc[0] != beta["campaign_id"].iloc[0]


def test_colliding_campaign_id_features_do_not_get_spliced_across_channels():
    """The actual demonstration of the bug this fix prevents: without
    channel-namespacing, `build_daily_feature_table`'s
    `groupby(["campaign_id", "date"])` aggregation step would SUM these two
    channels' spend/revenue together on every shared date, since both rows
    would carry the identical raw campaign_id '1024'."""
    alpha_raw = _make_channel_frame("1024", spend=10.0, revenue=100.0)
    beta_raw = _make_channel_frame("1024", spend=1000.0, revenue=9000.0)

    alpha, _ = ingest_dataframe(alpha_raw, source="alpha.csv", channel="alpha")
    beta, _ = ingest_dataframe(beta_raw, source="beta.csv", channel="beta")
    combined = pd.concat([alpha, beta], ignore_index=True)

    daily = build_daily_feature_table(combined)

    # must remain two distinct campaign groups, never blended into one
    assert daily["campaign_id"].nunique() == 2

    alpha_daily = daily[daily["campaign_id"] == "alpha::1024"].sort_values("date")
    beta_daily = daily[daily["campaign_id"] == "beta::1024"].sort_values("date")

    assert np.isclose(alpha_daily["spend"].iloc[10], 10.0)
    assert np.isclose(beta_daily["spend"].iloc[10], 1000.0)
    # the tell-tale sign of the old bug: a blended 10+1000=1010 row
    assert not np.isclose(alpha_daily["spend"].iloc[10], 1010.0)
    assert not np.isclose(beta_daily["spend"].iloc[10], 1010.0)


def test_ingest_directory_namespaces_campaign_id_end_to_end(tmp_path):
    """Same collision, exercised through the full directory-ingestion entry
    point (the actual path validate.py/generate_features.py use), with
    channel inferred from filename rather than passed explicitly."""
    data_dir = tmp_path / "collision_data"
    data_dir.mkdir()
    _make_channel_frame("77", spend=5.0, revenue=50.0).to_csv(data_dir / "google_ads_export.csv", index=False)
    _make_channel_frame("77", spend=500.0, revenue=4500.0).to_csv(data_dir / "meta_ads_export.csv", index=False)

    combined, _ = ingest_directory(str(data_dir))

    # filenames match the known google/meta patterns, so channel resolves to
    # those short names (not the full filename) -- the collision is on the
    # raw campaign_id "77" shared by both
    assert combined["campaign_id"].nunique() == 2
    assert set(combined["campaign_id"].unique()) == {"google::77", "meta::77"}


# ─────────────────────────────────────────────────────────────────────────────
# Bonus find while investigating "extra columns shouldn't break the model":
# an unrelated junk column can coincidentally fuzzy-match `channel`'s alias
# pool and silently overwrite every row's channel with that column's actual
# (garbage) values -- worse than a crash, since nothing downstream flags it.
# `channel` is now excluded from Pass 2 fuzzy matching (exact match only);
# it already has its own safer 3-tier fallback (explicit > native column >
# filename-derived).
# ─────────────────────────────────────────────────────────────────────────────
def test_junk_column_cannot_hijack_channel_via_fuzzy_match():
    n = 10
    df = _make_channel_frame("1", spend=1.0, revenue=1.0, n_days=n)
    # a column name that coincidentally scores well against channel's alias
    # pool ("platform", "source", "network") but whose VALUES are garbage
    df["some_new_platform_field_2026"] = "zz"

    out, report = ingest_dataframe(df, source="export_from_new_platform.csv", channel=None)

    assert "channel" not in report.fuzzy_mapped
    assert (out["channel"] == "zz").sum() == 0
    assert out["channel"].iloc[0] == "export_from_new_platform"


def test_exact_native_channel_column_still_works():
    """The fix must not break the legitimate case: a file that genuinely has
    its own `channel` column (exact name match, Pass 1) should still use it."""
    df = _make_channel_frame("1", spend=1.0, revenue=1.0, n_days=5)
    df["channel"] = "google_native"

    out, report = ingest_dataframe(df, source="some_export.csv", channel=None)

    assert out["channel"].iloc[0] == "google_native"
    assert report.mapped.get("channel") == "channel"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

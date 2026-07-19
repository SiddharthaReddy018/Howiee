"""
tests/test_campaign_consistency_validation.py
==============================================
Tests for `schema_mapper.validate_campaign_consistency` -- the brief names
"validating campaign consistency" as its own deliverable bullet, separate
from schema/type validation, and nothing previously checked for it.

Checks: duplicate (campaign_id, date) rows, a campaign_id reporting more
than one campaign_type over its history, and a campaign_id reporting more
than one campaign_name over its history. Cross-channel campaign_id
collisions don't need a separate check here -- Bug 4's channel-namespacing
fix (`channel::campaign_id`) already makes that case unrepresentable in the
canonical frame.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema_mapper import validate_campaign_consistency


def _base_frame(n=10):
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "channel": ["google"] * n,
        "campaign_id": ["google::1"] * n,
        "campaign_name": ["Search_Campaign"] * n,
        "date": dates,
        "spend": [10.0] * n,
        "revenue": [100.0] * n,
        "campaign_type": ["Search"] * n,
    })


def test_clean_frame_has_no_issues():
    df = _base_frame()
    assert validate_campaign_consistency(df) == []


def test_empty_frame_has_no_issues():
    assert validate_campaign_consistency(pd.DataFrame()) == []


def test_duplicate_campaign_id_date_detected():
    df = _base_frame()
    dup_row = df.iloc[[3]].copy()  # duplicate an existing (campaign_id, date) pair
    df2 = pd.concat([df, dup_row], ignore_index=True)

    issues = validate_campaign_consistency(df2)
    assert len(issues) == 1
    assert "duplicate (campaign_id, date)" in issues[0]


def test_inconsistent_campaign_type_detected():
    df = _base_frame()
    df.loc[df.index[-1], "campaign_type"] = "Shopping"  # same campaign_id, different type on the last day

    issues = validate_campaign_consistency(df)
    assert len(issues) == 1
    assert "campaign_type" in issues[0]
    assert "google::1" in issues[0]


def test_inconsistent_campaign_name_detected():
    df = _base_frame()
    df.loc[df.index[-1], "campaign_name"] = "Renamed_Campaign"

    issues = validate_campaign_consistency(df)
    assert len(issues) == 1
    assert "campaign_name" in issues[0]


def test_multiple_issue_types_all_reported_together():
    df = _base_frame()
    dup_row = df.iloc[[0]].copy()
    df = pd.concat([df, dup_row], ignore_index=True)
    df.loc[df.index[-1], "campaign_type"] = "Shopping"

    issues = validate_campaign_consistency(df)
    assert len(issues) == 2  # one duplicate-row issue, one campaign_type issue


def test_real_project_data_is_clean():
    """Sanity check against this project's own real, already-ingested data --
    should report zero issues (confirms the check isn't over-eager)."""
    from schema_mapper import ingest_directory
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    canonical_df, _ = ingest_directory(data_dir)
    assert validate_campaign_consistency(canonical_df) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

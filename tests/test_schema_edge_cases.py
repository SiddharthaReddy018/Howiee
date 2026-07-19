"""
tests/test_schema_edge_cases.py
================================
Regression tests for two silent-failure modes found by adversarial testing
beyond the mangled-Bing fixture in test_schema_robustness.py. Both are the
same family of bug as §3's currency-formatting fix (the documented one in
technical_documentation.md): something that LOOKS like a successful ingest
but is quietly wrong, with nothing in the report to catch it.

1. Channel-name pollution: a CSV whose filename doesn't match any of
   ingest_directory's _CHANNEL_FILE_PATTERNS previously fell back to the
   literal filename (extension included) as the channel value, e.g.
   'export_2026_q3.csv' instead of 'export_2026_q3'. That value propagates
   into every downstream channel-level breakdown, the app's channel
   selector, and the LLM grounding context's scope label.

2. All-blank mapped numeric column: a column that IS mapped (present, named
   correctly or fuzzy-recovered) but every value is blank/NaN previously
   produced zero warnings, because the existing >2%-of-nonempty-failed
   guard in _to_numeric_checked divides by n_nonempty, which is 0 for a
   fully-blank column -- the guard can never fire. The column silently
   becomes a valid-looking all-zero (or all-NaN) field with no trace in the
   ingestion report, worse than a genuinely missing column (which is at
   least logged in missing_optional).
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema_mapper import ingest_dataframe, ingest_directory

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1 — channel-name pollution on an unrecognized filename
# ─────────────────────────────────────────────────────────────────────────────
def test_unrecognized_filename_channel_has_no_extension(tmp_path):
    """A CSV whose name matches none of the bing/google/meta glob patterns
    must still produce a clean, extension-free channel value."""
    src = os.path.join(DATA_DIR, "bing_campaign_stats.csv")
    df = pd.read_csv(src)

    data_dir = tmp_path / "weird_data"
    data_dir.mkdir()
    weird_name = "export_2026_q3.csv"
    df.to_csv(data_dir / weird_name, index=False)

    combined, reports = ingest_directory(str(data_dir))

    channels = combined["channel"].unique().tolist()
    assert channels == ["export_2026_q3"], (
        f"expected clean filename-derived channel, got {channels!r} "
        f"(a literal '.csv' suffix means the extension-stripping fix regressed)"
    )
    assert not any(".csv" in str(c) for c in channels)

    report = reports[weird_name]
    assert report.channel == "export_2026_q3"
    # the warning text itself should also be extension-free
    assert any(
        "export_2026_q3" in w and ".csv" not in w.split("using '")[-1]
        for w in report.warnings
    )


def test_ingest_dataframe_direct_call_also_strips_extension():
    """Same fix, exercised via the lower-level ingest_dataframe entry point
    directly (not just through ingest_directory)."""
    df = pd.DataFrame({
        "campaign_id": ["c1"], "date": ["2026-01-01"],
        "spend": [10.0], "revenue": [50.0],
    })
    out, report = ingest_dataframe(df, source="some_weird_export.csv")
    assert out["channel"].iloc[0] == "some_weird_export"


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2 — a mapped-but-fully-blank numeric column must warn, not silently
# become a valid-looking zero
# ─────────────────────────────────────────────────────────────────────────────
def test_fully_blank_spend_column_warns_and_is_zero():
    df = pd.DataFrame({
        "campaign_id": ["c1", "c1", "c2"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-01"],
        "Spend": ["", "", ""],          # column exists, every value blank
        "Revenue": [100, 200, 300],
    })
    out, report = ingest_dataframe(df, source="blank_spend.csv")

    # value-level behavior is unchanged (still a defensible 0.0, not a crash)
    assert (out["spend"] == 0.0).all()

    # but it must now be visible in the report -- this is the actual fix
    assert any("spend" in w and "blank" in w.lower() for w in report.warnings), (
        f"expected a warning naming the fully-blank 'spend' column, got: {report.warnings!r}"
    )


def test_partially_blank_column_still_uses_original_percent_threshold():
    """Guard against over-correcting: a column with some real values and a
    few blanks should NOT trip the new all-blank warning, and should only
    trip the existing >2%-unparseable warning if genuinely warranted."""
    df = pd.DataFrame({
        "campaign_id": ["c1"] * 10,
        "date": pd.date_range("2026-01-01", periods=10).astype(str),
        "spend": [10.0, 20.0, "", 15.0, 12.0, 8.0, 9.0, 11.0, 14.0, 13.0],
        "revenue": [50.0] * 10,
    })
    out, report = ingest_dataframe(df, source="mostly_fine.csv")
    assert out["spend"].sum() > 0
    assert not any("every single value is blank" in w for w in report.warnings)


def test_fully_blank_optional_column_also_warns():
    """The same fully-blank check applies to optional numeric fields
    (e.g. clicks), not just required ones."""
    df = pd.DataFrame({
        "campaign_id": ["c1"], "date": ["2026-01-01"],
        "spend": [10.0], "revenue": [50.0],
        "clicks": [""],
    })
    out, report = ingest_dataframe(df, source="blank_clicks.csv")
    assert any("clicks" in w and "blank" in w.lower() for w in report.warnings)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

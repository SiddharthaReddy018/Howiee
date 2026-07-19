"""
tests/test_numeric_format_robustness.py
========================================
§A.4b — a numeric field that is mapped correctly BY NAME (exact or fuzzy)
must not silently become all-zero because of its number *formatting*.

Motivating bug (found by stress-testing beyond the existing mangled-schema
fixture, which only renames/drops/shuffles columns and never reformats
their *values*): a vendor export with currency-formatted spend/revenue
(`"$1,234.56"`, thousands separators, accounting-style `(123.45)` negatives,
or a stray `%`) maps onto the canonical `spend`/`revenue` columns just fine,
but the old `pd.to_numeric(..., errors="coerce")` step turned every such
value into NaN, and the very next `.fillna(0.0)` silently turned that into a
*valid-looking* zero — Pandera's `Check.ge(0)` passes trivially, the
ingestion report shows a clean full mapping, and `data_quality_report.txt`
says PASSED, while every downstream forecast for that file is computed off
zero spend and zero revenue. This is strictly worse than a missing column
(which is at least logged in `missing_optional`), and it reproduces
end-to-end through `run.sh` against the real `pickle/model.pkl` bundle:
`planned_future_daily_budget` for campaign 570837630 silently drops from
4.21 to 0.0 and its `plausibility_flag` flips from True (correctly flagged)
to False (looks "fine") on the corrupted input — i.e. the safety net gets
*less* suspicious of exactly the run it should be more suspicious of.

This file asserts the fix (`_clean_numeric_series` / `_to_numeric_checked`
in `schema_mapper.py`):
  - a currency/comma/percent/accounting-negative-formatted spend & revenue
    column parses to the SAME totals as the original clean numeric file
  - genuinely unparseable values (not a recognized format, e.g. free text)
    still degrade to NaN (never fabricated as zero) and raise a visible
    warning on the ingestion report, rather than failing silently
  - a normal, already-numeric column is untouched (no accidental double
    cleaning / no perf regression on the common case)
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema_mapper import (
    ingest_dataframe,
    _clean_numeric_series,
    _to_numeric_checked,
    IngestionReport,
)

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")


def _currency_mangle(df: pd.DataFrame) -> pd.DataFrame:
    """Reformat Spend/Revenue as currency strings; add junk columns; shuffle —
    everything else about the file (column names, order, other values) is
    otherwise untouched, isolating the effect to numeric *formatting*."""
    out = df.copy()
    out["Spend"] = out["Spend"].apply(lambda v: f"${v:,.2f}")
    out["Revenue"] = out["Revenue"].apply(lambda v: f"${v:,.2f}")
    out["extra_col_1"] = "foo"
    out["extra_col_2"] = 999
    return out


@pytest.fixture(scope="module")
def clean_bing_df() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "bing_campaign_stats.csv"))


def test_clean_numeric_series_handles_common_formats():
    s = pd.Series(["$1,234.56", "(500.00)", "12.5%", "1,000", " 42 ", "€10", "£5"])
    out = _clean_numeric_series(s)
    assert out.tolist() == pytest.approx([1234.56, -500.0, 12.5, 1000.0, 42.0, 10.0, 5.0])


def test_clean_numeric_series_is_noop_on_already_numeric_column():
    s = pd.Series([1.0, 2.5, 0.0, 3.75])
    out = _clean_numeric_series(s)
    assert out.tolist() == s.tolist()


def test_to_numeric_checked_flags_genuine_garbage_not_silent_zero():
    report = IngestionReport(source="<test>")
    s = pd.Series(["$1,234.56", "garbage_text", "42", ""])
    out = _to_numeric_checked(s, "spend", report)
    assert out.iloc[0] == pytest.approx(1234.56)
    assert pd.isna(out.iloc[1])          # unparseable -> NaN, never fabricated as 0
    assert out.iloc[2] == pytest.approx(42.0)
    assert any("spend" in w and "garbage_text" in w for w in report.warnings)


def test_currency_formatted_bing_file_matches_clean_totals(clean_bing_df):
    """The end-to-end regression check for the bug as originally found:
    ingest the currency-mangled file and the original clean file and assert
    the resulting canonical spend/revenue totals are identical — proving the
    formatting alone, not the underlying numbers, was ever the difference."""
    mangled = _currency_mangle(clean_bing_df)

    df_clean, report_clean = ingest_dataframe(clean_bing_df.copy(), source="clean.csv", channel="bing")
    df_mangled, report_mangled = ingest_dataframe(mangled, source="mangled.csv", channel="bing")

    assert df_mangled["spend"].sum() == pytest.approx(df_clean["spend"].sum())
    assert df_mangled["revenue"].sum() == pytest.approx(df_clean["revenue"].sum())
    # and it must NOT be a degenerate all-zero "pass"
    assert df_mangled["spend"].sum() > 0
    assert df_mangled["revenue"].sum() > 0
    assert not report_mangled.errors


def test_currency_formatted_file_produces_no_spurious_warning(clean_bing_df):
    """Every value in the real fixture IS a recognized currency format, so
    the parse-failure safety net should stay silent — it's a backstop for
    unrecognized formats, not a warning on every cleaned column."""
    mangled = _currency_mangle(clean_bing_df)
    _, report = ingest_dataframe(mangled, source="mangled.csv", channel="bing")
    assert not any("could not be parsed as numeric" in w for w in report.warnings)

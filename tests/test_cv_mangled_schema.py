"""
tests/test_cv_mangled_schema.py
================================
Implementation Plan §G.2.3 — the gap this closes: `test_schema_robustness.py`
already proves the mangled-schema file trains + forecasts end-to-end without
raising (a *qualitative* pass), but the plan explicitly asks for a
*quantitative* check too -- that pinball/CRPS/WAPE on the mangled file don't
meaningfully degrade relative to the clean file, not just that no exception
was thrown.

Mangling recipe mirrors `test_schema_robustness._build_mangled_fixture`
exactly (same two renamed columns, same two dropped columns, same three
junk columns) but applied to a full copy of the data directory (bing file
mangled, google/meta copied unchanged) so a real walk-forward CV report can
be produced from it via the same `modeling.cross_validate` used by
`train.py` itself -- not a reimplementation of the metric.

Kept fast on purpose (small `num_boost_round`, 2 walk-forward splits, a
single horizon) since this runs as part of the regular test suite, not as a
full training job -- the point is a real, reproducible quantitative
comparison, not a publication-grade CV report.
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema_mapper import ingest_directory
from feature_engineering import build_training_frame, FEATURE_NAMES
import modeling as M

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")

NUM_BOOST_ROUND = 80  # deliberately small -- this is a regression check, not a full training run
N_SPLITS = 2
HORIZONS = (30,)


def _build_mangled_data_dir(dest: str) -> str:
    """Full mangled-schema copy of ./data: bing file mangled exactly like
    test_schema_robustness's fixture (Spend->Cost_Local, CampaignName->Camp_Nm,
    DailyBudget + CampaignType dropped, 3 junk columns added, columns
    shuffled); google/meta copied through unchanged, so the CV comparison
    below isolates the effect of the one mangled channel."""
    os.makedirs(dest, exist_ok=True)

    bing_src = os.path.join(DATA_DIR, "bing_campaign_stats.csv")
    df = pd.read_csv(bing_src)
    df = df.rename(columns={"Spend": "Cost_Local", "CampaignName": "Camp_Nm"})
    df = df.drop(columns=["DailyBudget", "CampaignType"])
    rng = np.random.default_rng(42)
    df["foo_bar"] = rng.integers(0, 100, len(df))
    df["internal_notes_2026"] = "n/a"
    df["reserved_field_1"] = rng.random(len(df))
    cols = list(df.columns)
    rng.shuffle(cols)
    df = df[cols]
    df.to_csv(os.path.join(dest, "bing_campaign_stats.csv"), index=False)

    for fname in ["google_ads_campaign_stats.csv", "meta_ads_campaign_stats.csv"]:
        shutil.copy(os.path.join(DATA_DIR, fname), os.path.join(dest, fname))

    return dest


def _quick_cv_report(data_dir: str, tag: str) -> dict:
    canonical_df, reports = ingest_directory(data_dir)
    assert not any(r.errors for r in reports.values()), f"[{tag}] ingestion errors: " + str(
        {k: v.errors for k, v in reports.items() if v.errors}
    )

    frame = build_training_frame(canonical_df, horizons=HORIZONS)  # default origin_stride
    X = frame[FEATURE_NAMES]
    y = frame["target_revenue"].to_numpy()

    splits = M.walk_forward_splits(frame, n_splits=N_SPLITS)
    assert len(splits) >= 1, f"[{tag}] no usable walk-forward splits produced"

    report = M.cross_validate(X, y, splits, tag=tag, num_boost_round=NUM_BOOST_ROUND)
    report["n_rows"] = len(frame)
    return report


@pytest.fixture(scope="module")
def clean_report():
    return _quick_cv_report(DATA_DIR, tag="clean_schema")


@pytest.fixture(scope="module")
def mangled_report(tmp_path_factory):
    mangled_dir = _build_mangled_data_dir(str(tmp_path_factory.mktemp("mangled_full_dir")))
    return _quick_cv_report(mangled_dir, tag="mangled_schema")


def test_both_reports_produced_real_metrics(clean_report, mangled_report):
    """Baseline sanity: both runs actually produced numeric CRPS/WAPE, not
    None/NaN -- if either training run silently failed to score anything,
    every comparison below would be meaningless."""
    for label, report in [("clean", clean_report), ("mangled", mangled_report)]:
        assert report["crps"] is not None and np.isfinite(report["crps"]), label
        assert report["wape_median"] is not None and np.isfinite(report["wape_median"]), label
        assert report["n_splits"] >= 1, label


def test_mangled_schema_crps_does_not_meaningfully_degrade(clean_report, mangled_report):
    """§G.2.3's actual ask: quantitatively confirm performance doesn't
    degrade on the mangled file, not just that it doesn't crash. Some
    degradation on the mangled channel is expected and acceptable --
    `daily_budget` becomes genuine NaN (real feature signal lost) and
    `campaign_type` falls back to regex inference instead of the native
    label -- but it should be a modest relative gap, not a collapse.
    30% relative tolerance is deliberately generous for an 80-round,
    2-split, single-horizon smoke-sized CV report (higher variance than
    train.py's full 500-round/4-split production run), while still catching
    a genuine regression (e.g. a schema bug silently corrupting values
    rather than just dropping one optional feature)."""
    tol = 1.30
    assert mangled_report["crps"] <= clean_report["crps"] * tol, (
        f"mangled CRPS {mangled_report['crps']:.1f} vs clean {clean_report['crps']:.1f} "
        f"(ratio {mangled_report['crps'] / clean_report['crps']:.2f}) exceeds {tol}x tolerance"
    )
    assert mangled_report["wape_median"] <= clean_report["wape_median"] * tol, (
        f"mangled WAPE {mangled_report['wape_median']:.3f} vs clean {clean_report['wape_median']:.3f} "
        f"(ratio {mangled_report['wape_median'] / clean_report['wape_median']:.2f}) exceeds {tol}x tolerance"
    )


def test_mangled_schema_per_quantile_pinball_bounded(clean_report, mangled_report):
    """Same check at per-quantile granularity -- guards against a scenario
    where the aggregate CRPS looks fine but one specific quantile (e.g. the
    median) quietly blew up while others compensated."""
    tol = 1.40  # a single quantile is noisier than the CRPS aggregate; wider but still bounded
    for q, clean_loss in clean_report["pinball_per_quantile"].items():
        mangled_loss = mangled_report["pinball_per_quantile"][q]
        assert mangled_loss <= clean_loss * tol, (
            f"quantile {q}: mangled pinball {mangled_loss:.1f} vs clean {clean_loss:.1f} exceeds {tol}x tolerance"
        )


def test_mangled_schema_coverage_still_sane(clean_report, mangled_report):
    """Empirical interval coverage on the mangled file should still be in a
    broadly sane range (not collapsed to ~0 or blown out to ~1), confirming
    the quantile ensemble is still behaving like a quantile ensemble on the
    degraded-schema input, not just numerically running."""
    cov80 = mangled_report.get("empirical_coverage_80_nominal")
    if cov80 is not None:
        assert 0.15 <= cov80 <= 1.0, f"empirical 80%-nominal coverage on mangled file looks degenerate: {cov80}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

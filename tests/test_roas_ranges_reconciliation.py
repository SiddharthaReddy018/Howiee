"""
tests/test_roas_ranges_reconciliation.py
=========================================
Tests for the ROAS-range addition to `reconciliation.reconcile_forecast`.

The brief explicitly asks for "channel-level / campaign-type / campaign-level
ROAS ranges" as a named deliverable, alongside revenue. Before this, ROAS
only existed as a single derived point (`revenue_p50 / spend`) computed in
predict.py at the campaign level -- there was no ROAS range anywhere, and no
ROAS at all above the campaign level, since the reconciliation hierarchy
never tracked spend. This adds spend as a plain bottom-up sum through the
SAME hierarchy already used for revenue, then derives `roas_q{q}` at every
node by dividing that node's own (already-reconciled) revenue quantiles by
that node's own aggregated spend.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema_mapper import ingest_directory
from feature_engineering import build_training_frame, FEATURE_NAMES
import modeling as M
import reconciliation as R

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")


@pytest.fixture(scope="module")
def roas_fixture():
    canonical_df, _ = ingest_directory(DATA_DIR)
    frame = build_training_frame(canonical_df, horizons=(30,)).reset_index(drop=True)

    splits = M.walk_forward_splits(frame, n_splits=3)
    oof = M.generate_oof_median_predictions(frame, FEATURE_NAMES, splits, num_boost_round=80)
    oof["horizon_days"] = 30
    hist_long = R.build_calibration_long(oof)

    latest = frame.sort_values("origin_date").drop_duplicates("campaign_id", keep="last")
    X_latest = latest[FEATURE_NAMES]
    models, _ = M.train_quantile_ensemble(
        frame[FEATURE_NAMES], frame["target_revenue"].to_numpy(), splits[-1],
        quantiles=M.QUANTILES, num_boost_round=80, verbose=False,
    )
    q_preds = M.predict_quantiles(models, X_latest, quantiles=M.QUANTILES)
    q_preds = M.fix_quantile_crossing(np.clip(q_preds, 0, None))

    live_bottom = latest[["campaign_id", "channel", "campaign_type"]].copy()
    for i, q in enumerate(M.QUANTILES):
        live_bottom[f"q{q}"] = q_preds[:, i]
    # a plausible assumed-spend scenario: recent daily pace * horizon, with
    # one campaign forced to exactly 0 spend to test the divide-by-zero guard
    rng = np.random.default_rng(0)
    live_bottom["spend"] = rng.uniform(50, 5000, size=len(live_bottom)) * 30
    live_bottom.iloc[0, live_bottom.columns.get_loc("spend")] = 0.0

    return hist_long, live_bottom


@pytest.fixture(scope="module")
def reconciled_with_spend(roas_fixture):
    hist_long, live_bottom = roas_fixture
    return R.reconcile_forecast(hist_long, live_bottom, quantiles=M.QUANTILES)


def test_spend_column_present_and_coherent(reconciled_with_spend):
    df, _ = reconciled_with_spend
    assert "spend" in df.columns

    total_spend = df.loc[df["unique_id"] == "total", "spend"].iloc[0]
    channel_spend_sum = df[df["level"] == "channel"]["spend"].sum()
    campaign_spend_sum = df[df["level"] == "campaign"]["spend"].sum()
    assert total_spend == pytest.approx(channel_spend_sum, rel=1e-9)
    assert total_spend == pytest.approx(campaign_spend_sum, rel=1e-9)


def test_roas_columns_present_at_every_level(reconciled_with_spend):
    df, _ = reconciled_with_spend
    roas_cols = [f"roas_q{q}" for q in M.QUANTILES] + ["roas_reconciled_median"]
    for col in roas_cols:
        assert col in df.columns
    # every level named in the brief ("channel-level / campaign-type /
    # campaign-level ROAS ranges") must actually have non-null ROAS values
    for level in ("total", "channel", "campaign_type", "campaign"):
        sub = df[df["level"] == level]
        assert len(sub) > 0
        assert sub["roas_reconciled_median"].notna().any()


def test_roas_quantiles_are_monotonic(reconciled_with_spend):
    """Dividing every revenue quantile by the SAME (positive) spend value
    must preserve their order -- no separate crossing-fix should be needed."""
    df, _ = reconciled_with_spend
    roas_cols = [f"roas_q{q}" for q in M.QUANTILES]
    non_zero_spend = df[df["spend"] > 0]
    vals = non_zero_spend[roas_cols].to_numpy()
    assert np.all(np.diff(vals, axis=1) >= -1e-9)


def test_roas_matches_manual_division(reconciled_with_spend):
    """roas_q{q} must equal revenue q{q} / spend exactly (same node)."""
    df, _ = reconciled_with_spend
    row = df.loc[df["unique_id"] == "total"].iloc[0]
    for q in M.QUANTILES:
        expected = row[f"q{q}"] / row["spend"]
        assert row[f"roas_q{q}"] == pytest.approx(expected, rel=1e-9)


def test_zero_spend_node_gives_nan_not_inf_or_crash(roas_fixture):
    hist_long, live_bottom = roas_fixture
    # zero out ALL spend to force the total node itself to zero spend
    zeroed = live_bottom.copy()
    zeroed["spend"] = 0.0
    df, _ = R.reconcile_forecast(hist_long, zeroed, quantiles=M.QUANTILES)
    total_row = df.loc[df["unique_id"] == "total"].iloc[0]
    assert total_row["spend"] == 0.0
    assert np.isnan(total_row["roas_reconciled_median"])
    assert not np.isinf(total_row["roas_reconciled_median"])


def test_backward_compatible_without_spend_column():
    """reconcile_forecast must work exactly as before (no ROAS columns at
    all) when live_bottom has no `spend` column -- an old caller shouldn't
    break, and shouldn't silently get fabricated ROAS numbers either."""
    canonical_df, _ = ingest_directory(DATA_DIR)
    frame = build_training_frame(canonical_df, horizons=(30,)).reset_index(drop=True)
    splits = M.walk_forward_splits(frame, n_splits=3)
    oof = M.generate_oof_median_predictions(frame, FEATURE_NAMES, splits, num_boost_round=60)
    oof["horizon_days"] = 30
    hist_long = R.build_calibration_long(oof)

    latest = frame.sort_values("origin_date").drop_duplicates("campaign_id", keep="last")
    live_bottom = latest[["campaign_id", "channel", "campaign_type"]].copy()
    for q in M.QUANTILES:
        live_bottom[f"q{q}"] = latest["target_revenue"].to_numpy() * (0.5 + q / 2)
    live_bottom = live_bottom.sort_values("campaign_id")

    df, diag = R.reconcile_forecast(hist_long, live_bottom, quantiles=M.QUANTILES)
    assert "spend" not in df.columns
    assert not any(c.startswith("roas_") for c in df.columns)
    assert diag["max_abs_coherence_error"] < 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

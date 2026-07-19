"""
tests/test_reconciled_calibration_by_level.py
===============================================
§E re-audit — tests for `reconciliation.evaluate_reconciled_calibration`.

Coherence (max_abs_coherence_error == 0.0, already tested in
test_reconciliation_probabilistic.py) and calibration are separate
properties (Principato et al. 2024). This file checks the NEW pure-
evaluation function that measures the second one: does the reconciled
quantile band at the total/channel/campaign_type/campaign level still hit
its nominal coverage, pooled across many holdout snapshots — not just at
the base per-campaign-row level §6 already checks.

Two kinds of test:
  1. A real-data integration fixture (small scale, low boost rounds, a
     handful of holdout dates — a correctness/wiring check, not a quality
     benchmark), confirming the function runs end-to-end against the actual
     production `reconcile_forecast` path and returns a sane structure.
  2. A fully synthetic, controlled unit test where the true calibration is
     known by construction, so the reported empirical coverage can be
     checked against ground truth rather than just "did it run".
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
def calibration_fixture():
    """Real data, one horizon, a small quantile ensemble, and ONLY the last
    8 origin_dates held out (evaluate_reconciled_calibration calls
    reconcile_forecast once per origin_date -- capping this keeps the test
    fast without changing what's being tested)."""
    canonical_df, _ = ingest_directory(DATA_DIR)
    frame = build_training_frame(canonical_df, horizons=(30,)).reset_index(drop=True)

    dates_sorted = sorted(frame["origin_date"].unique())
    holdout_dates = set(dates_sorted[-8:])
    train_frame = frame[~frame["origin_date"].isin(holdout_dates)].reset_index(drop=True)
    holdout_frame = frame[frame["origin_date"].isin(holdout_dates)].reset_index(drop=True)

    splits = M.walk_forward_splits(train_frame, n_splits=3)
    oof = M.generate_oof_median_predictions(train_frame, FEATURE_NAMES, splits, num_boost_round=80)
    oof["horizon_days"] = 30
    oof_by_horizon = {30: oof}

    models, _ = M.train_quantile_ensemble(
        train_frame[FEATURE_NAMES], train_frame["target_revenue"].to_numpy(), splits[-1],
        quantiles=M.QUANTILES, num_boost_round=80, verbose=False,
    )
    q_preds = M.predict_quantiles(models, holdout_frame[FEATURE_NAMES], quantiles=M.QUANTILES)
    q_preds = M.fix_quantile_crossing(np.clip(q_preds, 0, None))

    holdout_df = holdout_frame[
        ["campaign_id", "channel", "campaign_type", "origin_date", "horizon_days", "target_revenue"]
    ].copy()
    for i, q in enumerate(M.QUANTILES):
        holdout_df[f"q{q}"] = q_preds[:, i]

    return holdout_df, oof_by_horizon


def test_runs_end_to_end_against_the_real_production_path(calibration_fixture):
    holdout_df, oof_by_horizon = calibration_fixture
    result = R.evaluate_reconciled_calibration(holdout_df, oof_by_horizon, quantiles=M.QUANTILES)
    assert result["n_snapshots"] > 0
    assert result["n_snapshots"] == holdout_df["origin_date"].nunique() - result["n_snapshots_skipped"]


def test_all_four_hierarchy_levels_are_reported(calibration_fixture):
    holdout_df, oof_by_horizon = calibration_fixture
    result = R.evaluate_reconciled_calibration(holdout_df, oof_by_horizon, quantiles=M.QUANTILES)
    for level in ("total", "channel", "campaign_type", "campaign"):
        assert level in result
        assert result[level]["n_observations"] > 0
        # same shape modeling.reliability_diagram always returns
        for band in ("90%", "80%", "50%"):
            assert band in result[level]
            assert 0.0 <= result[level][band]["empirical"] <= 1.0
            assert result[level][band]["nominal"] == pytest.approx({"90%": 0.9, "80%": 0.8, "50%": 0.5}[band])


def test_total_level_has_exactly_one_observation_per_snapshot(calibration_fixture):
    """Unlike channel/campaign_type/campaign, there is exactly one 'total'
    node per reconciled snapshot -- a direct sanity check on the pooling
    logic, not just "some number came back"."""
    holdout_df, oof_by_horizon = calibration_fixture
    result = R.evaluate_reconciled_calibration(holdout_df, oof_by_horizon, quantiles=M.QUANTILES)
    assert result["total"]["n_observations"] == result["n_snapshots"]


def test_no_crash_on_empty_holdout():
    empty = pd.DataFrame(columns=["campaign_id", "channel", "campaign_type", "origin_date",
                                   "horizon_days", "target_revenue"] + [f"q{q}" for q in M.QUANTILES])
    result = R.evaluate_reconciled_calibration(empty, {30: pd.DataFrame()}, quantiles=M.QUANTILES)
    assert result["n_snapshots"] == 0
    for level in ("total", "channel", "campaign_type", "campaign"):
        assert result[level]["n_observations"] == 0


def test_synthetic_perfectly_calibrated_bands_report_near_nominal_coverage():
    """Fully controlled ground truth: build a 2-channel/1-total hierarchy
    where each node's TRUE data-generating quantiles are known exactly (a
    normal distribution with a known std), feed reconcile_forecast bands
    built directly from that same normal distribution's quantiles (so the
    band SHOULD be well-calibrated by construction), and confirm the
    function reports empirical coverage close to nominal -- not just
    structurally valid output, but numerically correct given a known-good
    input."""
    from scipy import stats

    rng = np.random.default_rng(7)
    quantiles = M.QUANTILES
    n_days = 400
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")

    channels = {"chA": 1000.0, "chB": 2000.0}
    sigma_frac = 0.15  # true relative std for each channel's daily revenue

    oof_rows = []
    for ch, mu in channels.items():
        y = mu + rng.normal(0, mu * sigma_frac, size=n_days)
        for d, yv in zip(dates, y):
            oof_rows.append({
                "campaign_id": f"{ch}::only", "channel": ch, "campaign_type": "only",
                "origin_date": d, "target_revenue": yv, "pred_median": mu,
            })
    oof = pd.DataFrame(oof_rows)
    oof_by_horizon = {30: oof}

    # Holdout: a handful of FRESH days from the exact same generating
    # process, with bottom-level quantile bands built directly from the
    # known-true normal distribution (mu, mu*sigma_frac) -- i.e. genuinely
    # well-calibrated inputs by construction, not fitted/estimated.
    n_holdout_days = 20
    holdout_rows = []
    for d in pd.date_range("2026-06-01", periods=n_holdout_days, freq="D"):
        for ch, mu in channels.items():
            row = {"campaign_id": f"{ch}::only", "channel": ch, "campaign_type": "only",
                   "origin_date": d, "horizon_days": 30,
                   "target_revenue": mu + rng.normal(0, mu * sigma_frac)}
            for q in quantiles:
                row[f"q{q}"] = stats.norm.ppf(q, loc=mu, scale=mu * sigma_frac)
            holdout_rows.append(row)
    holdout_df = pd.DataFrame(holdout_rows)

    result = R.evaluate_reconciled_calibration(holdout_df, oof_by_horizon, quantiles=quantiles)
    assert result["n_snapshots"] == n_holdout_days

    # The channel level's bottom-node bands come straight from the true
    # distribution (only reconciliation-internal smoothing touches them) --
    # empirical coverage should land reasonably close to nominal, not be
    # wildly off (e.g. 50% nominal reporting 5% or 95% empirical would
    # indicate a real bug in the pooling/indexing logic).
    for band, nominal in (("90%", 0.9), ("80%", 0.8), ("50%", 0.5)):
        emp = result["channel"][band]["empirical"]
        assert abs(emp - nominal) < 0.35, f"{band}: empirical={emp} nominal={nominal} (n={n_holdout_days*2})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

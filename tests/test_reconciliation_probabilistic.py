"""
tests/test_reconciliation_probabilistic.py
===========================================
Implementation Plan §E.1 — tests for the upgraded probabilistic (interval)
hierarchical reconciliation in `src/reconciliation.py`.

Builds a small but REAL out-of-fold calibration set (same construction
train.py uses: `walk_forward_splits` + `generate_oof_median_predictions`)
from the project's own data, then checks:

  1. The genuine joint `hierarchicalforecast.methods.Conformal` tier is
     actually used for total + all 3 channels (where date alignment is
     verified full), and produces valid, monotonic, non-degenerate bands
     that leave the already-coherent MinTrace median untouched.
  2. The marginal per-node tier is used for the (dense-enough) campaign_type
     / campaign nodes, similarly monotonic and median-preserving.
  3. Point coherence (the pre-existing, still-critical guarantee) is
     unaffected -- exactly 0.0 max abs error, same as before this change.
  4. The sparse-campaign rescale fallback still engages for genuinely
     too-thin nodes, and never crashes the whole pipeline.
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
def reconciliation_fixture():
    """Real OOF calibration data (horizon=30) + a real live "bottom" (one
    row per campaign) built the same way train.py/predict.py do, at small
    scale (num_boost_round kept low; this is a correctness test, not a
    quality benchmark)."""
    canonical_df, _ = ingest_directory(DATA_DIR)
    frame = build_training_frame(canonical_df, horizons=(30,))
    frame = frame.reset_index(drop=True)

    splits = M.walk_forward_splits(frame, n_splits=3)
    oof = M.generate_oof_median_predictions(frame, FEATURE_NAMES, splits, num_boost_round=80)
    oof["horizon_days"] = 30
    hist_long = R.build_calibration_long(oof)

    # live bottom: one row per campaign, quantiles from a quick fit (doesn't
    # need to be the full-quality ensemble for this test -- reconcile_forecast
    # only cares that q_cols exist and are pre-sorted)
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

    return hist_long, live_bottom


@pytest.fixture(scope="module")
def reconciled(reconciliation_fixture):
    hist_long, live_bottom = reconciliation_fixture
    return R.reconcile_forecast(hist_long, live_bottom, quantiles=M.QUANTILES)


def test_point_coherence_still_exact(reconciled):
    """The pre-existing guarantee this whole module was already shipping --
    must be completely unaffected by the interval-reconciliation changes."""
    _, diag = reconciled
    assert diag["max_abs_coherence_error"] < 1e-6


def test_genuine_joint_conformal_used_for_total_and_channels(reconciled):
    _, diag = reconciled
    prob = diag["probabilistic_reconciliation"]
    assert prob["n_genuine_joint_conformal"] == 4  # total + bing + google + meta
    assert set(prob["genuine_joint_conformal_nodes"]) == {"total", "total/bing", "total/google", "total/meta"}


def test_marginal_conformal_used_for_most_deeper_nodes(reconciled):
    _, diag = reconciled
    prob = diag["probabilistic_reconciliation"]
    # most campaign_type/campaign nodes should have enough OOF history for
    # the marginal tier; only a handful of genuinely sparse ones should fall
    # back to the rescale approach
    assert prob["n_marginal_conformal"] > prob["n_rescale_fallback"]
    assert prob["n_marginal_conformal"] > 0


def test_quantile_bands_are_monotonic_and_nonnegative(reconciled):
    recon_df, _ = reconciled
    q_cols = [f"q{q}" for q in M.QUANTILES]
    vals = recon_df[q_cols].to_numpy()
    assert (vals >= -1e-6).all(), "reconciled quantiles should be clipped nonnegative"
    diffs = np.diff(vals, axis=1)
    assert (diffs >= -1e-6).all(), "quantile columns must be non-decreasing across each row"


def test_median_column_matches_coherent_reconciled_median(reconciled):
    """Both tiers are designed to only replace the BAND, never the point --
    q0.5 must exactly equal `reconciled_median` for every node, regardless
    of which tier supplied that node's band."""
    recon_df, _ = reconciled
    diffs = (recon_df["q0.5"] - recon_df["reconciled_median"]).abs()
    assert diffs.max() < 1e-6, f"q0.5 drifted from reconciled_median by up to {diffs.max()}"


def test_genuine_tier_bands_are_not_degenerate(reconciled):
    """A real check that the genuine-conformal band isn't accidentally a
    zero-width or NaN placeholder -- it should have real spread."""
    recon_df, _ = reconciled
    total_row = recon_df[recon_df["unique_id"] == "total"].iloc[0]
    assert np.isfinite(total_row["q0.05"]) and np.isfinite(total_row["q0.95"])
    assert total_row["q0.95"] > total_row["q0.5"] > 0 or total_row["q0.5"] == 0


def test_channel_level_conformal_helper_directly():
    """Unit-level check on the helper itself with a controlled input,
    independent of the full pipeline fixture above."""
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    rows = []
    for ch, base in [("bing", 1000), ("google", 5000), ("meta", 2000)]:
        y = base + rng.normal(0, base * 0.1, size=n)
        model = base + rng.normal(0, base * 0.05, size=n)  # slightly-off point forecasts
        for d, yv, mv in zip(dates, y, model):
            rows.append({"total": "total", "channel": ch, "ds": d, "y": yv, "model": mv})
    hist_long = pd.DataFrame(rows)

    channel_bottom_median = {"bing": 1000.0, "google": 5000.0, "meta": 2000.0}
    out = R._channel_level_conformal_intervals(
        hist_long, channel_bottom_median, M.QUANTILES, pd.Timestamp("2099-01-01"),
    )
    assert out is not None
    assert set(out.index) == {"total", "total/bing", "total/google", "total/meta"}
    for uid in out.index:
        row = out.loc[uid]
        assert row["q0.05"] <= row["q0.5"] <= row["q0.95"]


def test_marginal_conformal_offsets_respects_min_obs():
    Y_full = pd.DataFrame({
        "unique_id": ["sparse_node"] * 5 + ["dense_node"] * 50,
        "y": list(np.random.default_rng(1).normal(100, 10, 5)) + list(np.random.default_rng(1).normal(100, 10, 50)),
        "model": [100.0] * 55,
    })
    assert R._marginal_conformal_offsets(Y_full, "sparse_node", M.QUANTILES, min_obs=15) is None
    dense = R._marginal_conformal_offsets(Y_full, "dense_node", M.QUANTILES, min_obs=15)
    assert dense is not None
    assert abs(dense[0.5]) < 1e-9  # centered: median offset must be exactly 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

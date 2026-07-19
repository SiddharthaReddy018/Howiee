"""
tests/test_catboost_ensemble.py
================================
Unit tests for the §C.1c CatBoost point-model candidate
(`train_catboost_point_model` / `predict_catboost`) and the generalized
N-candidate + blend comparison (`compare_point_models_pinball_multi`) that
decides which point model actually ships.

Uses the same small real-data sample as `test_modeling_extensions.py`
(`modeling.py`'s training helpers hardcode this project's exact
CATEGORICAL_FEATURES/FEATURE_NAMES schema, so a bespoke synthetic column
set isn't a drop-in here either).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import modeling as M
from schema_mapper import ingest_directory
from feature_engineering import build_training_frame, FEATURE_NAMES

pytest.importorskip("catboost")

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")


@pytest.fixture(scope="module")
def synthetic_data():
    canonical_df, _ = ingest_directory(DATA_DIR)
    frame = build_training_frame(canonical_df, horizons=(30,))
    frame = frame.sample(n=min(1500, len(frame)), random_state=7).reset_index(drop=True)
    X = frame[FEATURE_NAMES]
    y = frame["target_revenue"].to_numpy()
    n = len(X)
    tr_idx = np.arange(0, int(n * 0.7))
    va_idx = np.arange(int(n * 0.7), n)
    return X, y, (tr_idx, va_idx)


# ─────────────────────────────────────────────────────────────────────────────
# §C.1c — CatBoost point model
# ─────────────────────────────────────────────────────────────────────────────
def test_train_catboost_point_model_returns_expected_tuple(synthetic_data):
    X, y, split = synthetic_data
    model, best_p, sweep_log = M.train_catboost_point_model(X, y, split, num_boost_round=80, verbose=False)
    assert model is not None
    assert best_p in M.TWEEDIE_VARIANCE_POWERS
    assert set(sweep_log.keys()) == set(M.TWEEDIE_VARIANCE_POWERS)
    assert all(v >= 0 for v in sweep_log.values())


def test_predict_catboost_is_nonnegative_and_reasonable(synthetic_data):
    X, y, split = synthetic_data
    model, _, _ = M.train_catboost_point_model(X, y, split, num_boost_round=80, verbose=False)
    tr_idx, va_idx = split
    pred = M.predict_catboost(model, X.iloc[va_idx])
    assert (pred >= 0).all()
    assert pred.shape == (len(va_idx),)
    # sanity: predictions should correlate with actuals better than a constant guess
    mae_model = np.mean(np.abs(pred - y[va_idx]))
    mae_mean_baseline = np.mean(np.abs(np.mean(y[tr_idx]) - y[va_idx]))
    assert mae_model < mae_mean_baseline * 1.5, (
        "catboost point model should be at least roughly competitive with a naive mean baseline"
    )


def test_catboost_sweep_picks_lowest_deviance_power(synthetic_data):
    X, y, split = synthetic_data
    _, best_p, sweep_log = M.train_catboost_point_model(X, y, split, num_boost_round=80, verbose=False)
    assert sweep_log[best_p] == min(sweep_log.values())


def test_catboost_multi_fold_sweep_matches_single_fold_call_shape(synthetic_data):
    """train_catboost_point_model must accept a LIST of splits (multi-fold
    averaged sweep) the same way train_point_model does, not just a single
    tuple -- same calling convention, same docstring claim of apples-to-
    apples comparison."""
    X, y, split = synthetic_data
    tr_idx, va_idx = split
    mid = len(tr_idx) // 2
    splits = [(tr_idx[:mid], tr_idx[mid:]), (tr_idx[mid:], tr_idx[:mid])]
    model, best_p, sweep_log = M.train_catboost_point_model(X, y, splits, num_boost_round=60, verbose=False)
    assert model is not None
    assert best_p in M.TWEEDIE_VARIANCE_POWERS


# ─────────────────────────────────────────────────────────────────────────────
# §C.1 — generalized multi-candidate + blend comparison
# ─────────────────────────────────────────────────────────────────────────────
def test_compare_multi_picks_best_of_three_plus_blend():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 1000, 500)
    # one clearly-best candidate, two noisier ones
    good_pred = y_true + rng.normal(0, 5, 500)
    noisy_pred_a = y_true + rng.normal(0, 200, 500)
    noisy_pred_b = y_true + rng.normal(0, 250, 500)

    report = M.compare_point_models_pinball_multi(
        y_true, {"tweedie": noisy_pred_a, "hurdle": noisy_pred_b, "catboost": good_pred},
    )
    assert report["winner"] == "catboost"
    assert set(report["pinball_q50_by_candidate"].keys()) == {
        "tweedie", "hurdle", "catboost", "blend_equal_weight",
    }
    assert report["relative_improvement_of_winner_vs_tweedie"] > 0


def test_compare_multi_blend_can_win_when_errors_are_uncorrelated():
    """The concrete point of the blend candidate: two candidates that are
    EACH about equally noisy, but wrong in roughly opposite, canceling
    directions, should let the blend beat both individually. (The blend
    scored by `compare_point_models_pinball_multi` averages EVERY candidate
    passed in, so this only isolates the cancellation effect cleanly with
    exactly the two canceling candidates -- a third, much-noisier candidate
    would dilute the average back down, which is itself the honest,
    expected behavior of an unweighted blend, not a bug.)"""
    rng = np.random.default_rng(1)
    n = 2000
    y_true = rng.uniform(100, 1000, n)
    # candidate A overshoots by a random amount, candidate B undershoots by
    # roughly the same amount on average -> their average is much closer to
    # y_true than either one alone
    noise = rng.normal(0, 80, n)
    pred_a = y_true + noise + rng.normal(0, 5, n)
    pred_b = y_true - noise + rng.normal(0, 5, n)

    report = M.compare_point_models_pinball_multi(y_true, {"tweedie": pred_a, "hurdle": pred_b})
    assert report["winner"] == "blend_equal_weight"
    blend_loss = report["pinball_q50_by_candidate"]["blend_equal_weight"]
    assert blend_loss < report["pinball_q50_by_candidate"]["tweedie"]
    assert blend_loss < report["pinball_q50_by_candidate"]["hurdle"]


def test_compare_multi_requires_at_least_two_candidates():
    with pytest.raises(ValueError):
        M.compare_point_models_pinball_multi(np.array([1.0, 2.0]), {"tweedie": np.array([1.0, 2.0])})


def test_compare_multi_end_to_end_three_real_candidates(synthetic_data):
    """Same call sequence train.py now runs for real: fit all three
    candidates on the same train split, compare on the same held-out slice."""
    X, y, split = synthetic_data
    tr_idx, va_idx = split
    X_tr, y_tr = X.iloc[tr_idx].reset_index(drop=True), y[tr_idx]
    tuning_split = (np.arange(0, int(len(X_tr) * 0.8)), np.arange(int(len(X_tr) * 0.8), len(X_tr)))

    tweedie_model, _, _ = M.train_point_model(X_tr, y_tr, tuning_split, num_boost_round=80, verbose=False)
    hurdle_models = M.train_hurdle_model(X_tr, y_tr, tuning_split, num_boost_round=80, verbose=False)
    catboost_model, _, _ = M.train_catboost_point_model(X_tr, y_tr, tuning_split, num_boost_round=80, verbose=False)

    X_va, y_va = X.iloc[va_idx], y[va_idx]
    preds = {
        "tweedie": np.clip(tweedie_model.predict(X_va), 0, None),
        "hurdle": M.predict_hurdle(hurdle_models, X_va),
        "catboost": M.predict_catboost(catboost_model, X_va),
    }
    report = M.compare_point_models_pinball_multi(y_va, preds)
    assert report["winner"] in ("tweedie", "hurdle", "catboost", "blend_equal_weight")
    assert all(v > 0 for v in report["pinball_q50_by_candidate"].values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

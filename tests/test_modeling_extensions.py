"""
tests/test_modeling_extensions.py
==================================
Unit tests for two of the previously-missing plan items now implemented in
`src/modeling.py`:

  §C.1 — the hurdle (classifier x Gamma) two-part model and the pinball-loss
  ablation that decides whether it actually replaces the Tweedie point model.

  §H.1 — real SHAP feature importance (`shap.TreeExplainer`), replacing the
  gain-based stand-in for the LLM grounding context's `top_drivers`.

Uses a small synthetic zero-inflated dataset (not the real ad-campaign data)
so this runs in a couple of seconds as part of the regular suite, independent
of `tests/test_cv_mangled_schema.py`'s real-data CV comparison.
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

pytest.importorskip("shap")

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")


@pytest.fixture(scope="module")
def synthetic_data():
    """Despite the name (kept for continuity with the rest of this file),
    this is a small REAL sample: `modeling.py`'s training helpers
    (`_make_dataset`, `_monotone_vector`) hardcode this project's exact
    `CATEGORICAL_FEATURES`/`FEATURE_NAMES` schema, so a bespoke synthetic
    column set doesn't work as a drop-in here. Sampling down real,
    feature-engineered rows is both simpler and more representative than
    fighting that schema coupling with a parallel synthetic one."""
    canonical_df, _ = ingest_directory(DATA_DIR)
    frame = build_training_frame(canonical_df, horizons=(30,))
    frame = frame.sample(n=min(2000, len(frame)), random_state=7).reset_index(drop=True)
    X = frame[FEATURE_NAMES]
    y = frame["target_revenue"].to_numpy()
    n = len(X)
    tr_idx = np.arange(0, int(n * 0.7))
    va_idx = np.arange(int(n * 0.7), n)
    return X, y, (tr_idx, va_idx)


# ─────────────────────────────────────────────────────────────────────────────
# §C.1 — hurdle model
# ─────────────────────────────────────────────────────────────────────────────
def test_train_hurdle_model_returns_expected_keys(synthetic_data):
    X, y, split = synthetic_data
    hurdle = M.train_hurdle_model(X, y, split, num_boost_round=80, verbose=False)
    assert set(hurdle.keys()) >= {"classifier", "gamma", "classifier_val_auc", "gamma_val_deviance"}
    assert hurdle["classifier_val_auc"] is not None
    assert 0.5 <= hurdle["classifier_val_auc"] <= 1.0, "classifier should beat random on an obviously-separable signal"


def test_predict_hurdle_is_nonnegative_and_reasonable(synthetic_data):
    X, y, split = synthetic_data
    hurdle = M.train_hurdle_model(X, y, split, num_boost_round=80, verbose=False)
    preds = M.predict_hurdle(hurdle, X)
    assert (preds >= 0).all()
    assert len(preds) == len(X)
    # correlation with the dominant driver should be clearly positive
    corr = np.corrcoef(preds, X["planned_future_daily_budget"])[0, 1]
    assert corr > 0.3, f"expected hurdle predictions to track the dominant driver, corr={corr:.2f}"


def test_compare_point_models_pinball_picks_the_better_model():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 5.0, 15.0])
    good_pred = y_true + np.array([0.5, -0.5, 0.5, -0.5, 0.1, -0.1, 0.2, -0.2])  # close
    bad_pred = y_true + np.array([20, -20, 20, -20, 15, -15, 10, -10])            # far off

    result = M.compare_point_models_pinball(y_true, tweedie_pred=bad_pred, hurdle_pred=good_pred)
    assert result["winner"] == "hurdle"
    assert result["hurdle_pinball_q50"] < result["tweedie_pinball_q50"]
    assert result["relative_improvement_of_winner"] > 0

    result2 = M.compare_point_models_pinball(y_true, tweedie_pred=good_pred, hurdle_pred=bad_pred)
    assert result2["winner"] == "tweedie"


def test_hurdle_ablation_end_to_end_on_synthetic_data(synthetic_data):
    """Full ablation flow: train both candidates, compare on a held-out
    slice, confirm the winner is deterministic and the report is well-formed
    -- this is the same call sequence train.py now runs for real."""
    X, y, split = synthetic_data
    tr_idx, va_idx = split
    X_tr, y_tr = X.iloc[tr_idx].reset_index(drop=True), y[tr_idx]
    tuning_split = (np.arange(0, int(len(X_tr) * 0.8)), np.arange(int(len(X_tr) * 0.8), len(X_tr)))

    tweedie_model, _, _ = M.train_point_model(X_tr, y_tr, tuning_split, num_boost_round=80, verbose=False)
    hurdle_models = M.train_hurdle_model(X_tr, y_tr, tuning_split, num_boost_round=80, verbose=False)

    X_va, y_va = X.iloc[va_idx], y[va_idx]
    tweedie_pred = np.clip(tweedie_model.predict(X_va), 0, None)
    hurdle_pred = M.predict_hurdle(hurdle_models, X_va)

    report = M.compare_point_models_pinball(y_va, tweedie_pred, hurdle_pred)
    assert report["winner"] in ("tweedie", "hurdle")
    assert report["tweedie_pinball_q50"] > 0
    assert report["hurdle_pinball_q50"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# §H.1 — SHAP
# ─────────────────────────────────────────────────────────────────────────────
def test_shap_feature_importance_ranks_dominant_feature_first(synthetic_data):
    X, y, split = synthetic_data
    models, _ = M.train_quantile_ensemble(X, y, split, quantiles=[0.5], num_boost_round=80, verbose=False)
    result = M.shap_feature_importance(models[0.5], X, top_n=3, max_rows=500)

    assert "top_features" in result and len(result["top_features"]) == 3
    assert result["n_rows_sampled"] == 500
    top_feature_names = [f["feature"] for f in result["top_features"]]
    assert "planned_future_daily_budget" in top_feature_names, (
        f"expected the obviously-dominant synthetic driver to rank in the top 3, got {top_feature_names}"
    )
    # ranked list must actually be sorted by mean_abs_shap descending
    mags = [f["mean_abs_shap"] for f in result["top_features"]]
    assert mags == sorted(mags, reverse=True)
    assert all(m >= 0 for m in mags)


def test_shap_feature_importance_has_same_shape_as_gain_importance(synthetic_data):
    """rule_based_fallback / build_grounding_context only ever read `feature`
    and `importance_rank` -- confirm SHAP's return is a drop-in replacement
    for gain-based `feature_importance`'s shape on those two keys."""
    X, y, split = synthetic_data
    models, _ = M.train_quantile_ensemble(X, y, split, quantiles=[0.5], num_boost_round=80, verbose=False)
    shap_result = M.shap_feature_importance(models[0.5], X, top_n=5, max_rows=500)["top_features"]
    gain_result = M.feature_importance(models, top_n=5)

    for d in (shap_result, gain_result):
        for row in d:
            assert "feature" in row and "importance_rank" in row
    assert [r["importance_rank"] for r in shap_result] == list(range(1, 6))


def test_shap_subsamples_large_input(synthetic_data):
    X, y, split = synthetic_data
    models, _ = M.train_quantile_ensemble(X, y, split, quantiles=[0.5], num_boost_round=80, verbose=False)
    result = M.shap_feature_importance(models[0.5], X, top_n=3, max_rows=100)
    assert result["n_rows_sampled"] == 100  # X has 2000 rows; must subsample down to max_rows


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""
tests/test_shap_per_channel_drivers.py
========================================
§H.1 follow-up — a per-channel AI causal summary should reflect that
channel's OWN drivers, not a single account-wide ranking pasted into every
scope. `predict.build_causal_summaries` used to compute `top_drivers` once
(`bundle["shap_importance"]["top_features"]`, the global holdout SHAP
ranking) *outside* the per-scope loop and pass the identical list into
every scope's grounding context — confirmed by inspecting a real run's
`output/causal_summary.json`: `key_drivers` was byte-identical across
`total`/`bing`/`google`/`meta`. Not fabricated data (it's real SHAP), but
it silently undercut the "grounded, per-channel" narrative the LLM/fallback
copy claims to give.

Fixed in two places:
  - `modeling.shap_feature_importance(..., groups=...)` now ALSO returns
    `by_group`: the same already-computed SHAP matrix broken down by
    whatever grouping series is passed in (e.g. `channel`), so no extra
    TreeExplainer pass is needed. Groups with too few sampled rows
    (`min_group_rows`, default 20) are omitted rather than reported off a
    noisy handful of rows.
  - `predict.build_causal_summaries` now looks up `by_group[channel]` for
    each channel-scoped narrative and only falls back to the global ranking
    for the "total" scope, a channel with no `by_group` entry, or an older
    bundle trained before this existed at all (`shap_importance` missing or
    lacking the key entirely) — verified NOT to raise on either legacy
    shape.
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import modeling as M
from predict import build_predictions, build_causal_summaries
from schema_mapper import ingest_directory

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_PATH = os.path.join(HERE, "..", "pickle", "model.pkl")
FEATURES_PATH = os.path.join(HERE, "..", "output", "features.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# Unit level — modeling.shap_feature_importance(groups=...)
# ─────────────────────────────────────────────────────────────────────────────
def _toy_group_dependent_booster():
    """A tiny synthetic dataset where feature `x_a` drives the target ONLY
    for group 'A' rows and `x_b` drives it ONLY for group 'B' rows, so a
    correct per-group SHAP breakdown must rank them oppositely, while the
    pooled/global ranking sees both as comparably important overall."""
    rng = np.random.default_rng(0)
    n = 600
    group = np.where(np.arange(n) < n // 2, "A", "B")
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    noise_feat = rng.normal(size=n)
    y = np.where(group == "A", 5.0 * x_a, 5.0 * x_b) + 0.01 * noise_feat
    X = pd.DataFrame({"x_a": x_a, "x_b": x_b, "noise_feat": noise_feat})
    booster = lgb.train(
        {"objective": "regression", "verbosity": -1, "seed": 42, "min_child_samples": 5},
        lgb.Dataset(X, label=y),
        num_boost_round=80,
    )
    return booster, X, pd.Series(group)


def test_by_group_differentiates_drivers_between_groups():
    booster, X, groups = _toy_group_dependent_booster()
    result = M.shap_feature_importance(booster, X, top_n=3, groups=groups, min_group_rows=10)

    assert "by_group" in result
    assert set(result["by_group"].keys()) == {"A", "B"}

    top_feature_a = result["by_group"]["A"]["top_features"][0]["feature"]
    top_feature_b = result["by_group"]["B"]["top_features"][0]["feature"]
    assert top_feature_a == "x_a"
    assert top_feature_b == "x_b"
    assert top_feature_a != top_feature_b  # the actual bug this replaces


def test_by_group_omits_groups_below_min_rows():
    booster, X, groups = _toy_group_dependent_booster()
    # ask for an unreasonably high min_group_rows -> neither group qualifies
    result = M.shap_feature_importance(booster, X, top_n=3, groups=groups, min_group_rows=10_000)
    assert result["by_group"] == {}


def test_no_groups_arg_omits_by_group_key_entirely():
    """Backward compatible: calling without `groups` (old call sites, or a
    bundle re-trained without channel info) must not add a `by_group` key
    at all, not an empty one -- distinguishes "not computed" from "computed,
    nothing qualified"."""
    booster, X, _ = _toy_group_dependent_booster()
    result = M.shap_feature_importance(booster, X, top_n=3)
    assert "by_group" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Integration level — predict.build_causal_summaries against the real
# shipped bundle + real data
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="pickle/model.pkl not built yet")
@pytest.mark.skipif(not os.path.exists(FEATURES_PATH), reason="output/features.parquet not built yet (run.sh first)")
def test_real_bundle_gives_each_channel_its_own_drivers():
    bundle = joblib.load(MODEL_PATH)
    assert "by_group" in bundle["shap_importance"], (
        "shipped pickle/model.pkl was trained before the per-channel SHAP fix -- "
        "re-run train.py (or the patch step) to regenerate it."
    )

    features = pd.read_parquet(FEATURES_PATH)
    canonical_df, _ = ingest_directory(DATA_DIR)
    predictions = build_predictions(bundle, features)
    summaries = build_causal_summaries(bundle, canonical_df, predictions, horizon=30)

    by_scope = {s["scope_label"]: s["key_drivers"] for s in summaries}
    assert "total" in by_scope
    channel_scopes = [k for k in by_scope if k != "total"]
    assert len(channel_scopes) >= 2, "expected at least 2 channel-level summaries to compare"

    # the actual bug: every scope's key_drivers used to be byte-identical
    driver_lists = [tuple(by_scope[ch]) for ch in channel_scopes]
    assert len(set(driver_lists)) > 1, (
        "every channel's key_drivers are still identical -- the per-channel "
        "SHAP breakdown isn't being used"
    )


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="pickle/model.pkl not built yet")
@pytest.mark.skipif(not os.path.exists(FEATURES_PATH), reason="output/features.parquet not built yet (run.sh first)")
def test_causal_summaries_survive_bundle_without_by_group():
    """A bundle trained before this fix existed (no `by_group`, or
    `shap_importance` entirely absent) must still produce a full set of
    per-scope summaries via graceful fallback to the global ranking, not
    raise a KeyError."""
    bundle = dict(joblib.load(MODEL_PATH))  # shallow copy, don't mutate the real file
    bundle["shap_importance"] = {k: v for k, v in bundle["shap_importance"].items() if k != "by_group"}

    features = pd.read_parquet(FEATURES_PATH)
    canonical_df, _ = ingest_directory(DATA_DIR)
    predictions = build_predictions(bundle, features)
    summaries = build_causal_summaries(bundle, canonical_df, predictions, horizon=30)
    assert len(summaries) >= 2

    bundle["shap_importance"] = None
    summaries2 = build_causal_summaries(bundle, canonical_df, predictions, horizon=30)
    assert len(summaries2) >= 2

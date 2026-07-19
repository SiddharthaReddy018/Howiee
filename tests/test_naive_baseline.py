"""
tests/test_naive_baseline.py
=============================
Unit tests for `modeling.naive_pace_forecast` (§G.2.5) — the trivial
"continue at recent pace" reference forecast added so the final-holdout
WAPE/CRPS numbers have an honest "compared to what?" baseline instead of
standing alone.

This is a pure, small-input unit test (no training, no real data) — the
integration point (does the production model actually beat this on the
real final holdout, and does the number make it into output/reliability.json)
is covered by running `train.py` + `predict.py` directly, not re-asserted
here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import modeling as M


def _toy_frame(pace, horizon):
    return pd.DataFrame({"revenue_roll_mean_28": pace, "horizon_days": horizon})


def test_naive_forecast_is_pace_times_horizon():
    X = _toy_frame(pace=[100.0, 50.0], horizon=[30, 90])
    out = M.naive_pace_forecast(X)
    np.testing.assert_allclose(out, [3000.0, 4500.0])


def test_naive_forecast_nan_pace_defaults_to_zero_not_crash():
    X = _toy_frame(pace=[np.nan, 20.0], horizon=[30, 30])
    out = M.naive_pace_forecast(X)
    np.testing.assert_allclose(out, [0.0, 600.0])


def test_naive_forecast_never_negative():
    # A pathological negative rolling mean (shouldn't occur upstream, but the
    # baseline must not silently produce a negative revenue forecast either way).
    X = _toy_frame(pace=[-10.0], horizon=[30])
    out = M.naive_pace_forecast(X)
    assert out[0] == 0.0


def test_naive_wape_is_worse_than_a_near_perfect_model_on_synthetic_data():
    """Sanity check that the WAPE-comparison logic used in train.py behaves as
    expected: a model that's actually close to the truth should show a large
    positive improvement over the naive pace baseline when the truth deviates
    a lot from the recent pace (e.g. a real budget change)."""
    rng = np.random.default_rng(0)
    pace = rng.uniform(50, 200, size=200)
    horizon = np.full(200, 30)
    X = _toy_frame(pace=pace, horizon=horizon)
    naive_pred = M.naive_pace_forecast(X)

    # True revenue is a 3x scale-up vs. recent pace (e.g. a real budget increase)
    # -- the naive baseline can't see this, a "model" that's aware of it can.
    y_true = pace * horizon * 3.0
    good_model_pred = y_true * rng.uniform(0.95, 1.05, size=200)  # +/-5% noise

    naive_wape = M.wape(y_true, naive_pred)
    model_wape = M.wape(y_true, good_model_pred)
    improvement = 1.0 - (model_wape / naive_wape)

    assert model_wape < naive_wape
    assert improvement > 0.5  # model should be substantially better here

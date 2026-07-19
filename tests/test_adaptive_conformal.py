"""
tests/test_adaptive_conformal.py
=================================
Implementation Plan §D.2 — tests for the previously entirely-missing
Adaptive Conformal Inference module (`src/adaptive_conformal.py`).

Three things checked:
  1. On a stationary (non-shifting) process, ACI's empirical coverage lands
     near the nominal target -- it shouldn't do anything strange when
     there's no distribution shift to react to.
  2. The actual point of ACI: under a simulated variance shift partway
     through the sequence (intervals become too narrow for the new regime),
     ACI's alpha_t adapts and post-adaptation coverage recovers toward
     nominal, while a static, one-shot correction fit before the shift does
     not.
  3. Basic API contracts (length mismatches, too-short sequences, the
     `window` parameter).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adaptive_conformal import run_adaptive_conformal_inference, compare_static_vs_adaptive


def _stationary_normal_process(n=600, sigma=1.0, alpha_target=0.2, seed=0):
    """i.i.d. N(0, sigma); q_lo/q_hi set to the TRUE nominal interval for
    this sigma, so a correctly-behaving conformal method (adaptive or not)
    should already show ~ (1-alpha_target) empirical coverage throughout."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, sigma, size=n)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha_target / 2)
    q_lo = np.full(n, -z * sigma)
    q_hi = np.full(n, z * sigma)
    return q_lo, q_hi, y


def _variance_shift_process(n=600, sigma_before=1.0, sigma_after=3.0, alpha_target=0.2, seed=0):
    """First half: N(0, sigma_before), intervals sized correctly for it.
    Second half: true generating variance jumps to sigma_after, but q_lo/q_hi
    stay FIXED at the sigma_before-sized interval (as if the base quantile
    model was never retrained) -- exactly the "stale calibration under
    distribution shift" scenario ACI exists for."""
    rng = np.random.default_rng(seed)
    half = n // 2
    y = np.concatenate([
        rng.normal(0, sigma_before, size=half),
        rng.normal(0, sigma_after, size=n - half),
    ])
    from scipy.stats import norm
    z = norm.ppf(1 - alpha_target / 2)
    q_lo = np.full(n, -z * sigma_before)
    q_hi = np.full(n, z * sigma_before)
    return q_lo, q_hi, y, half


# ─────────────────────────────────────────────────────────────────────────────
# 1. Stationary process — should already track nominal coverage
# ─────────────────────────────────────────────────────────────────────────────
def test_aci_tracks_nominal_coverage_on_stationary_process():
    alpha_target = 0.2
    q_lo, q_hi, y = _stationary_normal_process(n=800, alpha_target=alpha_target, seed=1)
    result = run_adaptive_conformal_inference(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.02, min_history=50)
    nominal = 1 - alpha_target
    assert abs(result["empirical_coverage_post_warmup"] - nominal) < 0.06, (
        f"expected ACI coverage near {nominal} on a stationary process, got {result['empirical_coverage_post_warmup']}"
    )


def test_aci_alpha_t_stays_near_target_when_no_shift_occurs():
    alpha_target = 0.1
    q_lo, q_hi, y = _stationary_normal_process(n=600, alpha_target=alpha_target, seed=2)
    result = run_adaptive_conformal_inference(q_lo, q_hi, y, alpha_target=alpha_target, min_history=50)
    assert abs(result["final_alpha_t"] - alpha_target) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# 2. The actual point of ACI: recovers coverage after a distribution shift,
#    where a static one-shot correction cannot
# ─────────────────────────────────────────────────────────────────────────────
def test_aci_recovers_coverage_after_variance_shift_better_than_static():
    alpha_target = 0.2
    q_lo, q_hi, y, half = _variance_shift_process(n=1000, sigma_before=1.0, sigma_after=3.0,
                                                    alpha_target=alpha_target, seed=3)
    comparison = compare_static_vs_adaptive(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.03, warmup_frac=0.2)

    nominal = 1 - alpha_target
    static_gap = abs(comparison["static"]["empirical_coverage_post_warmup"] - nominal)
    adaptive_gap = abs(comparison["adaptive"]["empirical_coverage_post_warmup"] - nominal)

    # the static correction, fit before the shift, should visibly undercover
    # post-shift (this is the failure mode ACI exists to fix)
    assert comparison["static"]["empirical_coverage_post_warmup"] < nominal - 0.05, (
        "expected the static correction to visibly undercover after the simulated variance shift"
    )
    # ACI must close a meaningful fraction of that gap
    assert adaptive_gap < static_gap, (
        f"expected ACI to track nominal coverage better than a static correction after a shift: "
        f"static_gap={static_gap:.3f} adaptive_gap={adaptive_gap:.3f}"
    )
    # and ACI's post-adaptation interval should have widened relative to the untouched raw interval
    raw_width = float(np.mean(q_hi - q_lo))
    assert comparison["adaptive"]["mean_interval_width_post_warmup"] > raw_width


def test_aci_widens_intervals_progressively_after_shift():
    alpha_target = 0.2
    q_lo, q_hi, y, half = _variance_shift_process(n=800, sigma_before=1.0, sigma_after=3.0,
                                                    alpha_target=alpha_target, seed=4)
    result = run_adaptive_conformal_inference(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.03, min_history=50)
    widths = np.array(result["interval_width_history"])
    # mean width well after the shift should exceed mean width well before it
    pre_shift_width = widths[100:half].mean()
    post_shift_width = widths[half + 150:].mean()
    assert post_shift_width > pre_shift_width, (
        f"expected intervals to widen after the shift: pre={pre_shift_width:.2f} post={post_shift_width:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. API contracts
# ─────────────────────────────────────────────────────────────────────────────
def test_aci_rejects_mismatched_lengths():
    with pytest.raises(AssertionError):
        run_adaptive_conformal_inference(np.zeros(10), np.zeros(9), np.zeros(10), min_history=2)


def test_aci_rejects_too_short_sequence():
    with pytest.raises(ValueError):
        run_adaptive_conformal_inference(np.zeros(5), np.ones(5), np.zeros(5), min_history=10)


def test_aci_sliding_window_option_runs():
    q_lo, q_hi, y = _stationary_normal_process(n=300, seed=5)
    result = run_adaptive_conformal_inference(q_lo, q_hi, y, min_history=30, window=50)
    assert result["n_steps"] == 300
    assert len(result["alpha_t_history"]) == 300


def test_compare_static_vs_adaptive_rejects_too_short_sequence():
    with pytest.raises(ValueError):
        compare_static_vs_adaptive(np.zeros(15), np.ones(15), np.zeros(15))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conformal PID control — same scenarios, extends run_adaptive_conformal_inference
# ─────────────────────────────────────────────────────────────────────────────
from adaptive_conformal import run_conformal_pid_control, compare_static_vs_adaptive_vs_pid


def test_pid_tracks_nominal_coverage_on_stationary_process():
    alpha_target = 0.2
    q_lo, q_hi, y = _stationary_normal_process(n=800, alpha_target=alpha_target, seed=1)
    result = run_conformal_pid_control(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.02, min_history=50)
    nominal = 1 - alpha_target
    assert abs(result["empirical_coverage_post_warmup"] - nominal) < 0.08, (
        f"expected PID coverage near {nominal} on a stationary process, got {result['empirical_coverage_post_warmup']}"
    )


def test_pid_recovers_coverage_after_variance_shift_better_than_static():
    alpha_target = 0.2
    q_lo, q_hi, y, half = _variance_shift_process(n=1000, sigma_before=1.0, sigma_after=3.0,
                                                    alpha_target=alpha_target, seed=3)
    comparison = compare_static_vs_adaptive_vs_pid(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.03, warmup_frac=0.2)

    nominal = 1 - alpha_target
    static_gap = abs(comparison["static"]["empirical_coverage_post_warmup"] - nominal)
    pid_gap = abs(comparison["pid"]["empirical_coverage_post_warmup"] - nominal)

    assert pid_gap < static_gap, (
        f"expected PID to track nominal coverage better than a static correction after a shift: "
        f"static_gap={static_gap:.3f} pid_gap={pid_gap:.3f}"
    )
    raw_width = float(np.mean(q_hi - q_lo))
    assert comparison["pid"]["mean_interval_width_post_warmup"] > raw_width


def test_pid_matches_plain_aci_when_i_and_d_gains_are_zero():
    """Sanity/regression check: with ki=0, kd=0, the PID controller's update
    rule collapses to EXACTLY the plain-ACI update rule (same P term, same
    formula) -- confirms the I/D terms are additive extensions, not a
    reimplementation that happens to diverge from the already-tested ACI
    behavior."""
    alpha_target = 0.2
    q_lo, q_hi, y = _stationary_normal_process(n=400, alpha_target=alpha_target, seed=7)
    aci = run_adaptive_conformal_inference(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.02, min_history=40)
    pid = run_conformal_pid_control(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.02,
                                     ki=0.0, kd=0.0, min_history=40)
    np.testing.assert_allclose(aci["alpha_t_history"], pid["alpha_t_history"], atol=1e-9)
    assert aci["final_alpha_t"] == pytest.approx(pid["final_alpha_t"], abs=1e-9)


def test_pid_integral_term_is_clipped_not_unbounded():
    """A long, deliberately extreme run of misses should not blow up alpha_t
    to a degenerate value -- confirms the anti-windup clip is actually
    doing something."""
    n = 500
    q_lo = np.zeros(n)
    q_hi = np.zeros(n)  # zero-width interval -> guarantees a "miss" every single step
    y = np.ones(n)
    result = run_conformal_pid_control(q_lo, q_hi, y, alpha_target=0.1, ki=0.05, integral_clip=5.0, min_history=20)
    assert 0.0 < result["final_alpha_t"] < 1.0
    assert np.isfinite(result["final_alpha_t"])


def test_pid_rejects_mismatched_lengths():
    with pytest.raises(AssertionError):
        run_conformal_pid_control(np.zeros(10), np.zeros(9), np.zeros(10), min_history=2)


def test_compare_three_way_reuses_identical_static_and_aci_numbers():
    """The three-way comparison must not silently recompute static/ACI with
    different results than the standalone two-way comparison -- same inputs,
    same settings, must give identical static+adaptive numbers."""
    alpha_target = 0.2
    q_lo, q_hi, y, half = _variance_shift_process(n=600, alpha_target=alpha_target, seed=9)
    two_way = compare_static_vs_adaptive(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.03, warmup_frac=0.2)
    three_way = compare_static_vs_adaptive_vs_pid(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.03, warmup_frac=0.2)
    assert two_way["static"] == three_way["static"]
    assert two_way["adaptive"] == three_way["adaptive"]
    assert "pid" in three_way
    assert "pid_learned_scorecaster" in three_way


# ─────────────────────────────────────────────────────────────────────────────
# §D.2c — genuine learned scorecaster D-term
# ─────────────────────────────────────────────────────────────────────────────
from adaptive_conformal import run_conformal_pid_control_learned_scorecaster, _fit_predict_next_score


def test_scorecaster_falls_back_to_mean_with_insufficient_history():
    assert _fit_predict_next_score([], n_lags=3) == 0.0
    assert _fit_predict_next_score([5.0], n_lags=3) == pytest.approx(5.0)
    assert _fit_predict_next_score([2.0, 4.0], n_lags=3) == pytest.approx(3.0)


def test_scorecaster_recovers_a_simple_linear_trend():
    """A perfectly linear score history (score_t = t) should let the
    ridge-AR scorecaster forecast the next point close to the true next
    value -- confirms it's actually fitting a real regression, not just
    echoing the last-seen value or the running mean."""
    hist = list(np.arange(0.0, 30.0))  # 0, 1, 2, ..., 29
    forecast = _fit_predict_next_score(hist, n_lags=3, ridge_lambda=1e-6)
    assert forecast == pytest.approx(30.0, abs=1.0), f"expected ~30.0 on a linear trend, got {forecast}"


def test_scorecaster_refits_fresh_each_call_not_cached():
    """Calling twice with different histories must give different forecasts
    -- confirms the model is genuinely refit from the passed-in history
    each call, not memoized/cached statefully somewhere."""
    hist_a = list(np.arange(0.0, 20.0))
    hist_b = list(np.arange(0.0, 20.0) * -1)
    fa = _fit_predict_next_score(hist_a, n_lags=3)
    fb = _fit_predict_next_score(hist_b, n_lags=3)
    assert fa != pytest.approx(fb)


def test_pid_learned_scorecaster_tracks_nominal_coverage_on_stationary_process():
    alpha_target = 0.2
    q_lo, q_hi, y = _stationary_normal_process(n=800, alpha_target=alpha_target, seed=1)
    result = run_conformal_pid_control_learned_scorecaster(
        q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.02, min_history=50,
    )
    nominal = 1 - alpha_target
    assert abs(result["empirical_coverage_post_warmup"] - nominal) < 0.08, (
        f"expected learned-scorecaster PID coverage near {nominal} on a stationary process, "
        f"got {result['empirical_coverage_post_warmup']}"
    )


def test_pid_learned_scorecaster_matches_proxy_pid_when_kd_is_zero():
    """With kd=0, the D-term contributes nothing regardless of which D-term
    implementation is used -- both PID variants must collapse to the
    IDENTICAL P+I-only trajectory, confirming the learned-scorecaster
    variant only changes the D-term and nothing about the P/I mechanics."""
    alpha_target = 0.2
    q_lo, q_hi, y = _stationary_normal_process(n=400, alpha_target=alpha_target, seed=7)
    pid_proxy = run_conformal_pid_control(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.02,
                                           ki=0.01, kd=0.0, min_history=40)
    pid_learned = run_conformal_pid_control_learned_scorecaster(
        q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.02, ki=0.01, kd=0.0, min_history=40,
    )
    np.testing.assert_allclose(pid_proxy["alpha_t_history"], pid_learned["alpha_t_history"], atol=1e-9)


def test_pid_learned_scorecaster_recovers_coverage_after_variance_shift_better_than_static():
    alpha_target = 0.2
    q_lo, q_hi, y, half = _variance_shift_process(n=1000, sigma_before=1.0, sigma_after=3.0,
                                                    alpha_target=alpha_target, seed=3)
    result = run_conformal_pid_control_learned_scorecaster(
        q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.03, min_history=200,
    )
    nominal = 1 - alpha_target
    learned_gap = abs(result["empirical_coverage_post_warmup"] - nominal)
    # static (no adaptation at all) leaves the pre-shift interval completely
    # unchanged through the post-shift regime -- compute its gap directly
    from modeling import fit_cqr_correction, apply_cqr, coverage as _coverage_fn
    warmup_n = 200
    q_hat_static = fit_cqr_correction(q_lo[:warmup_n], q_hi[:warmup_n], y[:warmup_n], alpha=alpha_target)
    lo_static, hi_static = apply_cqr(q_lo[warmup_n:], q_hi[warmup_n:], q_hat_static)
    static_gap = abs(_coverage_fn(y[warmup_n:], lo_static, hi_static) - nominal)
    assert learned_gap < static_gap, (
        f"expected learned-scorecaster PID to track nominal coverage better than a static "
        f"correction after a shift: static_gap={static_gap:.3f} learned_gap={learned_gap:.3f}"
    )


def test_pid_learned_scorecaster_rejects_mismatched_lengths():
    with pytest.raises(AssertionError):
        run_conformal_pid_control_learned_scorecaster(np.zeros(10), np.zeros(9), np.zeros(10), min_history=2)


def test_pid_learned_scorecaster_score_forecast_history_has_nones_during_warmup():
    alpha_target = 0.2
    min_history = 30
    q_lo, q_hi, y = _stationary_normal_process(n=200, alpha_target=alpha_target, seed=2)
    result = run_conformal_pid_control_learned_scorecaster(
        q_lo, q_hi, y, alpha_target=alpha_target, min_history=min_history, scorecaster_lags=3,
    )
    forecasts = result["score_forecast_history"]
    assert len(forecasts) == 200
    # t < min_history -> None (no forecast issued yet); by the time t reaches
    # min_history, scores_seen already has min_history entries (one appended
    # every step, including during warmup), which comfortably exceeds
    # scorecaster_lags=3, so real forecasts start immediately at t=min_history.
    assert all(f is None for f in forecasts[:min_history])
    assert all(f is not None for f in forecasts[min_history:])


def test_compare_vs_pid_learned_scorecaster_report_is_well_formed():
    alpha_target = 0.2
    q_lo, q_hi, y, half = _variance_shift_process(n=600, alpha_target=alpha_target, seed=9)
    report = compare_static_vs_adaptive_vs_pid(q_lo, q_hi, y, alpha_target=alpha_target, gamma=0.03, warmup_frac=0.2)
    pl = report["pid_learned_scorecaster"]
    assert 0.0 <= pl["empirical_coverage_post_warmup"] <= 1.0
    assert pl["mean_interval_width_post_warmup"] > 0
    assert 0.0 < pl["final_alpha_t"] < 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

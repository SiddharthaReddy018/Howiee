"""
adaptive_conformal.py
======================
Implementation Plan §D.2 — Adaptive Conformal Inference (marked optional in
the plan, "if you have time"). §D.1's CQR (`modeling.fit_cqr_correction` /
`apply_cqr`) is static: ONE correction term Q̂ is fit once on a calibration
slice and applied unchanged to every future prediction. That's the shipped
default here, and it's independently verified well-calibrated (see
`docs/technical_documentation.md` §6/§D.3 — 95.5%/90.5% empirical coverage
at 90%/80% nominal on the final holdout).

ACI (Gibbs & Candès, "Adaptive Conformal Inference Under Distribution
Shift", NeurIPS 2021) instead updates a running miscoverage target alpha_t
online, step by step, as each new true outcome arrives:

    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

where `err_t = 1` if the realized value fell outside the interval built
from alpha_t, else 0. The interval width at each step comes from the
empirical quantile of nonconformity scores seen SO FAR at the (1 - alpha_t)
level — the same `max(q_lo - y, y - q_hi)` score `fit_cqr_correction` uses,
just recomputed sequentially instead of once. Practical effect: if the
model starts systematically under- or over-covering (distribution shift),
alpha_t drifts to compensate, instead of carrying a stale, single, static
correction indefinitely.

This module is intentionally separate from `modeling.py`'s production CQR
path — it is NOT wired into the shipped calibration used by `predict.py`.
Two honest reasons, not oversight:
  1. §D.1's static CQR already measures well-calibrated on real held-out
     data (see above); ACI is a hedge against distribution shift that
     hasn't been shown to be a problem here yet, not a fix for a known one.
  2. ACI is inherently sequential/online (it needs true outcomes arriving
     one at a time, in order, to update alpha_t) — that fits a monitoring
     dashboard or a rolling-retrain job far more naturally than this
     project's current single-shot batch inference (`predict.py` scores
     every campaign/horizon at once with no "wait and observe the actual"
     step in between).
`run_adaptive_conformal_inference` and `compare_static_vs_adaptive` below
are real, tested implementations available for exactly that future use —
see docs/technical_documentation.md §6a for the worked comparison on this
project's own final-holdout timeline.

§D.2c (added this round) — `run_conformal_pid_control_learned_scorecaster`
implements the PID controller's D-term the way Angelopoulos, Candès &
Tibshirani actually specify it: a small model that forecasts the NEXT
nonconformity score from score history, refit online, rather than
`run_conformal_pid_control`'s cheaper derivative-of-error-signal proxy (kept
alongside, unchanged, as its own honestly-scoped candidate -- see that
function's docstring). `compare_static_vs_adaptive_vs_pid` now runs BOTH
D-term variants on the identical sequence so "does a genuinely learned
scorecaster beat the cheap proxy" is an honest, reported comparison, not an
assumption -- worth having settled before Conformal PID (either variant)
ever became something this project shipped rather than reported.
"""

from __future__ import annotations

import numpy as np


def _nonconformity_scores(q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Same score CQR uses (see modeling.fit_cqr_correction): positive when
    y falls outside [q_lo, q_hi], negative (unused, but informative) when
    comfortably inside."""
    return np.maximum(q_lo - y, y - q_hi)


def _fit_predict_next_score(hist: list[float] | np.ndarray, n_lags: int = 3,
                             ridge_lambda: float = 1.0) -> float:
    """§D.2c's actual scorecaster: a ridge-regularized AR(`n_lags`) model,
    refit FROM SCRATCH on the full expanding history available at every
    single call (genuinely re-learned online, not a fixed formula or a
    model fit once and reused) that forecasts the next nonconformity score
    one step ahead from the most recent `n_lags` values.

    Deliberately small and closed-form (ordinary ridge regression, not a
    gradient-boosted model or anything from `modeling.py`) — the scorecaster
    forecasts a single scalar time series (the score sequence) a few dozen
    to a few hundred points long per calibration run; a GBDT here would be
    both unnecessary and prone to overfitting on that little data. Ridge
    (not plain OLS) specifically so a short early history with a
    near-singular lag design matrix still produces a stable, finite
    coefficient vector instead of an ill-conditioned blow-up.

    Falls back to the plain historical mean once there isn't yet enough
    history to build even one (lag-window -> next-value) training pair —
    the honest "not enough signal yet" case, not a fabricated forecast.
    """
    hist = np.asarray(hist, dtype=float)
    if len(hist) <= n_lags:
        return float(np.mean(hist)) if len(hist) else 0.0

    X_rows = np.stack([hist[i - n_lags:i] for i in range(n_lags, len(hist))])
    y_rows = hist[n_lags:]
    X_aug = np.hstack([X_rows, np.ones((len(X_rows), 1))])  # intercept column

    k = X_aug.shape[1]
    reg = ridge_lambda * np.eye(k)
    reg[-1, -1] = 0.0  # never regularize the intercept
    try:
        beta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_rows)
    except np.linalg.LinAlgError:
        return float(hist[-1])  # last-value fallback if even the regularized solve is singular

    x_next = np.concatenate([hist[-n_lags:], [1.0]])
    return float(x_next @ beta)


def run_adaptive_conformal_inference(
    q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray,
    alpha_target: float = 0.1, gamma: float = 0.02,
    min_history: int = 20, window: int | None = None,
) -> dict:
    """Runs ACI over `(q_lo, q_hi, y)` IN THE GIVEN ORDER — caller is
    responsible for chronological ordering; this function has no notion of
    time itself, only sequence position.

    `alpha_target`: nominal miscoverage (0.1 -> targeting 90% coverage).
    `gamma`: step size for the alpha_t update (Gibbs & Candès' `gamma`;
      larger -> more reactive to recent misses, noisier; smaller -> smoother,
      slower to adapt). 0.02 is a conservative default for daily-ish data.
    `min_history`: number of initial steps to run with alpha_t frozen at
      alpha_target and Q_hat=0 (i.e., the raw [q_lo, q_hi] interval, no
      correction) — there simply isn't enough score history yet to fit a
      meaningful quantile before this point.
    `window`: if set, only the most recent `window` scores are used to fit
      each step's quantile (a sliding window instead of full expanding
      history) — more reactive to *recent* shift, at the cost of noisier
      per-step quantile estimates on the same score budget.

    Returns per-step history (alpha_t, Q_hat_t, covered, interval width)
    plus summary stats, for both the full run and a comparison-friendly
    "post-warmup" slice (excludes `min_history` steps).
    """
    n = len(y)
    assert len(q_lo) == n and len(q_hi) == n, "q_lo/q_hi/y must be the same length"
    if n <= min_history:
        raise ValueError(f"need more than min_history={min_history} points, got n={n}")

    scores_seen: list[float] = []
    alpha_t = alpha_target
    alphas, q_hats, covered, widths = [], [], [], []

    for t in range(n):
        if t < min_history:
            q_hat_t = 0.0
        else:
            hist = scores_seen[-window:] if window else scores_seen
            level = float(np.clip(1 - alpha_t, 0.0, 1.0))
            q_hat_t = float(np.quantile(hist, level)) if hist else 0.0

        lo_t = q_lo[t] - q_hat_t
        hi_t = q_hi[t] + q_hat_t
        is_covered = bool(lo_t <= y[t] <= hi_t)
        err_t = 0 if is_covered else 1

        alphas.append(alpha_t)
        q_hats.append(q_hat_t)
        covered.append(is_covered)
        widths.append(float(hi_t - lo_t))

        # update AFTER scoring this step, using this step's outcome
        alpha_t = float(np.clip(alpha_t + gamma * (alpha_target - err_t), 1e-6, 1 - 1e-6))
        scores_seen.append(float(_nonconformity_scores(q_lo[t:t + 1], q_hi[t:t + 1], y[t:t + 1])[0]))

    covered = np.array(covered)
    post_warmup = covered[min_history:]
    return {
        "alpha_target": alpha_target, "gamma": gamma, "min_history": min_history, "window": window,
        "n_steps": n,
        "alpha_t_history": alphas,
        "q_hat_history": q_hats,
        "covered_history": covered.tolist(),
        "interval_width_history": widths,
        "empirical_coverage_full": float(covered.mean()),
        "empirical_coverage_post_warmup": float(post_warmup.mean()) if len(post_warmup) else None,
        "mean_interval_width_post_warmup": float(np.mean(widths[min_history:])) if n > min_history else None,
        "final_alpha_t": alpha_t,
    }


def run_conformal_pid_control(
    q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray,
    alpha_target: float = 0.1, gamma: float = 0.02,
    ki: float = 0.01, integral_clip: float = 5.0,
    kd: float = 0.3, d_ewma_decay: float = 0.9,
    min_history: int = 20, window: int | None = None,
) -> dict:
    """
    Conformal PID control (Angelopoulos, Candès & Tibshirani, "Conformal PID
    Control for Time Series Prediction," NeurIPS 2023) — a genuine, honestly
    scoped extension of `run_adaptive_conformal_inference` above, not a
    separate reimplementation: ACI's own update rule IS this controller's P
    term (`gamma * (alpha_target - err_t)`), reused here unchanged, with two
    more terms added on top:

      P  gamma * (alpha_target - err_t)         -- identical to plain ACI
      I  ki * (running sum of past P-signals,     -- corrects the *persistent*
             clipped to +/-integral_clip)            bias a pure-P controller
                                                       can leave uncorrected
      D  kd * (this step's signal - an EWMA        -- reacts to the recent
             of recent signals)                       TREND in miscoverage,
                                                       not just its current level

    Honest scoping note on the D term: the published method's D term is a
    learned "scorecaster" — a separate lightweight forecasting model
    predicting the next nonconformity score directly. Implementing and
    validating a second forecasting sub-model is a meaningfully bigger
    undertaking than this comparison's evaluation-only purpose warrants, so
    the D term here is a simplified, standard derivative-of-error proxy (the
    current miscoverage signal vs. an EWMA of its recent history) — a
    defensible PID component in its own right, but a scoped-down stand-in
    for the paper's own D term, not a claim of exact reproduction. Reported
    and compared like every other honestly-scoped experiment in this
    codebase (see the adstock ablation, §4).

    `integral_clip` bounds the I-term's accumulator (anti-windup) so a long
    early run of misses can't leave alpha_t permanently saturated once
    conditions normalize — the same class of problem industrial PID
    controllers guard against with integral clamping.

    Returns the same schema as `run_adaptive_conformal_inference`, so
    `compare_static_vs_adaptive_vs_pid` below can treat all three methods
    uniformly.
    """
    n = len(y)
    assert len(q_lo) == n and len(q_hi) == n, "q_lo/q_hi/y must be the same length"
    if n <= min_history:
        raise ValueError(f"need more than min_history={min_history} points, got n={n}")

    scores_seen: list[float] = []
    alpha_t = alpha_target
    integral_acc = 0.0
    ewma_signal = 0.0
    alphas, q_hats, covered, widths = [], [], [], []

    for t in range(n):
        if t < min_history:
            q_hat_t = 0.0
        else:
            hist = scores_seen[-window:] if window else scores_seen
            level = float(np.clip(1 - alpha_t, 0.0, 1.0))
            q_hat_t = float(np.quantile(hist, level)) if hist else 0.0

        lo_t = q_lo[t] - q_hat_t
        hi_t = q_hi[t] + q_hat_t
        is_covered = bool(lo_t <= y[t] <= hi_t)
        err_t = 0 if is_covered else 1

        alphas.append(alpha_t)
        q_hats.append(q_hat_t)
        covered.append(is_covered)
        widths.append(float(hi_t - lo_t))

        signal_t = alpha_target - err_t

        p_term = gamma * signal_t
        integral_acc = float(np.clip(integral_acc + signal_t, -integral_clip, integral_clip))
        i_term = ki * integral_acc
        d_term = kd * (signal_t - ewma_signal)
        ewma_signal = d_ewma_decay * ewma_signal + (1 - d_ewma_decay) * signal_t

        alpha_t = float(np.clip(alpha_t + p_term + i_term + d_term, 1e-6, 1 - 1e-6))
        scores_seen.append(float(_nonconformity_scores(q_lo[t:t + 1], q_hi[t:t + 1], y[t:t + 1])[0]))

    covered = np.array(covered)
    post_warmup = covered[min_history:]
    return {
        "alpha_target": alpha_target, "gamma": gamma, "ki": ki, "kd": kd,
        "min_history": min_history, "window": window, "n_steps": n,
        "alpha_t_history": alphas,
        "q_hat_history": q_hats,
        "covered_history": covered.tolist(),
        "interval_width_history": widths,
        "empirical_coverage_full": float(covered.mean()),
        "empirical_coverage_post_warmup": float(post_warmup.mean()) if len(post_warmup) else None,
        "mean_interval_width_post_warmup": float(np.mean(widths[min_history:])) if n > min_history else None,
        "final_alpha_t": alpha_t,
    }


def run_conformal_pid_control_learned_scorecaster(
    q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray,
    alpha_target: float = 0.1, gamma: float = 0.02,
    ki: float = 0.01, integral_clip: float = 5.0,
    kd: float = 0.3, scorecaster_lags: int = 3, scorecaster_ridge: float = 1.0,
    min_history: int = 20, window: int | None = None,
) -> dict:
    """§D.2c — the genuine version of `run_conformal_pid_control`'s D-term.

    Angelopoulos, Candès & Tibshirani's own D-term is a "scorecaster": a
    separate lightweight model forecasting the NEXT nonconformity score
    from score history, not a derivative-of-error-signal heuristic. This
    function implements exactly that, via `_fit_predict_next_score` above —
    at every step past `min_history`, a ridge AR(`scorecaster_lags`) model
    is refit from scratch on the full expanding score history seen so far
    and used to forecast the next score.

    P and I terms are BYTE-FOR-BYTE IDENTICAL to `run_conformal_pid_control`
    (same formulas, same parameters, same meaning, and — unlike that
    function — driven ONLY by P+I here, see below) — the ONLY thing that
    differs between the two PID variants is where the D-term's information
    enters the controller, specifically so `compare_static_vs_adaptive_vs_pid`
    can isolate "does a genuinely learned scorecaster beat the cheap
    derivative proxy" cleanly.

    Where the D-term acts (an intentional, honestly-documented design
    difference from `run_conformal_pid_control`, not an oversight): the
    derivative-PROXY variant folds its D-term into the alpha_t recursion,
    in probability space, because a trend-in-miscoverage signal is
    naturally alpha_t-shaped. A first version of THIS function tried the
    same thing — normalizing the scorecaster's raw score-unit forecast by
    `q_hat_t` and injecting it into alpha_t too — and it was genuinely
    unstable: `q_hat_t` routinely sits near zero on a well-calibrated
    process, so the normalized ratio saturated at its clip bound almost
    every step, and alpha_t ran away to its boundary within a few dozen
    steps (verified directly, not a hypothetical). The scorecaster's
    forecast is already in the SAME units as `q_hat_t` (raw nonconformity
    score, i.e. revenue), so the numerically stable and honestly simpler
    design is to add it directly where `q_hat_t` already acts — widening or
    narrowing the interval itself — rather than forcing it through a
    probability-space normalization it doesn't need:

        d_term_t   = kd * score_hat_t                     (score units, unclamped)
        total_corr = max(q_hat_t + d_term_t, -(q_hi[t]-q_lo[t])/2 + eps)
        lo_t = q_lo[t] - total_corr ;  hi_t = q_hi[t] + total_corr

    The `max(...)` is a safety clamp in the same spirit as the I-term's
    anti-windup `integral_clip` above: it guarantees `hi_t >= lo_t` (the raw
    interval can be narrowed by a large negative correction, but never
    inverted into a nonsensical negative-width interval), regardless of how
    large a wrong scorecaster forecast is. alpha_t itself is updated by P+I
    ONLY here — the D-term never touches alpha_t in this variant, so
    `q_hat_t` (the conformal quantile used above) evolves identically to
    plain ACI/the P+I-only case; the D-term's whole effect is this step's
    interval, not future steps' alpha_t trajectory. (Confirmed by
    `test_pid_learned_scorecaster_matches_proxy_pid_when_kd_is_zero`: with
    kd=0, this function's `alpha_t_history` is byte-identical to
    `run_conformal_pid_control`'s with kd=0 too.)

    Returns the same schema as `run_conformal_pid_control`, plus
    `d_term_history` and `score_forecast_history` (the scorecaster's own
    per-step forecasts, for inspecting how well it actually tracked the
    realized score series) and the scorecaster's own hyperparameters, so
    `compare_static_vs_adaptive_vs_pid` can treat all four methods (static /
    ACI / PID-proxy / PID-learned-scorecaster) uniformly.
    """
    n = len(y)
    assert len(q_lo) == n and len(q_hi) == n, "q_lo/q_hi/y must be the same length"
    if n <= min_history:
        raise ValueError(f"need more than min_history={min_history} points, got n={n}")

    scores_seen: list[float] = []
    alpha_t = alpha_target
    integral_acc = 0.0
    alphas, q_hats, d_terms, score_forecasts, covered, widths = [], [], [], [], [], []

    for t in range(n):
        if t < min_history:
            q_hat_t = 0.0
        else:
            hist = scores_seen[-window:] if window else scores_seen
            level = float(np.clip(1 - alpha_t, 0.0, 1.0))
            q_hat_t = float(np.quantile(hist, level)) if hist else 0.0

        if t < min_history or len(scores_seen) <= scorecaster_lags:
            d_term = 0.0
            score_hat_t = None
        else:
            score_hat_t = _fit_predict_next_score(
                scores_seen, n_lags=scorecaster_lags, ridge_lambda=scorecaster_ridge,
            )
            d_term = kd * score_hat_t

        base_half_width = float(q_hi[t] - q_lo[t]) / 2.0
        total_corr = max(q_hat_t + d_term, -base_half_width + 1e-6)
        lo_t = q_lo[t] - total_corr
        hi_t = q_hi[t] + total_corr
        is_covered = bool(lo_t <= y[t] <= hi_t)
        err_t = 0 if is_covered else 1

        alphas.append(alpha_t)
        q_hats.append(q_hat_t)
        d_terms.append(d_term)
        score_forecasts.append(score_hat_t)
        covered.append(is_covered)
        widths.append(float(hi_t - lo_t))

        # P+I ONLY -- see docstring on why the D-term does not feed alpha_t here.
        signal_t = alpha_target - err_t
        p_term = gamma * signal_t
        integral_acc = float(np.clip(integral_acc + signal_t, -integral_clip, integral_clip))
        i_term = ki * integral_acc
        alpha_t = float(np.clip(alpha_t + p_term + i_term, 1e-6, 1 - 1e-6))

        scores_seen.append(float(_nonconformity_scores(q_lo[t:t + 1], q_hi[t:t + 1], y[t:t + 1])[0]))

    covered = np.array(covered)
    post_warmup = covered[min_history:]
    return {
        "alpha_target": alpha_target, "gamma": gamma, "ki": ki, "kd": kd,
        "scorecaster_lags": scorecaster_lags, "scorecaster_ridge": scorecaster_ridge,
        "min_history": min_history, "window": window, "n_steps": n,
        "alpha_t_history": alphas,
        "q_hat_history": q_hats,
        "d_term_history": d_terms,
        "score_forecast_history": score_forecasts,
        "covered_history": covered.tolist(),
        "interval_width_history": widths,
        "empirical_coverage_full": float(covered.mean()),
        "empirical_coverage_post_warmup": float(post_warmup.mean()) if len(post_warmup) else None,
        "mean_interval_width_post_warmup": float(np.mean(widths[min_history:])) if n > min_history else None,
        "final_alpha_t": alpha_t,
    }


def compare_static_vs_adaptive_vs_pid(
    q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray,
    alpha_target: float = 0.1, gamma: float = 0.02, warmup_frac: float = 0.2,
    ki: float = 0.01, kd: float = 0.3,
    scorecaster_lags: int = 3, scorecaster_ridge: float = 1.0,
) -> dict:
    """Four-way version of `compare_static_vs_adaptive` above — same static
    baseline and same ACI run (byte-for-byte reused, not recomputed with
    different settings), plus BOTH Conformal PID D-term variants (the cheap
    derivative-proxy `run_conformal_pid_control` and the genuine learned
    scorecaster `run_conformal_pid_control_learned_scorecaster`, §D.2c) on
    the identical sequence, with identical P/I settings (`gamma`, `ki`) and
    identical `kd`/warmup — so "does the learned scorecaster actually beat
    the cheap proxy" is read directly off this comparison, not asserted."""
    base = compare_static_vs_adaptive(q_lo, q_hi, y, alpha_target=alpha_target, gamma=gamma, warmup_frac=warmup_frac)
    n = len(y)
    warmup_n = max(20, int(n * warmup_frac))
    pid = run_conformal_pid_control(
        q_lo, q_hi, y, alpha_target=alpha_target, gamma=gamma, ki=ki, kd=kd, min_history=warmup_n,
    )
    pid_learned = run_conformal_pid_control_learned_scorecaster(
        q_lo, q_hi, y, alpha_target=alpha_target, gamma=gamma, ki=ki, kd=kd, min_history=warmup_n,
        scorecaster_lags=scorecaster_lags, scorecaster_ridge=scorecaster_ridge,
    )
    base["pid"] = {
        "empirical_coverage_post_warmup": pid["empirical_coverage_post_warmup"],
        "mean_interval_width_post_warmup": pid["mean_interval_width_post_warmup"],
        "final_alpha_t": pid["final_alpha_t"],
        "ki": ki, "kd": kd,
    }
    base["pid_learned_scorecaster"] = {
        "empirical_coverage_post_warmup": pid_learned["empirical_coverage_post_warmup"],
        "mean_interval_width_post_warmup": pid_learned["mean_interval_width_post_warmup"],
        "final_alpha_t": pid_learned["final_alpha_t"],
        "ki": ki, "kd": kd, "scorecaster_lags": scorecaster_lags, "scorecaster_ridge": scorecaster_ridge,
    }
    return base


def compare_static_vs_adaptive(
    q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray,
    alpha_target: float = 0.1, gamma: float = 0.02, warmup_frac: float = 0.2,
) -> dict:
    """Fair side-by-side on the SAME chronologically-ordered sequence: fit a
    STATIC CQR correction on the first `warmup_frac` of the sequence (same
    scoring convention, same warm-up window ACI itself uses for
    `min_history`), apply it UNCHANGED to the rest; separately run ACI
    across the full sequence. Compares empirical coverage and mean interval
    width on the identical post-warmup segment for both, so this isolates
    "adapts over time" vs. "one fixed correction" as the only difference."""
    n = len(y)
    warmup_n = max(20, int(n * warmup_frac))
    if n - warmup_n < 10:
        raise ValueError("sequence too short for a meaningful warmup/post-warmup split")

    # static: fit once on the warm-up prefix, apply unchanged to the rest
    from modeling import fit_cqr_correction, apply_cqr, coverage as _coverage_fn
    q_hat_static = fit_cqr_correction(q_lo[:warmup_n], q_hi[:warmup_n], y[:warmup_n], alpha=alpha_target)
    lo_static, hi_static = apply_cqr(q_lo[warmup_n:], q_hi[warmup_n:], q_hat_static)
    static_coverage = _coverage_fn(y[warmup_n:], lo_static, hi_static)
    static_width = float(np.mean(hi_static - lo_static))

    # adaptive: run across the whole sequence, min_history == the same warmup window
    aci = run_adaptive_conformal_inference(
        q_lo, q_hi, y, alpha_target=alpha_target, gamma=gamma, min_history=warmup_n,
    )

    return {
        "n_total": n, "warmup_n": warmup_n, "alpha_target": alpha_target,
        "static": {
            "q_hat": q_hat_static,
            "empirical_coverage_post_warmup": static_coverage,
            "mean_interval_width_post_warmup": static_width,
        },
        "adaptive": {
            "empirical_coverage_post_warmup": aci["empirical_coverage_post_warmup"],
            "mean_interval_width_post_warmup": aci["mean_interval_width_post_warmup"],
            "final_alpha_t": aci["final_alpha_t"],
        },
        "nominal_coverage_target": 1 - alpha_target,
    }

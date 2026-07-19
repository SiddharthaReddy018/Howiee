"""
train.py
========
Offline training entry point. Not part of `run.sh` (which only does
feature-generation + inference) — this is what produces `pickle/model.pkl`.

Pipeline: ingest (§A) -> origin-based features (§B) -> chronological
train/calibration/final-holdout split -> walk-forward + grouped CV (§C.6/§G.2)
-> Tweedie point model (§C.1) + quantile ensemble (§C.2) -> CQR calibration
(§D.1) -> ONE honest final-holdout evaluation (§G.2.4) -> reliability diagram
(§D.3) -> Hill curves (§F) + ROAS sanity bounds (§G.3) computed on the full
canonical dataset -> out-of-fold predictions per horizon for hierarchical
reconciliation (§E) -> everything serialized into a single model bundle.

Usage:
    python3 src/train.py --data-dir ./data --model-out ./pickle/model.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from schema_mapper import ingest_directory, print_ingestion_log, validate_canonical_schema, validate_campaign_consistency
from feature_engineering import (
    build_training_frame, FEATURE_NAMES, CATEGORICAL_FEATURES,
    detect_anomalous_segments, compute_anomaly_weights,
)
import modeling as M
import adaptive_conformal as AC
import reconciliation as R
from budget_curves import fit_hill_curves
from sanity_clamps import compute_roas_bounds

HORIZONS = (30, 60, 90)
CQR_PAIRS = [(0.05, 0.95, 0.10), (0.10, 0.90, 0.20), (0.25, 0.75, 0.50)]  # (lo, hi, alpha)


def _chronological_split(frame: pd.DataFrame, train_q: float = 0.70, calib_q: float = 0.85):
    dates = pd.to_datetime(frame["origin_date"])
    resolved = dates + pd.to_timedelta(frame["horizon_days"], unit="D")
    train_cutoff = dates.quantile(train_q)
    calib_cutoff = dates.quantile(calib_q)

    train_mask = (resolved <= train_cutoff).to_numpy()
    calib_mask = ((dates > train_cutoff) & (resolved <= calib_cutoff)).to_numpy()
    holdout_mask = (dates > calib_cutoff).to_numpy()
    print(f"  split cutoffs: train_cutoff={train_cutoff.date()}  calib_cutoff={calib_cutoff.date()}")
    print(f"  rows: train={train_mask.sum()}  calibration={calib_mask.sum()}  final_holdout={holdout_mask.sum()}")
    return train_mask, calib_mask, holdout_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--model-out", default="./pickle/model.pkl")
    ap.add_argument("--origin-stride", type=int, default=1)
    ap.add_argument("--num-boost-round", type=int, default=500)
    ap.add_argument("--calibration-max-snapshots", type=int, default=None,
                     help="Dev-only speed knob for §8c's per-level calibration re-audit -- subsamples this "
                          "many origin_dates per horizon instead of every one. Leave unset for a real/final "
                          "run; every number in the delivered docs was produced with this unset.")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 80)
    print("§A — ingesting + schema-mapping raw files")
    print("=" * 80)
    canonical_df, ingestion_reports = ingest_directory(args.data_dir)
    print_ingestion_log(ingestion_reports)
    pandera_errors = validate_canonical_schema(canonical_df)
    print(f"\nPandera validation errors on canonical frame: {len(pandera_errors)}")
    consistency_issues = validate_campaign_consistency(canonical_df)
    print(f"Campaign consistency issues on canonical frame: {len(consistency_issues)}")
    for issue in consistency_issues:
        print(f"  - {issue}")

    print("\n" + "=" * 80)
    print("§B — building origin-based aggregate-window training frame")
    print("=" * 80)
    frame = build_training_frame(canonical_df, horizons=HORIZONS, origin_stride=args.origin_stride)
    print(f"  training frame: {frame.shape}")

    print("\n" + "=" * 80)
    print("Part-2-review §7 — detecting anomalous segments (generic, not hardcoded)")
    print("=" * 80)
    anomalous_segments = detect_anomalous_segments(canonical_df)
    frame["anomaly_weight"] = compute_anomaly_weights(frame, anomalous_segments)
    n_downweighted = int((frame["anomaly_weight"] < 1.0).sum())
    print(f"  {len(anomalous_segments)} anomalous segment(s) detected across all (channel, campaign_type) "
          f"groups; {n_downweighted}/{len(frame)} training rows have a target window overlapping one")
    for seg in anomalous_segments:
        print(f"    {seg['channel']}/{seg['campaign_type']}  {seg['start_date']} .. {seg['end_date']}  "
              f"(n_days={seg['n_days']}, z={seg['max_z_score']}, weight={seg['suggested_weight']})")

    train_mask, calib_mask, holdout_mask = _chronological_split(frame)
    X_all = frame[FEATURE_NAMES]
    y_all = frame["target_revenue"].to_numpy()

    frame_trainval = frame[train_mask | calib_mask].reset_index(drop=True)
    frame_train = frame[train_mask].reset_index(drop=True)
    frame_calib = frame[calib_mask].reset_index(drop=True)
    frame_holdout = frame[holdout_mask].reset_index(drop=True)

    print("\n" + "=" * 80)
    print("§C.6/§G.2 — development CV: walk-forward AND grouped-by-campaign")
    print("=" * 80)
    wf_splits_dev = M.walk_forward_splits(frame_trainval, n_splits=4)
    grp_splits_dev = M.grouped_campaign_splits(frame_trainval, n_splits=5)
    X_trainval = frame_trainval[FEATURE_NAMES]
    y_trainval = frame_trainval["target_revenue"].to_numpy()

    wf_report = M.cross_validate(X_trainval, y_trainval, wf_splits_dev, weight=frame_trainval["anomaly_weight"].to_numpy(),
                                  tag="walk_forward", num_boost_round=args.num_boost_round)
    grp_report = M.cross_validate(X_trainval, y_trainval, grp_splits_dev, weight=frame_trainval["anomaly_weight"].to_numpy(),
                                   tag="grouped_by_campaign", num_boost_round=args.num_boost_round)
    print(f"  walk-forward CRPS={wf_report['crps']:.2f}  WAPE={wf_report['wape_median']:.3f}")
    print(f"  grouped     CRPS={grp_report['crps']:.2f}  WAPE={grp_report['wape_median']:.3f}")
    overfit_gap = (grp_report["crps"] - wf_report["crps"]) / wf_report["crps"] if wf_report["crps"] else None
    print(f"  overfitting readout (grouped-vs-walkforward CRPS relative gap): "
          f"{overfit_gap:.1%}" if overfit_gap is not None else "  overfitting readout: n/a")

    print("\n" + "=" * 80)
    print("§C.1/§C.2 — fitting TRAIN-only models (for CQR calibration scores)")
    print("=" * 80)
    X_train, y_train = frame_train[FEATURE_NAMES], frame_train["target_revenue"].to_numpy()
    wf_splits_train = M.walk_forward_splits(frame_train, n_splits=3)
    tuning_split_train = wf_splits_train[-1] if wf_splits_train else (
        np.arange(len(X_train) * 8 // 10), np.arange(len(X_train) * 8 // 10, len(X_train)))
    # Full fold list for the multi-fold-averaged hyperparameter/power
    # selection in train_quantile_ensemble/train_point_model (falls back to
    # the single split above, wrapped, if walk-forward couldn't produce
    # multiple folds on this slice) -- train_hurdle_model still gets just
    # the one split below, since it's a losing-so-far ablation candidate,
    # not the shipped point model, and not worth doubling its cost too.
    tuning_splits_train = wf_splits_train if wf_splits_train else [tuning_split_train]
    models_trainonly, cfg = M.train_quantile_ensemble(
        X_train, y_train, tuning_splits_train, weight=frame_train["anomaly_weight"].to_numpy(),
        num_boost_round=args.num_boost_round, verbose=True)

    print("\n§D.1 — computing CQR corrections on the calibration slice")
    X_calib, y_calib = frame_calib[FEATURE_NAMES], frame_calib["target_revenue"].to_numpy()
    q_calib = M.predict_quantiles(models_trainonly, X_calib)
    cqr_corrections = {}
    for lo, hi, alpha in CQR_PAIRS:
        lo_idx, hi_idx = M.QUANTILES.index(lo), M.QUANTILES.index(hi)
        q_hat = M.fit_cqr_correction(q_calib[:, lo_idx], q_calib[:, hi_idx], y_calib, alpha=alpha)
        cqr_corrections[(lo, hi)] = q_hat
        print(f"  CQR correction for ({lo},{hi}) alpha={alpha}: Q_hat={q_hat:.2f}")

    print("\n" + "=" * 80)
    print("§C.1 ablation — Tweedie vs. hurdle (classifier x Gamma) vs. CatBoost Tweedie, "
          "+ an equal-weight blend of all three")
    print("=" * 80)
    print("  train-only fits of every candidate, evaluated on the untouched calibration slice --")
    print("  same eval window CQR itself uses, and never the final holdout (§G.2.4).")
    point_model_trainonly, _, _ = M.train_point_model(
        X_train, y_train, tuning_splits_train, weight=frame_train["anomaly_weight"].to_numpy(),
        num_boost_round=args.num_boost_round, verbose=False)
    hurdle_trainonly = M.train_hurdle_model(
        X_train, y_train, tuning_split_train, weight=frame_train["anomaly_weight"].to_numpy(),
        num_boost_round=args.num_boost_round, verbose=True)
    # CatBoost gets the SAME single-split economy as hurdle (not the full
    # multi-fold `tuning_splits_train`) -- it's a losing-so-far ablation
    # candidate at this point in the pipeline too, not the shipped model,
    # and the sweep-over-5-variance-powers cost is already paid once here
    # deliberately (see train_catboost_point_model's docstring on why the
    # sweep itself still needs to be fair/apples-to-apples with the
    # LightGBM sweep, even though the outer split count is trimmed for cost
    # the same way hurdle's already is).
    catboost_trainonly, catboost_p_train, _ = M.train_catboost_point_model(
        X_train, y_train, tuning_split_train, weight=frame_train["anomaly_weight"].to_numpy(),
        num_boost_round=args.num_boost_round, verbose=True)
    tweedie_calib_pred = np.clip(point_model_trainonly.predict(X_calib), 0, None)
    hurdle_calib_pred = M.predict_hurdle(hurdle_trainonly, X_calib)
    catboost_calib_pred = M.predict_catboost(catboost_trainonly, X_calib)
    point_model_ablation = M.compare_point_models_pinball_multi(
        y_calib, {"tweedie": tweedie_calib_pred, "hurdle": hurdle_calib_pred, "catboost": catboost_calib_pred},
    )
    _losses = point_model_ablation["pinball_q50_by_candidate"]
    print("  pinball(q0.5) by candidate: " +
          "  ".join(f"{name}={loss:.2f}" for name, loss in _losses.items()))
    print(f"  winner={point_model_ablation['winner']} "
          f"(+{point_model_ablation['relative_improvement_of_winner_vs_tweedie']:.1%} vs. tweedie)"
          if point_model_ablation['relative_improvement_of_winner_vs_tweedie'] is not None else "")

    print("\n" + "=" * 80)
    print("§C.5 — refitting PRODUCTION models on train+calibration combined")
    print("=" * 80)
    wf_splits_trainval = M.walk_forward_splits(frame_trainval, n_splits=3)
    tuning_split_trainval = wf_splits_trainval[-1] if wf_splits_trainval else tuning_split_train
    tuning_splits_trainval = wf_splits_trainval if wf_splits_trainval else tuning_splits_train
    quantile_models, prod_cfg = M.train_quantile_ensemble(
        X_trainval, y_trainval, tuning_splits_trainval, weight=frame_trainval["anomaly_weight"].to_numpy(),
        num_boost_round=args.num_boost_round, verbose=True)
    point_model, best_p, tweedie_sweep = M.train_point_model(
        X_trainval, y_trainval, tuning_splits_trainval, weight=frame_trainval["anomaly_weight"].to_numpy(),
        num_boost_round=args.num_boost_round, verbose=True)

    needs_hurdle_prod = point_model_ablation["winner"] in ("hurdle", "blend_equal_weight")
    needs_catboost_prod = point_model_ablation["winner"] in ("catboost", "blend_equal_weight")

    hurdle_models_prod = None
    if needs_hurdle_prod:
        print(f"  §C.1: hurdle needed for production (winner={point_model_ablation['winner']}) "
              "-> refitting hurdle on train+calibration")
        hurdle_models_prod = M.train_hurdle_model(
            X_trainval, y_trainval, tuning_split_trainval, weight=frame_trainval["anomaly_weight"].to_numpy(),
            num_boost_round=args.num_boost_round, verbose=True)

    catboost_model_prod = None
    if needs_catboost_prod:
        print(f"  §C.1: catboost needed for production (winner={point_model_ablation['winner']}) "
              "-> refitting catboost on train+calibration")
        catboost_model_prod, catboost_p_prod, _ = M.train_catboost_point_model(
            X_trainval, y_trainval, tuning_split_trainval, weight=frame_trainval["anomaly_weight"].to_numpy(),
            num_boost_round=args.num_boost_round, verbose=True)

    if not needs_hurdle_prod and not needs_catboost_prod:
        print("  §C.1: Tweedie remains the production point model "
              "(neither hurdle nor catboost beat it, alone or blended, on held-out pinball loss)")


    print("\n" + "=" * 80)
    print("§G.2.4 — ONE final holdout evaluation (never touched during tuning)")
    print("=" * 80)
    X_hold, y_hold = frame_holdout[FEATURE_NAMES], frame_holdout["target_revenue"].to_numpy()
    q_hold_raw = M.predict_quantiles(quantile_models, X_hold)

    q_hold_cqr = q_hold_raw.copy()
    for (lo, hi), q_hat in cqr_corrections.items():
        lo_idx, hi_idx = M.QUANTILES.index(lo), M.QUANTILES.index(hi)
        new_lo, new_hi = M.apply_cqr(q_hold_raw[:, lo_idx], q_hold_raw[:, hi_idx], q_hat)
        q_hold_cqr[:, lo_idx], q_hold_cqr[:, hi_idx] = new_lo, new_hi
    q_hold_cqr = M.fix_quantile_crossing(q_hold_cqr)

    median_idx = M.QUANTILES.index(0.5)
    holdout_metrics_raw = {
        "pinball_per_quantile": {q: M.pinball_loss(y_hold, q_hold_raw[:, i], q) for i, q in enumerate(M.QUANTILES)},
        "crps": M.crps_from_quantiles(y_hold, q_hold_raw, M.QUANTILES),
        "wape_median": M.wape(y_hold, q_hold_raw[:, median_idx]),
        "smape_median": M.smape(y_hold, q_hold_raw[:, median_idx]),
    }
    holdout_metrics_cqr = {
        "pinball_per_quantile": {q: M.pinball_loss(y_hold, q_hold_cqr[:, i], q) for i, q in enumerate(M.QUANTILES)},
        "crps": M.crps_from_quantiles(y_hold, q_hold_cqr, M.QUANTILES),
        "wape_median": M.wape(y_hold, q_hold_cqr[:, median_idx]),
        "smape_median": M.smape(y_hold, q_hold_cqr[:, median_idx]),
    }
    reliability = M.reliability_diagram(y_hold, q_hold_cqr, M.QUANTILES)
    print(f"  FINAL HOLDOUT (raw)  CRPS={holdout_metrics_raw['crps']:.2f}  WAPE={holdout_metrics_raw['wape_median']:.3f}")
    print(f"  FINAL HOLDOUT (CQR)  CRPS={holdout_metrics_cqr['crps']:.2f}  WAPE={holdout_metrics_cqr['wape_median']:.3f}")
    print(f"  reliability diagram: {reliability}")

    print("\n" + "=" * 80)
    print("§G.2.5 — naive baseline on the SAME final holdout (\"compared to what?\")")
    print("=" * 80)
    naive_pred_hold = M.naive_pace_forecast(X_hold)
    naive_wape = M.wape(y_hold, naive_pred_hold)
    naive_smape = M.smape(y_hold, naive_pred_hold)
    wape_improvement_vs_naive_pct = (
        1.0 - (holdout_metrics_cqr["wape_median"] / naive_wape) if naive_wape else None
    )
    naive_baseline_report = {
        "description": (
            "Naive 'continue at recent pace' baseline: trailing 28-day mean daily "
            "revenue (revenue_roll_mean_28) x horizon_days. Never trained on; "
            "scored on the identical holdout rows/metric as the production model above."
        ),
        "wape_median": naive_wape,
        "smape_median": naive_smape,
        "model_wape_improvement_vs_naive_pct": wape_improvement_vs_naive_pct,
    }
    print(f"  NAIVE BASELINE          WAPE={naive_wape:.3f}")
    if wape_improvement_vs_naive_pct is not None:
        print(f"  Model WAPE improvement vs. naive baseline: {wape_improvement_vs_naive_pct:+.1%}")

    holdout_by_horizon = {}
    for h in HORIZONS:
        hmask = (frame_holdout["horizon_days"] == h).to_numpy()
        if hmask.sum() < 5:
            continue
        naive_wape_h = M.wape(y_hold[hmask], naive_pred_hold[hmask])
        holdout_by_horizon[h] = {
            "n": int(hmask.sum()),
            "wape_median": M.wape(y_hold[hmask], q_hold_cqr[hmask, median_idx]),
            "crps": M.crps_from_quantiles(y_hold[hmask], q_hold_cqr[hmask], M.QUANTILES),
            "naive_baseline_wape_median": naive_wape_h,
            "model_wape_improvement_vs_naive_pct": (
                1.0 - (M.wape(y_hold[hmask], q_hold_cqr[hmask, median_idx]) / naive_wape_h)
                if naive_wape_h else None
            ),
        }

    print("\n" + "=" * 80)
    print("§D.2/§D.2b — Adaptive Conformal Inference & Conformal PID control (both optional")
    print("       per the plan; reported, not shipped as the default calibration -- see")
    print("       docs/technical_documentation.md §6a/§6b for why)")
    print("=" * 80)
    try:
        order = np.argsort(frame_holdout["origin_date"].to_numpy(), kind="stable")
        lo_i, hi_i = M.QUANTILES.index(0.05), M.QUANTILES.index(0.95)
        aci_report = AC.compare_static_vs_adaptive_vs_pid(
            q_hold_cqr[order, lo_i], q_hold_cqr[order, hi_i], y_hold[order],
            alpha_target=0.10, gamma=0.03, warmup_frac=0.2,
        )
        s, a, p = aci_report["static"], aci_report["adaptive"], aci_report["pid"]
        pl = aci_report["pid_learned_scorecaster"]
        print(f"  static   post-warmup coverage={s['empirical_coverage_post_warmup']:.3f}  "
              f"mean_width={s['mean_interval_width_post_warmup']:.1f}")
        print(f"  adaptive post-warmup coverage={a['empirical_coverage_post_warmup']:.3f}  "
              f"mean_width={a['mean_interval_width_post_warmup']:.1f}  final_alpha_t={a['final_alpha_t']:.3f}")
        print(f"  pid (derivative-proxy D)      post-warmup coverage={p['empirical_coverage_post_warmup']:.3f}  "
              f"mean_width={p['mean_interval_width_post_warmup']:.1f}  final_alpha_t={p['final_alpha_t']:.3f}")
        print(f"  pid (learned-scorecaster D)   post-warmup coverage={pl['empirical_coverage_post_warmup']:.3f}  "
              f"mean_width={pl['mean_interval_width_post_warmup']:.1f}  final_alpha_t={pl['final_alpha_t']:.3f}")
    except Exception as exc:
        print(f"  ACI/PID comparison skipped ({exc!r})")
        aci_report = None

    print("\n" + "=" * 80)
    print("§E prep — out-of-fold predictions per horizon (for reconciliation)")
    print("=" * 80)
    oof_by_horizon = {}
    for h in HORIZONS:
        fh = frame_trainval[frame_trainval["horizon_days"] == h].reset_index(drop=True)
        splits_h = M.walk_forward_splits(fh, n_splits=3)
        if not splits_h:
            oof_by_horizon[h] = pd.DataFrame()
            continue
        oof = M.generate_oof_median_predictions(fh, FEATURE_NAMES, splits_h,
                                                 weight=fh["anomaly_weight"].to_numpy(), num_boost_round=300)
        oof["horizon_days"] = h
        oof_by_horizon[h] = oof
        print(f"  horizon={h}: {len(oof)} OOF rows")

    print("\n" + "=" * 80)
    print("§E re-audit — reconciled-band calibration by hierarchy level (total /")
    print("       channel / campaign_type / campaign), not just the base per-row")
    print("       level §6 already checks (Principato et al. 2024: coherence and")
    print("       calibration are separate properties -- coherence alone doesn't")
    print("       verify this)")
    print("=" * 80)
    holdout_for_calibration = frame_holdout[
        ["campaign_id", "channel", "campaign_type", "origin_date", "horizon_days", "target_revenue"]
    ].copy()
    for i, q in enumerate(M.QUANTILES):
        holdout_for_calibration[f"q{q}"] = q_hold_cqr[:, i]
    try:
        reconciled_calibration_by_level = R.evaluate_reconciled_calibration(
            holdout_for_calibration, oof_by_horizon, quantiles=M.QUANTILES,
            max_snapshots_per_horizon=args.calibration_max_snapshots,
        )
        for level in ("total", "channel", "campaign_type", "campaign"):
            lv = reconciled_calibration_by_level[level]
            bands = "  ".join(
                f"{b} nominal={lv[b]['nominal']:.2f} empirical={lv[b]['empirical']:.3f}"
                for b in ("90%", "80%", "50%") if b in lv
            )
            print(f"  {level:<14s} n_observations={lv['n_observations']:<6d} {bands}")
        print(f"  ({reconciled_calibration_by_level['n_snapshots']} origin_date/horizon snapshots "
              f"reconciled, {reconciled_calibration_by_level['n_snapshots_skipped']} skipped)")
    except Exception as exc:
        print(f"  reconciled calibration-by-level check skipped ({exc!r})")
        reconciled_calibration_by_level = None

    print("\n" + "=" * 80)
    print("§F — fitting Hill saturation curves (channel x campaign_type)")
    print("=" * 80)
    hill_curves = fit_hill_curves(canonical_df)
    n_fit_ok = sum(1 for c in hill_curves.values() if c["fit_ok"])
    n_recency_weighted = sum(1 for c in hill_curves.values() if c.get("recency_weighted"))
    print(f"  fit_ok for {n_fit_ok}/{len(hill_curves)} groups "
          f"({n_recency_weighted} recency-weighted, half_life="
          f"{next(iter(hill_curves.values()), {}).get('recency_half_life_days')} days)")

    print("\n" + "=" * 80)
    print("§G.3 — computing per-(channel, campaign_type) ROAS plausibility bounds")
    print("=" * 80)
    roas_bounds = compute_roas_bounds(canonical_df)

    feature_importance = M.feature_importance(quantile_models)
    print("\nTop features by gain:")
    for f in feature_importance:
        print(f"  {f['importance_rank']}. {f['feature']} (gain={f['gain']:.1f})")

    print("\n" + "=" * 80)
    print("§H.1 — computing SHAP feature importance (median quantile booster)")
    print("=" * 80)
    shap_sample = X_hold if len(X_hold) > 0 else X_trainval
    shap_sample_frame = frame_holdout if len(X_hold) > 0 else frame_trainval
    try:
        shap_importance = M.shap_feature_importance(
            quantile_models[0.5], shap_sample, top_n=10,
            groups=shap_sample_frame["channel"],
        )
        print(f"  sampled {shap_importance['n_rows_sampled']} rows; top features by mean |SHAP|:")
        for f in shap_importance["top_features"]:
            print(f"    {f['importance_rank']}. {f['feature']} "
                  f"(mean|SHAP|={f['mean_abs_shap']:.1f}, signed={f['mean_signed_shap']:+.1f})")
        if shap_importance.get("by_group"):
            print(f"  per-channel SHAP breakdown computed for: {list(shap_importance['by_group'].keys())}")
    except Exception as exc:
        print(f"  SHAP computation failed ({exc!r}); bundle will fall back to gain-based importance only")
        shap_importance = None

    bundle = {
        "version": "aignition3.0-v2",
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "feature_names": FEATURE_NAMES,
        "categorical_features": CATEGORICAL_FEATURES,
        "quantiles": M.QUANTILES,
        "horizons": list(HORIZONS),
        "quantile_models": quantile_models,
        "point_model": point_model,
        "point_model_tweedie_p": best_p,
        "tweedie_sweep_log": tweedie_sweep,
        "point_model_selected": point_model_ablation["winner"],
        "point_model_ablation": point_model_ablation,
        "hurdle_models": hurdle_models_prod,
        "catboost_model": catboost_model_prod,
        "quantile_model_config": prod_cfg,
        "cqr_corrections": cqr_corrections,
        "hill_curves": hill_curves,
        "roas_bounds": roas_bounds,
        "anomalous_segments": anomalous_segments,
        "feature_importance": feature_importance,
        "shap_importance": shap_importance,
        "oof_by_horizon": oof_by_horizon,
        "cv_reports": {"walk_forward": wf_report, "grouped_by_campaign": grp_report,
                       "overfit_gap_relative": overfit_gap},
        "adaptive_conformal_report": aci_report,
        "final_holdout": {
            "raw": holdout_metrics_raw, "cqr_calibrated": holdout_metrics_cqr,
            "naive_baseline": naive_baseline_report,
            "reliability_diagram": reliability, "by_horizon": holdout_by_horizon,
            "reconciled_calibration_by_level": reconciled_calibration_by_level,
            "n_rows": int(len(frame_holdout)),
            "date_range": [str(frame_holdout["origin_date"].min().date()),
                           str(frame_holdout["origin_date"].max().date())] if len(frame_holdout) else None,
        },
        "ingestion_reports": {k: v.to_dict() for k, v in ingestion_reports.items()},
        "pandera_errors_on_train_data": pandera_errors,
        "campaign_consistency_issues_on_train_data": consistency_issues,
        "training_frame_shape": list(frame.shape),
    }

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(bundle, args.model_out, compress=3)
    size_mb = os.path.getsize(args.model_out) / 1e6
    print(f"\nSaved model bundle to {args.model_out} ({size_mb:.1f} MB)")
    print(f"Total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

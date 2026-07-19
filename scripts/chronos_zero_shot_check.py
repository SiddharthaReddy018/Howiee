"""
scripts/chronos_zero_shot_check.py
====================================
§5a re-audit, 2026 literature — a zero-shot foundation-model cross-check
against the account-total daily revenue series, in the SAME honestly-
reported style as the adstock ablation (§4): run once, measure on the real
holdout, keep the finding whichever way it goes.

NOT RUN IN THE DELIVERED SUBMISSION. This needs `torch` + `chronos-forecasting`
(Amazon's zero-shot time-series foundation model, Oct 2025 release, the
"Chronos-2" checkpoint). The sandbox this project was developed in has a
disk quota too small for torch's default Linux wheel, which pulls in the
full CUDA dependency stack (~4GB+) even for pure CPU inference -- confirmed
directly: torch 2.13.0's `_load_global_deps()` unconditionally tries to
preload `libcublasLt.so` and fails at IMPORT time (not at `.cuda()` call
time) if the nvidia-* packages aren't present, so there's no lighter
CPU-only install path available from a plain PyPI install in this
environment. This is an environment constraint, not a decision to skip the
comparison -- run this script on a machine with ~6GB of free disk (a laptop,
Colab, or any normal dev box) to get the actual number.

Honest framing, unchanged from the original assessment: Chronos-2 (unlike
the original Chronos) DOES support covariate-conditioned zero-shot
forecasting as of its Oct 2025 release, so the earlier blanket claim that
"no zero-shot model can condition on a planned future budget" is now
partially outdated. This script still runs the plain UNCONDITIONAL mode
(extrapolate history only) for simplicity and because it's the mode every
zero-shot foundation model supports without exception -- a fair baseline
comparison, not a best-case one. If you have time after running this,
`ChronosPipeline.predict_quantiles` also accepts `future_covariates` in the
2.x API for a fairer, covariate-aware comparison; left as a documented next
step here rather than attempted, to keep this script simple and correct
over trying to be maximally fair on a method with no representation in the
actual submission.

Usage
-----
    pip install chronos-forecasting          # pulls in torch; needs ~6GB free
    python3 scripts/chronos_zero_shot_check.py --data-dir ./data

Prints a plain comparison table: naive-pace baseline WAPE vs. Chronos-2
zero-shot WAPE vs. (if you paste in the number from your own reliability.json)
this project's production model WAPE, all on the same held-out window.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_account_total_daily_series(data_dir: str) -> pd.DataFrame:
    """Reuses this project's own schema mapper -- same canonical ingestion
    path as train.py, not a separate ad hoc CSV read -- then aggregates to
    one row per calendar date, summed across every channel/campaign."""
    from schema_mapper import ingest_directory  # project's own ingestion entrypoint

    canonical_df, _reports = ingest_directory(data_dir)
    daily = (
        canonical_df.groupby("date", as_index=False)["revenue"].sum()
        .sort_values("date").reset_index(drop=True)
    )
    return daily


def naive_pace_wape(train_rev: np.ndarray, test_rev: np.ndarray, window: int) -> float:
    pace = train_rev[-28:].mean() if len(train_rev) >= 28 else train_rev.mean()
    pred = np.full(window, pace)
    actual = test_rev[:window]
    return float(np.sum(np.abs(actual - pred)) / max(np.sum(np.abs(actual)), 1e-9))


def chronos_zero_shot_wape(train_rev: np.ndarray, test_rev: np.ndarray, window: int) -> float:
    import torch
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32,
    )
    context = torch.tensor(train_rev, dtype=torch.float32)
    forecast = pipeline.predict(context=context, prediction_length=window)
    median_path = np.quantile(forecast[0].numpy(), 0.5, axis=0)  # per-day median path
    actual = test_rev[:window]
    pred_total = median_path.sum()
    actual_total = actual.sum()
    return float(abs(actual_total - pred_total) / max(abs(actual_total), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--windows", type=int, nargs="+", default=[30, 60, 90])
    args = ap.parse_args()

    daily = load_account_total_daily_series(args.data_dir)
    rev = daily["revenue"].to_numpy(dtype=float)

    max_window = max(args.windows)
    if len(rev) <= max_window + 28:
        raise SystemExit(f"Not enough history ({len(rev)} days) for a {max_window}-day held-out window.")

    train_rev, test_rev = rev[:-max_window], rev[-max_window:]

    print(f"{'window':>8} | {'naive WAPE':>12} | {'chronos-2 WAPE':>15}")
    print("-" * 42)
    for w in sorted(args.windows):
        naive = naive_pace_wape(train_rev, test_rev, w)
        try:
            chronos = chronos_zero_shot_wape(train_rev, test_rev, w)
            chronos_str = f"{chronos:.1%}"
        except ImportError:
            chronos_str = "torch/chronos not installed"
        print(f"{w:>8} | {naive:>11.1%} | {chronos_str:>15}")

    print(
        "\nCompare against this project's own production-model WAPE by horizon in "
        "output/reliability.json -> final_holdout.by_horizon (this script deliberately "
        "doesn't hardcode that number so it can't silently drift out of sync with "
        "whichever model.pkl you've actually trained)."
    )


if __name__ == "__main__":
    main()

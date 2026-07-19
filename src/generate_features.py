"""
generate_features.py
=====================
Stage 1 of `run.sh`. Ingests raw CSVs from --data-dir through the schema
robustness layer (§A), builds the live forecast-origin feature snapshot
(§B's `build_latest_snapshot`), expands it to one row per (campaign, horizon)
for every horizon the trained model bundle supports, and writes the result
to --out (features.parquet) for predict.py to score.

The "planned future daily budget" scenario defaults to each campaign's
trailing 28-day mean spend (§B.5's "continue at the recent pace" baseline) —
predict.py's budget "what-if" scenario support layers on top of this default
at inference/frontend time, it does not change what's written here.
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from schema_mapper import ingest_directory
from feature_engineering import build_latest_snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    horizons = bundle.get("horizons", [30, 60, 90])

    canonical_df, reports = ingest_directory(args.data_dir)
    snapshot = build_latest_snapshot(canonical_df)

    rows = []
    for h in horizons:
        chunk = snapshot.copy()
        chunk["horizon_days"] = h
        # baseline "continue at recent pace" scenario; the frontend/predict.py
        # budget slider overwrites this column per-scenario at serving time.
        chunk["planned_future_daily_budget"] = chunk["spend_roll_mean_28"].fillna(chunk["spend_lag_7"]).fillna(0.0)
        rows.append(chunk)
    multi = pd.concat(rows, ignore_index=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    multi.to_parquet(args.out, index=False)
    print(f"Wrote {len(multi)} feature rows ({len(snapshot)} campaigns x {len(horizons)} horizons) to {args.out}")


if __name__ == "__main__":
    main()

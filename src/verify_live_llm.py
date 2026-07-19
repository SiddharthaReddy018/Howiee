"""
verify_live_llm.py
====================
§H.3's open item: "the live LLM call path has never actually run end-to-end
-- no ANTHROPIC_API_KEY in the dev sandbox, only the mocked parsing/
validation logic has been exercised." This script is the one-command way to
close that out for real, against the real Anthropic API, using the exact
`llm_insights.generate_causal_summary` code path `predict.py` calls in
production -- nothing reimplemented.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python src/verify_live_llm.py

Reuses the already-trained bundle + already-generated predictions.csv (run
`run.sh` first if those don't exist yet) and calls the LLM exactly once, for
the "total" scope, to keep this cheap -- it is a smoke test, not a full
regeneration of causal_summary.json (that still happens automatically the
next time you run predict.py with the key exported, since
`generate_causal_summary` already reads ANTHROPIC_API_KEY from the
environment with no code changes needed).

Prints, and never silently swallows:
  - whether the response came back `source: "llm"` (not `rule_based_fallback`)
  - whether the grounding validator accepted it
  - live call latency
  - the actual generated summary text, so you can eyeball it before a demo
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from schema_mapper import ingest_directory
import llm_insights as L


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "pickle", "model.pkl"))
    ap.add_argument("--predictions", default=os.path.join(os.path.dirname(__file__), "..", "output", "predictions.csv"))
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--horizon", type=int, default=30)
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "API KEY is not set in this environment.\n"
            "Export a real key first:\n\n"
            "    export GROQ_API_KEY=gsk_...\n"
            "    python src/verify_live_llm.py\n\n"
            "(Script will now exit rather than gracefully falling back, so you don't "
            "mistake a fallback for a successful live run during verification.)"
        )
        sys.exit(1)

    for path, label in [(args.model, "model bundle"), (args.predictions, "predictions.csv")]:
        if not os.path.exists(path):
            print(f"No {label} at {path}. Run `bash run.sh` first.")
            sys.exit(1)

    print(f"Loading {args.model} and {args.predictions}...")
    bundle = joblib.load(args.model)
    predictions = pd.read_csv(args.predictions, dtype={"campaign_id": str})

    print(f"Re-ingesting {args.data_dir} for anomaly/period-over-period context...")
    canonical_df, _ = ingest_directory(args.data_dir)
    daily = canonical_df.assign(date=pd.to_datetime(canonical_df["date"]).dt.normalize())

    hpred = predictions[predictions["horizon_days"] == args.horizon]
    if len(hpred) == 0:
        print(f"No predictions at horizon_days={args.horizon}.")
        sys.exit(1)

    shap_imp = bundle.get("shap_importance")
    global_drivers = shap_imp["top_features"] if shap_imp else bundle["feature_importance"]

    p10 = float(hpred["revenue_p10"].sum()) if "revenue_p10" in hpred else None
    p50 = float(hpred["revenue_p50"].sum())
    p90 = float(hpred["revenue_p90"].sum()) if "revenue_p90" in hpred else None
    total_spend = float(hpred["assumed_spend_total"].sum())
    roas_p50 = (p50 / total_spend) if total_spend > 0 else None

    pop = L.compute_period_over_period(daily, {}, window=args.horizon)
    anomalies_all = L.detect_anomalies(daily)[:5]

    ctx = L.build_grounding_context(
        scope={"channel": None, "campaign_type": None, "window_days": args.horizon},
        forecast={"revenue_p10": p10, "revenue_p50": p50, "revenue_p90": p90, "roas_p50": roas_p50},
        top_drivers=global_drivers,
        period_over_period=pop,
        anomalies=anomalies_all,
        saturation_status={"status": "mixed_across_channels"},
    )

    print("\nCalling the live API for the 'total' scope, 30-day window...")
    t0 = time.time()
    result = L.generate_causal_summary(ctx)
    latency = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"source:     {result['source']}")
    print(f"validated:  {result['validated']}")
    print(f"latency:    {latency:.2f}s")
    print(f"{'=' * 70}")
    print(f"\nsummary:\n  {result['summary']}")
    print(f"\nkey_drivers:")
    for d in result["key_drivers"]:
        print(f"  - {d}")
    print(f"\nrisk_flags:")
    for r in result["risk_flags"]:
        print(f"  - {r}")
    print(f"\nconfidence_note:\n  {result['confidence_note']}")
    print()

    if result["source"] == "llm":
        print(f"PASS -- live LLM call succeeded and passed the §H.4 grounding validator ({latency:.2f}s).")
        sys.exit(0)
    else:
        print(
            "NOT CONFIRMED -- generate_causal_summary fell back to the rule-based narrator "
            "even with a key set. Re-run with normal stdout/stderr visible: call_llm prints "
            "the specific exception (auth/rate-limit/network) or the validator's rejection "
            "reason on the line above rather than swallowing it silently."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()

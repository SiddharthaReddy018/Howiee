"""
validate.py
===========
Standalone CLI diagnostic: run the schema robustness layer (§A) against a
data directory and print the ingestion + Pandera validation report, without
needing to run the full train/predict pipeline.

Usage:
    python validate.py --data-dir ./data
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from schema_mapper import ingest_directory, print_ingestion_log, validate_canonical_schema, validate_campaign_consistency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args()

    print(f"Validating data directory: {args.data_dir}\n")
    canonical_df, reports = ingest_directory(args.data_dir)
    print_ingestion_log(reports)

    errors = validate_canonical_schema(canonical_df)
    print(f"\nPandera schema validation: {'PASSED' if not errors else f'{len(errors)} issue(s)'}")
    for e in errors[:20]:
        print(f"  - {e}")

    consistency_issues = validate_campaign_consistency(canonical_df)
    print(f"\nCampaign consistency validation: {'PASSED' if not consistency_issues else f'{len(consistency_issues)} issue(s)'}")
    for issue in consistency_issues:
        print(f"  - {issue}")

    n_skipped = sum(1 for r in reports.values() if r.errors)
    print(f"\nSummary: {len(reports)} file(s) seen, {n_skipped} skipped, "
          f"{len(canonical_df):,} canonical rows, {canonical_df['campaign_id'].nunique()} campaigns.")
    sys.exit(1 if n_skipped == len(reports) else 0)


if __name__ == "__main__":
    main()

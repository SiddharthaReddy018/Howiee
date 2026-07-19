#!/usr/bin/env bash
set -euo pipefail

# Load .env if present (optional — LLM narration only). Absence, an empty
# file, or unset keys are all fine: src/llm_insights.py falls back to the
# rule-based narrator whenever no key is set or the API call fails/times out
# (network or no network), so this block never blocks or fails the run.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Accept arguments, fall back to defaults for local runs
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

OUT_DIR="$(dirname "$OUTPUT_PATH")"
mkdir -p "$OUT_DIR"
FEATURES_PATH="$OUT_DIR/features.parquet"

# Resolve python binary for local testing compatibility
PYTHON="python3"
command -v python3 &>/dev/null || PYTHON="python"

# 0. Data-quality / schema-robustness report (Section A) - standalone
#    diagnostic, run first so a fundamentally unusable data directory
#    fails loudly here rather than deeper in the pipeline.
$PYTHON src/validate.py --data-dir "$DATA_DIR" | tee "$OUT_DIR/data_quality_report.txt"

# 1. Generate the features the model expects from the data
$PYTHON src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out "$FEATURES_PATH" \
    --model "$MODEL_PATH"

# 2. Load the pickled model and produce predictions (plus reconciled
#    hierarchy, data-health, reliability, hill curves, and grounded causal
#    summaries - see src/predict.py's module docstring for the full list)
$PYTHON src/predict.py \
    --features "$FEATURES_PATH" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    --data-dir "$DATA_DIR"

echo "Done. Predictions written to $OUTPUT_PATH"


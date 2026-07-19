#!/usr/bin/env bash
set -euo pipefail

echo "==> Setting up clean clone test directory..."
TEST_DIR="/tmp/aignition_clean_test_$(date +%s)"
mkdir -p "$TEST_DIR"

echo "==> Cloning local repository..."
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git clone "file://$REPO_DIR" "$TEST_DIR"

cd "$TEST_DIR"

echo "==> Creating fresh virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip >/dev/null

echo "==> Installing pinned requirements..."
pip install -r requirements.txt >/dev/null

echo "==> Verifying run.sh is executable..."
chmod +x run.sh

echo "==> Running the pipeline as the grading bot will..."
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv

echo "==> Test complete. Output files generated:"
ls -la ./output


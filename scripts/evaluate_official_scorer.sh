#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==============================================================================
# Configuration & Defaults
# ==============================================================================
PARQUET_FILE=${1:-"model_eval_logs/qwen3-8B-ocr-hipe_aggregated-meta_hipe_aggregated_test_meta_results.parquet"}
RAW_REF_DIR=${2:-"./HIPE-OCRepair-2026-data/data/v0.9/"}
OUTPUT_DIR="official_scorer_data"
FLAT_REF_DIR="${OUTPUT_DIR}/reference"
TEAM_NAME="TMDUONG"
RUN_NAME="run1"

# Extract just the base name of the parquet file to name our final JSON report
BASE_NAME=$(basename "$PARQUET_FILE" .parquet)
RESULT_FILE="${OUTPUT_DIR}/${BASE_NAME}_official_scores.json"

echo "==========================================================="
echo " HIPE-OCRepair Official Scorer Pipeline"
echo "==========================================================="
echo "Input Parquet: $PARQUET_FILE"
echo "Raw Ref Dir:   $RAW_REF_DIR"
echo "Team Name:     $TEAM_NAME ($RUN_NAME)"
echo "-----------------------------------------------------------"

# ==============================================================================
# Step 1: Clean Workspace
# ==============================================================================
echo "[1/4] Purging old files to prevent schema/ghost errors..."
mkdir -p "${OUTPUT_DIR}/hypothesis"
mkdir -p "$FLAT_REF_DIR"
rm -rf "${OUTPUT_DIR}/hypothesis"/*
rm -rf "${FLAT_REF_DIR}"/*

# ==============================================================================
# Step 2: Flatten Reference Directory
# ==============================================================================
echo "[2/4] Flattening reference test sets for the official scorer..."

find "$RAW_REF_DIR" -type f -name "*.jsonl" | \
    grep "test" | grep -v "masked" | grep -v "dev" | grep -v "train" | \
    xargs -I {} cp {} "$FLAT_REF_DIR"/

# ==============================================================================
# Step 3: Generate Hypotheses
# ==============================================================================
echo "[3/4] Generating strict schema-compliant JSONL splits..."
python scripts/generate_hypotheses.py \
  --input "$PARQUET_FILE" \
  --ref_dir "$RAW_REF_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --team "$TEAM_NAME" \
  --run "$RUN_NAME"

# ==============================================================================
# Step 4: Evaluate and Export
# ==============================================================================
echo "[4/4] Running official HIPE scorer and exporting metrics..."
hipe-ocrepair-scorer \
  --reference-dir "$FLAT_REF_DIR" \
  --hypothesis-dir "${OUTPUT_DIR}/hypothesis/" > "$RESULT_FILE"

echo "-----------------------------------------------------------"
echo "SUCCESS! Official evaluation completed."
echo "Results saved to: $RESULT_FILE"
echo "==========================================================="
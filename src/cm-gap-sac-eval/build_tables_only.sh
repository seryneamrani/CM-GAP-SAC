#!/usr/bin/env bash
# =============================================================================
# Build tables only — from existing results file, no eval re-run
# =============================================================================
set -euo pipefail

EVAL_ROOT="/home/seryne/limo_jazzy_ws/src/cm-gap-sac-eval"
RESULTS="${EVAL_ROOT}/outputs/results_backup_20260727_1157.jsonl"
OUTPUT_DIR="${EVAL_ROOT}/outputs/tables_finales"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Building tables"
echo "  Input:  ${RESULTS}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

python3 "${EVAL_ROOT}/analysis/build_tables.py" \
    --results "${RESULTS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "DONE — tables générées dans ${OUTPUT_DIR}/"
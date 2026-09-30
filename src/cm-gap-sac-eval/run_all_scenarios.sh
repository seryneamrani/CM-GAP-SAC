#!/usr/bin/env bash
# =============================================================================
# CM-GAP-SAC — Full eval runbook (7 scenarios × active methods)
# =============================================================================
# Runs whatever methods are currently ACTIVE (uncommented) in eval_config.yaml.
#
# Phase 1 workflow (current):
#   - eval_config.yaml has only cm_gap_sac_full uncommented
#   - Run this script → 700 episodes → results.jsonl + 5 tables
#
# Phase 2 workflow (later):
#   - Uncomment additional methods in eval_config.yaml (ablation_no_cbf, etc.)
#   - Rerun this script → 700 new episodes appended → results.jsonl + more tables
#     (build_tables.py auto-detects new methods and generates comparison tables)
#
# Prerequisites:
#   - Gazebo hospital world running
#   - ROS 2 workspace sourced (rclpy + cm_gap_sac_navigation importable)
#   - Nav2 planner running (since cm_gap_sac_full uses nav2_global_planner)
#
# Usage:
#   bash run_all_scenarios.sh /path/to/training_config.yaml
# =============================================================================

set -euo pipefail

TRAIN_CONFIG="${1:-/path/to/training_config.yaml}"
EVAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
EVAL_CONFIG="${EVAL_ROOT}/config/eval_config.yaml"
PED_CONFIG_DIR="${EVAL_ROOT}/configs/pedestrian_scenarios"
OUTPUT_DIR="${EVAL_ROOT}/outputs"
RESULTS="${OUTPUT_DIR}/results.jsonl"

mkdir -p "${OUTPUT_DIR}"

# All available scenarios
ALL_SCENARIOS=(
    S1_static_only
    S2_density3_low
    S3_density3_high
    S4_density5_low
    S5_density5_high
    S6_density7_low
    S7_density7_high
)

# If additional args passed after TRAIN_CONFIG, use them as scenario filter.
# Otherwise run all 7. Enables splitting across sessions:
#   ./run_all_scenarios.sh /path/train_config.yaml                        # all 7
#   ./run_all_scenarios.sh /path/train_config.yaml S1_static_only S2_density3_low
if [ $# -ge 2 ]; then
    SCENARIOS=("${@:2}")
    echo "[runbook] Filtered scenarios: ${SCENARIOS[*]}"
else
    SCENARIOS=("${ALL_SCENARIOS[@]}")
    echo "[runbook] Running ALL ${#SCENARIOS[@]} scenarios"
fi
echo ""

# ─── Pre-eval one-shot setup ─────────────────────────────────────────────────
echo "=========================================="
echo "PRE-EVAL SETUP"
echo "=========================================="

echo "[setup] Generating snapshots (7 PNG + overview)..."
python3 "${EVAL_ROOT}/generate_snapshots.py" \
    --config "${EVAL_CONFIG}" \
    --output-dir "${OUTPUT_DIR}/snapshots"

echo ""
echo "[setup] Generating pedestrian configs (7 YAML with YOUR waypoints)..."
python3 "${EVAL_ROOT}/generate_pedestrian_configs.py" \
    --eval-config "${EVAL_CONFIG}" \
    --output-dir "${PED_CONFIG_DIR}"

echo ""
echo "Starting sequential eval — 7 scenarios × active methods in eval_config.yaml"
echo ""

# ─── Main scenario loop ──────────────────────────────────────────────────────
for scenario in "${SCENARIOS[@]}"; do
    echo "=========================================="
    echo "SCENARIO: ${scenario}"
    echo "=========================================="

    PED_CFG="${PED_CONFIG_DIR}/${scenario}.yaml"
    MANIFEST="${OUTPUT_DIR}/manifest_${scenario}.jsonl"

    # ---- 1. Launch pedestrian_manager (skip for static-only) ----
    PED_PID=""
    if [ "${scenario}" != "S1_static_only" ]; then
        echo "[runbook] Launching pedestrian_manager with ${PED_CFG}"
        ros2 run pedestrian_manager pedestrian_manager_node \
            --ros-args -p config_file:="${PED_CFG}" \
            > "${OUTPUT_DIR}/ped_manager_${scenario}.log" 2>&1 &
        PED_PID=$!
        echo "[runbook] pedestrian_manager PID=${PED_PID}, waiting 3s for init..."
        sleep 3
    else
        echo "[runbook] Static-only — no pedestrian_manager"
    fi

    # ---- 2. Generate manifest (all active methods × this scenario) ----
    python3 "${EVAL_ROOT}/expand_scenarios.py" \
        --config "${EVAL_CONFIG}" \
        --filter-scenario "${scenario}" \
        --output "${MANIFEST}"

    # ---- 3. Run eval (skips already-done via episode_id → resumable) ----
    echo "[runbook] Running eval — appends to ${RESULTS}"
    python3 "${EVAL_ROOT}/run_eval.py" \
        --eval-config "${EVAL_CONFIG}" \
        --train-config "${TRAIN_CONFIG}" \
        --manifest "${MANIFEST}" \
        --output "${RESULTS}"

    # ---- 4. Kill pedestrian_manager ----
    if [ -n "${PED_PID}" ]; then
        echo "[runbook] Stopping pedestrian_manager (PID=${PED_PID})"
        kill "${PED_PID}" 2>/dev/null || true
        wait "${PED_PID}" 2>/dev/null || true
        sleep 1
    fi

    # ---- 5. Progress ----
    N_DONE=$(wc -l < "${RESULTS}" 2>/dev/null || echo 0)
    echo "[runbook] ${scenario} complete. results.jsonl: ${N_DONE} lines"
    echo ""
done

# ─── Post-eval analysis ──────────────────────────────────────────────────────
echo "=========================================="
echo "ANALYSIS — generating tables"
echo "=========================================="
python3 "${EVAL_ROOT}/analysis/build_tables.py" \
    --results "${RESULTS}" \
    --output-dir "${OUTPUT_DIR}/tables"

echo ""
echo "=========================================="
echo "DONE"
echo "=========================================="
echo "  Results:   ${RESULTS}"
echo "  Tables:    ${OUTPUT_DIR}/tables/"
echo "  Snapshots: ${OUTPUT_DIR}/snapshots/"
echo ""
echo "Phase 2 workflow: uncomment additional methods in ${EVAL_CONFIG},"
echo "then rerun this script. results.jsonl will be appended and new"
echo "comparison tables will auto-generate."

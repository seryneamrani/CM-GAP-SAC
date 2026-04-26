#!/usr/bin/env bash
# =============================================================================
# run_evaluation.sh — Lance une evaluation complete pour une (archi, monde)
# =============================================================================
#
# Usage:
#   bash run_evaluation.sh <architecture> <world> [repetitions]
#
# Exemples:
#   bash run_evaluation.sh A1 hospital
#   bash run_evaluation.sh A1 warehouse 3
#   bash run_evaluation.sh A2 dynamic 5
#
# Apres le run, les resultats sont dans /tmp/limo_eval_results/<arch>_<world>/
# =============================================================================

set -e

ARCH="${1:-A1}"
WORLD="${2:-hospital}"
REPS="${3:-3}"

if [[ ! "$ARCH" =~ ^A[1-4]$ ]]; then
    echo "ERREUR : architecture doit etre A1, A2, A3, ou A4 (recu: $ARCH)"
    exit 1
fi

if [[ ! "$WORLD" =~ ^(hospital|warehouse|dynamic|limo)$ ]]; then
    echo "ERREUR : monde doit etre hospital, warehouse, dynamic, ou limo (recu: $WORLD)"
    exit 1
fi

echo "================================================"
echo "  EVALUATION : $ARCH sur $WORLD ($REPS rep/paire)"
echo "  -> $((5 * REPS)) runs total"
echo "================================================"
echo ""

# Verifier que la carte existe
PKG_DIR="$HOME/limo_jazzy_ws/src/limo_description"
MAP_FILE="$PKG_DIR/config/maps/${WORLD}_map.yaml"
if [ ! -f "$MAP_FILE" ]; then
    echo "ERREUR : carte introuvable : $MAP_FILE"
    echo "  As-tu lance mapping.launch.py + save_map.sh $WORLD avant ?"
    exit 1
fi
echo "Carte detectee : $MAP_FILE"
echo ""

# Cleanup avant
echo "Cleanup des process residuels..."
pkill -f "gz sim|ros_gz|rviz2|robot_state|fix_scan|odom_tf|topic_tools|nav2|amcl|cartographer|evaluation_node|goal_runner|ros2 bag" 2>/dev/null || true
sleep 3

# Source ROS
source /opt/ros/jazzy/setup.bash
source ~/limo_jazzy_ws/install/setup.bash

# Lancement
echo ""
echo "Lancement evaluation.launch.py..."
echo ""

ros2 launch limo_evaluation evaluation.launch.py \
    world:="$WORLD" \
    architecture:="$ARCH" \
    repetitions:="$REPS"

# Cleanup apres
echo ""
echo "================================================"
echo "  Cleanup final..."
echo "================================================"
pkill -f "gz sim|ros_gz|rviz2|robot_state|fix_scan|odom_tf|topic_tools|nav2|amcl|cartographer|evaluation_node|goal_runner|ros2 bag" 2>/dev/null || true
sleep 2

echo ""
echo "================================================"
echo "  TERMINE"
echo "================================================"
echo "Resultats : /tmp/limo_eval_results/${ARCH}_${WORLD}/"
echo "  - metrics.csv"
echo "  - bag_${ARCH}_${WORLD}/"
echo ""
echo "Pour analyser le CSV :"
echo "  cat /tmp/limo_eval_results/${ARCH}_${WORLD}/metrics.csv"

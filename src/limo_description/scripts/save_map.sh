#!/usr/bin/env bash
# ============================================================
# save_map.sh — Sauvegarde la carte Cartographer dans config/maps/
# ============================================================
#
# Usage:
#   bash save_map.sh <nom_carte>
#
# Exemple:
#   bash save_map.sh hospital
#   => genere config/maps/hospital_map.pgm + hospital_map.yaml
#
# Prerequis :
#   - mapping.launch.py doit tourner (Cartographer actif, /map publie)
#   - le robot doit avoir explore tout l'environnement voulu
# ============================================================

set -e

if [ -z "$1" ]; then
    echo "Usage : bash save_map.sh <nom_carte>"
    echo "Exemple : bash save_map.sh hospital"
    exit 1
fi

NAME="$1"
PKG_DIR="$HOME/limo_jazzy_ws/src/limo_description"
MAPS_DIR="$PKG_DIR/config/maps"
OUT_PATH="$MAPS_DIR/${NAME}_map"

mkdir -p "$MAPS_DIR"

echo "=== Sourcing ROS 2 ==="
source /opt/ros/jazzy/setup.bash
source "$HOME/limo_jazzy_ws/install/setup.bash"

echo ""
echo "=== Verification : /map est-il publie ? ==="
if ! timeout 3 ros2 topic echo --once /map > /dev/null 2>&1; then
    echo "ERREUR : /map n'est pas publie. mapping.launch.py tourne-t-il ?"
    exit 1
fi
echo "  OK /map detecte"

echo ""
echo "=== 1. Finalisation de la trajectoire Cartographer ==="
ros2 service call /finish_trajectory cartographer_ros_msgs/srv/FinishTrajectory "{trajectory_id: 0}" \
    || echo "  (service finish_trajectory non disponible, on continue)"

echo ""
echo "=== 2. Appel de nav2_map_server save_map ==="
ros2 run nav2_map_server map_saver_cli \
    -f "$OUT_PATH" \
    --ros-args -p save_map_timeout:=10.0 -p free_thresh_default:=0.25 -p occupied_thresh_default:=0.65

echo ""
echo "=== 3. Verification ==="
if [ -f "${OUT_PATH}.pgm" ] && [ -f "${OUT_PATH}.yaml" ]; then
    echo "  OK Carte sauvegardee :"
    ls -la "${OUT_PATH}.pgm" "${OUT_PATH}.yaml"
    echo ""
    echo "  Pour naviguer avec cette carte :"
    echo "    ros2 launch limo_description navigation.launch.py world:=$NAME map:=$NAME"
else
    echo "  ERREUR : fichiers non crees. Verifie les logs ci-dessus."
    exit 1
fi

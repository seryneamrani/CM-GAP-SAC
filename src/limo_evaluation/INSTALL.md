# LIMO Evaluation Harness — Installation

## Etape 1 : Creer le package dans ton workspace

```bash
cd ~/limo_jazzy_ws/src
# Copie tous les fichiers de ce package ici, dans un dossier limo_evaluation/
```

## Etape 2 : Build

```bash
cd ~/limo_jazzy_ws
colcon build --packages-select limo_evaluation --symlink-install
source install/setup.bash
```

## Etape 3 : Workflow d'utilisation

### 3.1 Construire la carte (une fois par monde)

```bash
ros2 launch limo_description mapping.launch.py world:=hospital
# Teleop dans un autre terminal pour explorer
# Puis :
bash ~/limo_jazzy_ws/src/limo_description/scripts/save_map.sh hospital
```

### 3.2 Capturer les vraies coordonnees waypoints (une fois par monde)

```bash
# Lance navigation
ros2 launch limo_description navigation.launch.py world:=hospital map:=hospital
# Initialise AMCL via "2D Pose Estimate" dans RViz

# Dans un autre terminal, lance pose_picker
ros2 run limo_evaluation pose_picker

# Deplace le robot dans Gazebo (via teleop ou clic Gazebo)
# Pour chaque position voulue, appuie ENTREE dans le terminal pose_picker
# Note les coordonnees affichees

# Edite waypoints.yaml avec les vraies coordonnees
nano ~/limo_jazzy_ws/src/limo_evaluation/config/waypoints.yaml
# Rebuild apres edition (juste pour copier le YAML dans share/) :
cd ~/limo_jazzy_ws && colcon build --packages-select limo_evaluation --symlink-install
```

### 3.3 Lancer l'evaluation

```bash
# Architecture A1 sur hospital, 3 repetitions par paire = 15 runs
bash ~/limo_jazzy_ws/src/limo_evaluation/scripts/run_evaluation.sh A1 hospital 3
```

## Output

Apres le run, tu trouveras dans `/tmp/limo_eval_results/A1_hospital/` :
- `metrics.csv` — un row par run avec toutes les metriques
- `bag_A1_hospital/` — bag ROS 2 pour reanalyse

## Architecture des metriques

Voir `limo_evaluation/metrics.py` pour le detail. Categories :
- Robotique : success, distance, duration, collisions, replanif, d_min, jerk, SPL, TDI, TTC_min
- Perception (A2/A4) : latence, precision-recall (a venir)
- Apprentissage (A3/A4) : reward, convergence (logge separement durant entrainement SAC)

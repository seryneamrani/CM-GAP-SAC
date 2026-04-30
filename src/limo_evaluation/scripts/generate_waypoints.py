"""
generate_waypoints.py - Genere automatiquement des waypoints A1 depuis une carte PGM.

Strategie :
  1. Charge la carte (PGM + YAML)
  2. Erode les obstacles pour ne garder que les zones libres "safe"
     (a au moins 0.4m d'un obstacle)
  3. Detecte les composants connectes (zones navigables)
  4. Genere 5 paires start/goal :
     - pair_1 : trajet court (distance ~ 1-2m)
     - pair_2 : trajet moyen 90deg (distance ~ 2-3m)
     - pair_3 : trajet moyen droit (distance ~ 3-4m)
     - pair_4 : trajet long (distance ~ 4-6m)
     - pair_5 : trajet diagonal long
  5. Verifie qu'un chemin existe entre start et goal (BFS sur zone libre)
  6. Ecrit dans waypoints.yaml (SANS objets numpy)

Usage:
  python3 generate_waypoints.py <map_yaml_path> <world_name> <output_yaml>
                                 [--num-pairs 5] [--robot-radius 0.20] [--seed 42]
"""
import sys
import os
import math
import random
import argparse
from collections import deque

import numpy as np
import yaml
from PIL import Image


# Seuils Nav2 standards
FREE_THRESH = 200
OBSTACLE_THRESH = 100


# ─── YAML dumper sans numpy ────────────────────────────────────────────────────
class CleanDumper(yaml.Dumper):
    """Dumper YAML qui convertit tous les scalaires numpy en types Python natifs."""
    pass

def _represent_numpy_float(dumper, data):
    return dumper.represent_float(float(data))

def _represent_numpy_int(dumper, data):
    return dumper.represent_int(int(data))

for np_float_type in [np.float16, np.float32, np.float64]:
    CleanDumper.add_representer(np_float_type, _represent_numpy_float)

for np_int_type in [np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64]:
    CleanDumper.add_representer(np_int_type, _represent_numpy_int)
# ──────────────────────────────────────────────────────────────────────────────


def load_map(yaml_path):
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    pgm_path = os.path.join(os.path.dirname(yaml_path), meta['image'])
    img = np.array(Image.open(pgm_path))

    return {
        'image': img,
        'resolution': meta['resolution'],
        'origin_x': meta['origin'][0],
        'origin_y': meta['origin'][1],
    }


def world_to_pixel(x, y, map_data):
    res = map_data['resolution']
    ox = map_data['origin_x']
    oy = map_data['origin_y']
    h = map_data['image'].shape[0]
    col = int((x - ox) / res)
    row = h - 1 - int((y - oy) / res)
    return (col, row)


def pixel_to_world(col, row, map_data):
    res = map_data['resolution']
    ox = map_data['origin_x']
    oy = map_data['origin_y']
    h = map_data['image'].shape[0]
    x = col * res + ox
    y = (h - 1 - row) * res + oy
    return (float(x), float(y))


def erode_free_space(img, robot_radius_m, resolution):
    radius_px = int(math.ceil(robot_radius_m / resolution))
    free_mask = img >= FREE_THRESH
    from scipy.ndimage import binary_erosion
    y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
    kernel = (x*x + y*y <= radius_px*radius_px)
    safe_mask = binary_erosion(free_mask, structure=kernel)
    return safe_mask


def find_connected_component(safe_mask, seed):
    h, w = safe_mask.shape
    visited = np.zeros_like(safe_mask, dtype=bool)
    component = []
    seed_col, seed_row = seed
    if not (0 <= seed_row < h and 0 <= seed_col < w):
        return component
    if not safe_mask[seed_row, seed_col]:
        return component
    queue = deque([(seed_col, seed_row)])
    visited[seed_row, seed_col] = True
    while queue:
        col, row = queue.popleft()
        component.append((int(col), int(row)))  # cast to native int
        for dcol, drow in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = col + dcol, row + drow
            if 0 <= nr < h and 0 <= nc < w:
                if safe_mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nc, nr))
    return component


def find_largest_navigable_component(safe_mask):
    h, w = safe_mask.shape
    visited = np.zeros_like(safe_mask, dtype=bool)
    largest = []
    for row in range(h):
        for col in range(w):
            if safe_mask[row, col] and not visited[row, col]:
                component = find_connected_component(safe_mask, (col, row))
                for c, r in component:
                    visited[r, c] = True
                if len(component) > len(largest):
                    largest = component
    return largest


def is_path_clear(start_px, end_px, safe_mask, n_samples=50):
    h, w = safe_mask.shape
    sx, sy = start_px
    ex, ey = end_px
    for i in range(n_samples + 1):
        t = i / n_samples
        x = int(sx + t * (ex - sx))
        y = int(sy + t * (ey - sy))
        if 0 <= y < h and 0 <= x < w:
            if not safe_mask[y, x]:
                return False
    return True


def generate_waypoint_pairs(map_data, num_pairs=5, robot_radius=0.20, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    img = map_data['image']
    res = map_data['resolution']

    print(f"Carte : {img.shape[1]}x{img.shape[0]} px, res={res}m/px")
    print(f"Erosion avec robot_radius={robot_radius}m...")
    safe_mask = erode_free_space(img, robot_radius, res)

    n_safe = safe_mask.sum()
    print(f"Pixels navigables : {n_safe} ({100*n_safe/img.size:.1f}%)")

    if n_safe < 100:
        print("ERREUR : zone navigable trop petite.")
        sys.exit(1)

    print("Recherche de la plus grande zone connexe...")
    largest = find_largest_navigable_component(safe_mask)
    print(f"Plus grande zone : {len(largest)} pixels")

    if len(largest) < 100:
        print("ERREUR : aucune grande zone connexe trouvee")
        sys.exit(1)

    largest_arr = np.array(largest)

    distance_targets = [
        (1.0, 2.0, "Trajet court"),
        (2.0, 3.5, "Virage moyen"),
        (3.0, 4.5, "Trajet droit moyen"),
        (4.0, 6.0, "Trajet long"),
        (3.0, 5.5, "Diagonal long"),
    ]
    distance_targets = distance_targets[:num_pairs]

    pairs = []

    for pair_idx, (d_min, d_max, desc) in enumerate(distance_targets, 1):
        d_min_px = d_min / res
        d_max_px = d_max / res
        print(f"\nPair {pair_idx} : {desc} (distance {d_min}-{d_max}m)")

        found = False
        for attempt in range(500):
            i1, i2 = random.sample(range(len(largest_arr)), 2)
            p1 = (int(largest_arr[i1][0]), int(largest_arr[i1][1]))
            p2 = (int(largest_arr[i2][0]), int(largest_arr[i2][1]))

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            d_px = math.sqrt(dx*dx + dy*dy)

            if not (d_min_px <= d_px <= d_max_px):
                continue
            if not is_path_clear(p1, p2, safe_mask):
                continue

            # Convertir en float Python natif — PAS numpy
            start_x, start_y = pixel_to_world(p1[0], p1[1], map_data)
            goal_x, goal_y = pixel_to_world(p2[0], p2[1], map_data)

            yaw_start = float(math.atan2(goal_y - start_y, goal_x - start_x))
            d_world = float(math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2))

            print(f"  Trouve apres {attempt+1} essais : "
                  f"({start_x:.2f},{start_y:.2f}) -> ({goal_x:.2f},{goal_y:.2f}) "
                  f"dist={d_world:.2f}m yaw={math.degrees(yaw_start):.0f}deg")

            pairs.append({
                'pair_id': int(pair_idx),
                'description': str(desc),
                'start': {
                    'x': round(float(start_x), 3),
                    'y': round(float(start_y), 3),
                    'yaw': round(float(yaw_start), 4),
                },
                'goal': {
                    'x': round(float(goal_x), 3),
                    'y': round(float(goal_y), 3),
                    'yaw': round(float(yaw_start), 4),
                },
                'distance': round(float(d_world), 3),
            })
            found = True
            break

        if not found:
            print(f"  ATTENTION : aucune paire trouvee pour cette plage")

    return pairs


def update_waypoints_yaml(waypoints_path, world_name, pairs):
    if os.path.exists(waypoints_path):
        try:
            with open(waypoints_path) as f:
                data = yaml.safe_load(f) or {}
            # Si le fichier est corrompu (contient numpy), repart de zero
            if not isinstance(data, dict):
                data = {}
        except Exception:
            print(f"[WARN] Fichier existant corrompu, repart de zero.")
            data = {}
    else:
        data = {}

    if 'worlds' not in data:
        data['worlds'] = {}

    world_section = {}
    for p in pairs:
        pair_key = f"pair_{p['pair_id']}"
        world_section[pair_key] = {
            'description': p['description'],
            'start': p['start'],
            'goal': p['goal'],
        }

    data['worlds'][world_name] = world_section

    if 'run_config' not in data:
        data['run_config'] = {
            'timeout_seconds': 60.0,
            'goal_tolerance_xy': 0.25,
            'goal_tolerance_yaw': 0.25,
            'collision_distance_threshold': 0.05,
            'reset_pause_seconds': 2.0,
        }

    # Écrire avec CleanDumper — garantit zéro objet numpy
    with open(waypoints_path, 'w') as f:
        f.write("# Waypoints d'evaluation pour LIMO Pro\n")
        f.write("# 5 paires start/goal par monde, genere automatiquement\n\n")
        yaml.dump(data, f, Dumper=CleanDumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)

    print(f"\nFichier mis a jour : {waypoints_path}")
    print(f"Section : worlds.{world_name} ({len(pairs)} paires)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('map_yaml')
    parser.add_argument('world_name')
    parser.add_argument('output_yaml')
    parser.add_argument('--num-pairs', type=int, default=5)
    parser.add_argument('--robot-radius', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.map_yaml):
        print(f"ERREUR : carte introuvable : {args.map_yaml}")
        sys.exit(1)

    print(f"Carte source : {args.map_yaml}")
    print(f"Monde        : {args.world_name}")
    print(f"Output       : {args.output_yaml}")
    print(f"Robot radius : {args.robot_radius}m")
    print(f"Seed random  : {args.seed}")
    print()

    map_data = load_map(args.map_yaml)
    pairs = generate_waypoint_pairs(
        map_data,
        num_pairs=args.num_pairs,
        robot_radius=args.robot_radius,
        seed=args.seed,
    )

    if not pairs:
        print("\nERREUR : aucune paire generee")
        sys.exit(1)

    update_waypoints_yaml(args.output_yaml, args.world_name, pairs)

    print("\n=== Recap ===")
    for p in pairs:
        print(f"  pair_{p['pair_id']} ({p['description']}) : "
              f"{p['start']['x']:>6.2f},{p['start']['y']:>6.2f} -> "
              f"{p['goal']['x']:>6.2f},{p['goal']['y']:>6.2f}  "
              f"distance={p['distance']:.2f}m")


if __name__ == '__main__':
    main()

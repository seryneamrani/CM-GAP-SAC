#!/usr/bin/env python3
"""
auto_scan_map_v3.py
-------------------
Carte d'occupation 2D par téléportation + scans LiDAR.

Corrections v2 → v3 :
  [CRITIQUE] Ray tracing Bresenham : chaque rayon libère toutes les cellules
             traversées avant de marquer l'endpoint comme occupé.
  [CRITIQUE] Stockage (robot_pos, endpoint) par rayon — le découplage scan/pose
             de v2 rendait le ray tracing impossible en post-traitement.
  [CRITIQUE] Grille log-odds probabiliste — plus de mise à jour binaire.
             Un seul faux positif ne peut plus verrouiller une cellule.
  [QUALITÉ]  Paramètres corrigés : grid_step 0.5-0.8m recommandé, pas 1.5m.
  [QUALITÉ]  Post-traitement : fermeture morphologique + lissage gaussien optionnel.
  [CONSERVÉ] Vérification pose Gazebo, transformation LiDAR→monde, structure CLI.

Usage :
  python3 auto_scan_map_v3.py hospital /tmp/map_hospital \
      --bounds -8 8 -8 8 --grid-step 0.6 --resolution 0.05
"""

import os
import sys
import math
import time
import argparse
import subprocess
import json

import numpy as np
from PIL import Image
import cv2  # pour post-traitement morphologique

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


# ─────────────────────────────────────────────────────────────
# PARAMÈTRES LOG-ODDS
# ─────────────────────────────────────────────────────────────

# Probabilités du modèle de capteur (ajustables)
P_OCC_HIT   = 0.90
P_FREE_PASS = 0.45   # P(cellule occupée | rayon la traverse)

# Conversion en log-odds
def _p2l(p):
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1.0 - p))

L_OCC  =  _p2l(P_OCC_HIT)    # ≈ +1.10
L_FREE =  _p2l(P_FREE_PASS)  # ≈ -0.62  (négatif car P < 0.5)

# Saturation : évite sur-confiance après beaucoup de mises à jour
L_MIN = -3.0
L_MAX =  3.0

# Seuil export : log-odds > L_THRESH → occupé dans PGM
L_THRESH_OCC  =  0.5   # P ≈ 0.62
L_THRESH_FREE = -0.5   # P ≈ 0.38


# ─────────────────────────────────────────────────────────────
# BRESENHAM RAY TRACING
# ─────────────────────────────────────────────────────────────

def bresenham_ray(x0: int, y0: int, x1: int, y1: int):
    """
    Génère toutes les cellules traversées de (x0,y0) vers (x1,y1).
    Retourne (free_cells, hit_cell) :
      - free_cells : liste de (col,row) libres (trajet du rayon, endpoint EXCLU)
      - hit_cell   : (col,row) de l'obstacle, ou None si max_range atteint
    """
    free = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0

    while True:
        if x == x1 and y == y1:
            break
        free.append((x, y))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x   += sx
        if e2 < dx:
            err += dx
            y   += sy

    return free, (x1, y1)


# ─────────────────────────────────────────────────────────────
# GRILLE LOG-ODDS
# ─────────────────────────────────────────────────────────────

class LogOddsGrid:
    """
    Grille 2D à mise à jour log-odds.
    Axe col = X monde, axe row = Y monde (row 0 = Y_min, row H-1 = Y_max).
    On inverse au moment de l'export PGM (convention ROS : row 0 = Y_min).
    """

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        self.res    = resolution
        self.x_min  = x_min
        self.y_min  = y_min
        self.W      = int(math.ceil((x_max - x_min) / resolution))
        self.H      = int(math.ceil((y_max - y_min) / resolution))
        # Grille log-odds, initialisée à 0.0 → probabilité 0.5 (inconnu)
        self.grid   = np.zeros((self.H, self.W), dtype=np.float32)
        print(f"[Grid] {self.W}×{self.H} cellules | res={resolution}m | "
              f"X[{x_min:.1f},{x_max:.1f}] Y[{y_min:.1f},{y_max:.1f}]")

    def world_to_cell(self, wx, wy):
        """Coordonnées monde → indices (col, row). Retourne None si hors grille."""
        col = int((wx - self.x_min) / self.res)
        row = int((wy - self.y_min) / self.res)
        if 0 <= col < self.W and 0 <= row < self.H:
            return col, row
        return None

    def update_ray(self, robot_wx, robot_wy,
                   endpoint_wx, endpoint_wy,
                   is_hit: bool):
        """
        Met à jour la grille pour un rayon unique.
        is_hit=True  → endpoint est un obstacle (log-odds += L_OCC)
        is_hit=False → rayon a atteint max_range, pas d'obstacle détecté
                       (on libère le trajet mais on ne marque pas l'endpoint)
        """
        src = self.world_to_cell(robot_wx, robot_wy)
        dst = self.world_to_cell(endpoint_wx, endpoint_wy)

        if src is None:
            return  # robot hors grille → ignorer

        # Si endpoint hors grille, on fait quand même le trajet jusqu'au bord
        if dst is None:
            # Clipper endpoint sur le bord de la grille
            dst = self._clip_to_border(src, endpoint_wx, endpoint_wy)
            if dst is None:
                return
            is_hit = False  # pas d'obstacle sur le bord

        free_cells, hit_cell = bresenham_ray(src[0], src[1], dst[0], dst[1])

        # Libérer les cellules traversées
        for col, row in free_cells:
            self.grid[row, col] = max(L_MIN,
                                      self.grid[row, col] + L_FREE)

        # Marquer l'endpoint (obstacle uniquement si is_hit)
        col, row = hit_cell
        if 0 <= col < self.W and 0 <= row < self.H:
            if is_hit:
                self.grid[row, col] = min(L_MAX,
                                          self.grid[row, col] + L_OCC)
            else:
                self.grid[row, col] = max(L_MIN,
                                          self.grid[row, col] + L_FREE)

    def _clip_to_border(self, src_cell, wx, wy):
        """Clippe un point monde hors-grille sur le bord de la grille (ligne de vue)."""
        col = int((wx - self.x_min) / self.res)
        row = int((wy - self.y_min) / self.res)
        col = max(0, min(self.W - 1, col))
        row = max(0, min(self.H - 1, row))
        return col, row

    def to_pgm_array(self):
        """
        Convertit la grille log-odds en image PGM (uint8) :
          0   = occupé  (noir)
          254 = libre   (blanc)
          205 = inconnu (gris)
        Convention ROS : row 0 de l'image = Y_min du monde.
        """
        img = np.full((self.H, self.W), 205, dtype=np.uint8)

        occ_mask  = self.grid >  L_THRESH_OCC
        free_mask = self.grid < L_THRESH_FREE

        img[free_mask] = 254
        img[occ_mask]  = 0

        # Flip vertical : ROS attend Y croissant vers le haut,
        # PIL/numpy a row 0 en haut → on flippe pour export PGM
        return np.flipud(img)

    def apply_postprocessing(self, gaussian_sigma=0.8, morph_close_px=2):
        """
        Post-traitement sur la grille log-odds avant export :
          1. Fermeture morphologique : ferme les petits trous dans les murs
          2. Lissage gaussien léger : réduit le bruit de mesure
        Opère directement sur self.grid.
        """
        kernel = np.ones((morph_close_px * 2 + 1,) * 2, np.float32)

        # Masque occupé → dilatation puis érosion (fermeture)
        occ_binary = (self.grid > L_THRESH_OCC).astype(np.uint8)
        closed = cv2.morphologyEx(occ_binary, cv2.MORPH_CLOSE, kernel)
        # Ajouter les cellules fermées comme occupées faibles
        new_occ = (closed == 1) & (self.grid <= L_THRESH_OCC)
        self.grid[new_occ] = L_THRESH_OCC + 0.1

        # Lissage gaussien sur la grille log-odds complète
        if gaussian_sigma > 0:
            self.grid = cv2.GaussianBlur(
                self.grid,
                ksize=(0, 0),
                sigmaX=gaussian_sigma,
                sigmaY=gaussian_sigma
            )
        print("[Grid] Post-traitement appliqué (fermeture + gaussien).")


# ─────────────────────────────────────────────────────────────
# NODE ROS2
# ─────────────────────────────────────────────────────────────

class AutoScanMapperV3(Node):
    def __init__(self, world_name='hospital', model_name='limo'):
        super().__init__('auto_scan_mapper_v3')
        self.world_name = world_name
        self.model_name = model_name
        self.set_pose_service = f'/world/{world_name}/set_pose'

        self.last_scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        # CORRECTION v3 : on stocke des RAYONS (robot_pos, endpoint, is_hit)
        # et non des points isolés → permet le ray tracing dans build_map
        self.rays = []   # liste de (robot_x, robot_y, ep_x, ep_y, is_hit)

        self.get_logger().info(f'V3 prêt. Service: {self.set_pose_service}')

    # ── Callback scan ─────────────────────────────────────────

    def _scan_cb(self, msg):
        self.last_scan = msg

    # ── Téléportation ────────────────────────────────────────

    def teleport(self, x, y, yaw=0.0):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        req = (
            f'name: "{self.model_name}", '
            f'position: {{x: {x:.4f}, y: {y:.4f}, z: 0.2}}, '
            f'orientation: {{x: 0, y: 0, z: {qz:.6f}, w: {qw:.6f}}}'
        )
        cmd = [
            'gz', 'service',
            '-s', self.set_pose_service,
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1500',
            '--req', req,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=3, text=True)
            return 'data: true' in result.stdout.lower()
        except Exception:
            return False

    # ── Pose vérité Gazebo ────────────────────────────────────

    def get_robot_pose_from_gz(self):
        try:
            cmd = [
                'gz', 'topic',
                '-e', '-t', f'/world/{self.world_name}/pose/info',
                '-n', '1', '--json-output',
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=3, text=True)
            data = json.loads(result.stdout)
            for p in data.get('pose', []):
                if p.get('name') == self.model_name:
                    pos = p.get('position', {})
                    ori = p.get('orientation', {})
                    # Extraire yaw depuis quaternion
                    qz = ori.get('z', 0.0)
                    qw = ori.get('w', 1.0)
                    yaw = 2.0 * math.atan2(qz, qw)
                    return (pos.get('x', 0.0), pos.get('y', 0.0), yaw)
        except Exception:
            pass
        return None

    def verify_teleport(self, target_x, target_y, tolerance=0.3, max_tries=8):
        for _ in range(max_tries):
            pose = self.get_robot_pose_from_gz()
            if pose is not None:
                d = math.sqrt((pose[0]-target_x)**2 + (pose[1]-target_y)**2)
                if d < tolerance:
                    return True, d
            time.sleep(0.15)
        return False, math.inf

    # ── Capture scan → RAYONS ─────────────────────────────────

    def capture_rays_at(self, robot_x, robot_y, robot_yaw,
                        range_min_override=0.15,
                        range_max_override=8.0,
                        n_scans=2):
        """
        CORRECTION PRINCIPALE :
        Au lieu de retourner une liste de points (wx, wy),
        on retourne une liste de RAYONS :
          (robot_x, robot_y, endpoint_wx, endpoint_wy, is_hit)

        is_hit=False si le rayon a atteint range_max (pas de mur → ne pas marquer occupé)
        is_hit=True  si le rayon a terminé sur un obstacle réel

        On moyenne N scans pour réduire le bruit de mesure.
        """
        collected_rays = []

        for _ in range(n_scans):
            # Drainer anciens messages
            self.last_scan = None
            deadline = time.time() + 2.0
            while time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.last_scan is not None:
                    break

            if self.last_scan is None:
                continue

            scan = self.last_scan
            cos_y = math.cos(robot_yaw)
            sin_y = math.sin(robot_yaw)

            for i, r in enumerate(scan.ranges):
                # Ignorer NaN, Inf, out-of-range
                if not math.isfinite(r):
                    continue

                # Rayon max_range : libre mais pas d'obstacle
                if r >= min(range_max_override, scan.range_max * 0.99):
                    # Projeter au max_range, marquer libre seulement
                    angle  = scan.angle_min + i * scan.angle_increment
                    r_clip = min(range_max_override, scan.range_max)
                    lx = r_clip * math.cos(angle)
                    ly = r_clip * math.sin(angle)
                    wx = robot_x + lx * cos_y - ly * sin_y
                    wy = robot_y + lx * sin_y + ly * cos_y
                    collected_rays.append((robot_x, robot_y, wx, wy, False))
                    continue

                # Rayon trop court (réflexion, corps du robot)
                if r < range_min_override or r < scan.range_min:
                    continue

                angle  = scan.angle_min + i * scan.angle_increment
                lx = r * math.cos(angle)
                ly = r * math.sin(angle)

                # Transformation LiDAR → monde (rotation 2D + translation)
                wx = robot_x + lx * cos_y - ly * sin_y
                wy = robot_y + lx * sin_y + ly * cos_y

                collected_rays.append((robot_x, robot_y, wx, wy, True))

        return collected_rays

    # ── Boucle de scan sur grille ─────────────────────────────

    def scan_grid(self, x_min, x_max, y_min, y_max, grid_step,
                  yaws=None, scan_settle_time=0.4, n_scans_per_pos=2):

        if yaws is None:
            yaws = [0.0, math.pi/2, math.pi, -math.pi/2]

        xs = np.arange(x_min, x_max + grid_step / 2, grid_step)
        ys = np.arange(y_min, y_max + grid_step / 2, grid_step)
        total = len(xs) * len(ys) * len(yaws)
        self.get_logger().info(
            f'Grille : {len(xs)}×{len(ys)} × {len(yaws)} yaws = {total} scans'
        )

        n = n_ok = n_fail_tp = n_fail_vf = 0

        for x in xs:
            for y in ys:
                # On téléporte une fois par position (pas par yaw)
                tp_ok = self.teleport(float(x), float(y), 0.0)
                if not tp_ok:
                    n_fail_tp += len(yaws)
                    n += len(yaws)
                    continue

                time.sleep(0.15)  # laisser Gazebo appliquer la pose

                ok, d = self.verify_teleport(float(x), float(y), tolerance=0.3)
                if not ok:
                    # Position non atteinte = probablement à l'intérieur d'un obstacle
                    n_fail_vf += len(yaws)
                    n += len(yaws)
                    self.get_logger().debug(
                        f'  Skip ({x:.1f},{y:.1f}) : hors portée après téléport'
                    )
                    continue

                for yaw in yaws:
                    n += 1
                    # Ré-orienter uniquement (pas de re-téléportation XY)
                    self.teleport(float(x), float(y), float(yaw))
                    time.sleep(scan_settle_time)

                    # CORRECTION : lire la pose RÉELLE après chaque orientation
                    # (le yaw de Gazebo peut différer légèrement de la commande)
                    actual_pose = self.get_robot_pose_from_gz()
                    if actual_pose is None:
                        actual_pose = (float(x), float(y), float(yaw))
                    rx, ry, ryaw = actual_pose

                    # Capturer les rayons avec la pose réelle
                    rays = self.capture_rays_at(rx, ry, ryaw,
                                                n_scans=n_scans_per_pos)
                    self.rays.extend(rays)
                    n_ok += 1

                if n % 20 == 0:
                    n_hits  = sum(1 for r in self.rays if r[4])
                    n_free  = sum(1 for r in self.rays if not r[4])
                    self.get_logger().info(
                        f'  {n}/{total} ({100*n/total:.0f}%) — '
                        f'{n_hits} rayons hit / {n_free} free — '
                        f'échecs: {n_fail_tp} tp, {n_fail_vf} vf'
                    )

        n_hits = sum(1 for r in self.rays if r[4])
        self.get_logger().info(
            f'\nScan terminé : {len(self.rays)} rayons total '
            f'({n_hits} hits / {len(self.rays)-n_hits} free)'
        )


# ─────────────────────────────────────────────────────────────
# CONSTRUCTION DE CARTE
# ─────────────────────────────────────────────────────────────

def build_map(rays, resolution=0.05, padding=1.5,
              postprocess=True, gaussian_sigma=0.8):
    """
    Construit la grille log-odds à partir des rayons.
    Chaque rayon (robot_pos → endpoint) est tracé avec Bresenham :
      - Cellules traversées : mise à jour libre (log-odds négatif)
      - Endpoint is_hit=True : mise à jour occupé (log-odds positif)
    """
    if not rays:
        print("ERREUR: aucun rayon collecté.")
        sys.exit(1)

    # Emprise automatique depuis tous les points
    all_x = [r[0] for r in rays] + [r[2] for r in rays]
    all_y = [r[1] for r in rays] + [r[3] for r in rays]
    x_min = min(all_x) - padding
    x_max = max(all_x) + padding
    y_min = min(all_y) - padding
    y_max = max(all_y) + padding

    grid = LogOddsGrid(x_min, x_max, y_min, y_max, resolution)

    total = len(rays)
    print(f"Ray tracing : {total} rayons...")
    for i, (rx, ry, ex, ey, is_hit) in enumerate(rays):
        grid.update_ray(rx, ry, ex, ey, is_hit)
        if (i + 1) % 50000 == 0:
            print(f"  {i+1}/{total} ({100*(i+1)/total:.1f}%)")

    if postprocess:
        grid.apply_postprocessing(
            gaussian_sigma=gaussian_sigma,
            morph_close_px=2
        )

    img = grid.to_pgm_array()

    n_occ  = (img == 0).sum()
    n_free = (img == 254).sum()
    n_unk  = (img == 205).sum()
    total_px = img.size
    print(f"\nCarte finale :")
    print(f"  Occupé  : {n_occ:6d} px ({100*n_occ/total_px:.1f}%)")
    print(f"  Libre   : {n_free:6d} px ({100*n_free/total_px:.1f}%)")
    print(f"  Inconnu : {n_unk:6d} px ({100*n_unk/total_px:.1f}%)")

    return img, x_min, y_min


# ─────────────────────────────────────────────────────────────
# EXPORT PGM + YAML
# ─────────────────────────────────────────────────────────────

def save_map(img, origin_x, origin_y, resolution, output_path):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    pgm_path  = f"{output_path}.pgm"
    yaml_path = f"{output_path}.yaml"

    Image.fromarray(img, mode='L').save(pgm_path)

    yaml_content = (
        f"image: {os.path.basename(pgm_path)}\n"
        f"mode: trinary\n"
        f"resolution: {resolution}\n"
        f"origin: [{origin_x:.4f}, {origin_y:.4f}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nFichiers sauvegardés :")
    print(f"  {pgm_path}")
    print(f"  {yaml_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Carte d\'occupation 2D par téléportation + ray tracing log-odds'
    )
    parser.add_argument('world_name',   help='Nom du monde Gazebo (ex: hospital)')
    parser.add_argument('output_name',  help='Chemin de sortie sans extension (ex: /tmp/map)')
    parser.add_argument('--bounds', type=float, nargs=4,
                        default=[-7, 7, -7, 7],
                        metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'),
                        help='Emprise du scan en mètres (défaut: -7 7 -7 7)')
    parser.add_argument('--grid-step', type=float, default=0.6,
                        help='Pas entre positions de téléportation en m (défaut: 0.6)')
    parser.add_argument('--resolution', type=float, default=0.05,
                        help='Résolution de la carte en m/px (défaut: 0.05)')
    parser.add_argument('--n-yaws', type=int, default=4,
                        help='Nombre d\'orientations par position (défaut: 4)')
    parser.add_argument('--n-scans', type=int, default=2,
                        help='Scans moyennés par orientation (défaut: 2)')
    parser.add_argument('--scan-settle', type=float, default=0.4,
                        help='Pause stabilisation après téléport en s (défaut: 0.4)')
    parser.add_argument('--padding', type=float, default=1.5,
                        help='Marge autour des points détectés en m (défaut: 1.5)')
    parser.add_argument('--no-postprocess', action='store_true',
                        help='Désactiver fermeture morphologique + lissage')
    parser.add_argument('--model', default='limo',
                        help='Nom du modèle Gazebo (défaut: limo)')
    args = parser.parse_args()

    # ── Initialisation ROS ─────────────────────────────────────
    rclpy.init()
    mapper = AutoScanMapperV3(world_name=args.world_name, model_name=args.model)

    print(f"\n{'='*50}")
    print(f"  auto_scan_map_v3 — ROS2 Jazzy / Gazebo Harmonic")
    print(f"{'='*50}")
    print(f"  Monde       : {args.world_name}")
    print(f"  Modèle      : {args.model}")
    print(f"  Bornes XY   : x[{args.bounds[0]}, {args.bounds[1]}]"
          f"  y[{args.bounds[2]}, {args.bounds[3]}]")
    print(f"  Grid step   : {args.grid_step} m")
    print(f"  Orientations: {args.n_yaws} yaws / position")
    print(f"  Scans/pose  : {args.n_scans}")
    print(f"  Résolution  : {args.resolution} m/px")
    print(f"  Settle time : {args.scan_settle} s")
    print()

    # ── Vérification /scan ─────────────────────────────────────
    print("Vérification /scan...")
    for _ in range(30):
        rclpy.spin_once(mapper, timeout_sec=0.1)
        if mapper.last_scan is not None:
            s = mapper.last_scan
            print(f"  OK : {len(s.ranges)} rayons, "
                  f"range [{s.range_min:.2f}, {s.range_max:.2f}] m, "
                  f"angle [{math.degrees(s.angle_min):.1f}°, "
                  f"{math.degrees(s.angle_max):.1f}°]")
            break
        time.sleep(0.1)
    else:
        print("ERREUR : /scan absent. Vérifier bridge ROS-Gazebo.")
        sys.exit(1)

    # ── Vérification pose Gazebo ───────────────────────────────
    print("Vérification pose Gazebo...")
    pose = mapper.get_robot_pose_from_gz()
    if pose is None:
        print(f"ERREUR : impossible de lire /world/{args.world_name}/pose/info")
        print(f"  → Vérifier : gz topic -e -t /world/{args.world_name}/pose/info -n 1")
        sys.exit(1)
    print(f"  OK : robot à ({pose[0]:.2f}, {pose[1]:.2f}), yaw={math.degrees(pose[2]):.1f}°")

    # ── Scan sur la grille ─────────────────────────────────────
    yaws = [i * 2 * math.pi / args.n_yaws for i in range(args.n_yaws)]
    mapper.scan_grid(
        x_min=args.bounds[0], x_max=args.bounds[1],
        y_min=args.bounds[2], y_max=args.bounds[3],
        grid_step=args.grid_step,
        yaws=yaws,
        scan_settle_time=args.scan_settle,
        n_scans_per_pos=args.n_scans,
    )

    # ── Construction carte ─────────────────────────────────────
    img, ox, oy = build_map(
        mapper.rays,
        resolution=args.resolution,
        padding=args.padding,
        postprocess=not args.no_postprocess,
        gaussian_sigma=0.8,
    )
    save_map(img, ox, oy, args.resolution, args.output_name)

    mapper.destroy_node()
    rclpy.shutdown()
    print("\nDone.")


if __name__ == '__main__':
    main()

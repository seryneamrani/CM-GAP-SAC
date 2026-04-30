"""
auto_scan_map.py - Genere une carte d'occupation 2D en teleportant un robot
sur une grille et capturant les scans LiDAR.

Pre-requis:
  - Gazebo + bridge ROS<->Gazebo doivent tourner
  - Le robot 'limo' doit etre spawn dans le monde
  - /scan doit publier
  - Le service /world/<world>/set_pose doit fonctionner

Usage:
  python3 auto_scan_map.py <world_name> <output_name>
      [--bounds X_MIN X_MAX Y_MIN Y_MAX]
      [--grid-step 1.0]
      [--scans-per-pos 20]
      [--resolution 0.05]
      [--robot-radius 0.20]

Exemple:
  python3 auto_scan_map.py hospital hospital_map \\
      --bounds -10 10 -10 10 --grid-step 1.5
"""
import os
import sys
import math
import time
import argparse
import subprocess

import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class AutoScanMapper(Node):
    def __init__(self, world_name='hospital'):
        super().__init__('auto_scan_mapper')
        self.world_name = world_name
        self.set_pose_service = f'/world/{world_name}/set_pose'

        self.last_scan = None
        self.last_odom_pos = None

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self.scan_points = []  # liste de tuples (x, y) dans le repere monde
        self.robot_positions = []  # positions du robot (libres)

        self.get_logger().info(f'AutoScanMapper pret. Service: {self.set_pose_service}')

    def _scan_cb(self, msg):
        self.last_scan = msg

    def _odom_cb(self, msg):
        self.last_odom_pos = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

    def teleport(self, x, y, yaw=0.0):
        """Teleporte le robot et attend confirmation via /odom."""
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        req = (
            f'name: "limo", '
            f'position: {{x: {x}, y: {y}, z: 0.2}}, '
            f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}'
        )
        cmd = [
            'gz', 'service',
            '-s', self.set_pose_service,
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', req,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=4, text=True)
            ok = 'data: true' in result.stdout.lower()
        except Exception:
            ok = False
        return ok

    def wait_odom_at(self, x, y, tolerance=0.4, timeout=2.5):
        """Attend que /odom reflete la position cible."""
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.last_odom_pos is not None:
                dx = self.last_odom_pos[0] - x
                dy = self.last_odom_pos[1] - y
                if math.sqrt(dx*dx + dy*dy) < tolerance:
                    return True
            time.sleep(0.1)
        return False

    def capture_scan_at(self, robot_x, robot_y, robot_yaw):
        """Capture un scan LiDAR et convertit en points 2D dans le repere monde."""
        # Drainer les anciens scans
        self.last_scan = None
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.last_scan is not None:
                break
            time.sleep(0.05)

        if self.last_scan is None:
            return []

        scan = self.last_scan
        points = []
        n = len(scan.ranges)
        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            if r < scan.range_min or r > scan.range_max:
                continue

            # Angle dans le repere du laser
            angle = scan.angle_min + i * scan.angle_increment

            # Point dans le repere du laser (X devant, Y a gauche)
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)

            # Transformer dans le repere monde
            cos_y = math.cos(robot_yaw)
            sin_y = math.sin(robot_yaw)
            wx = robot_x + lx * cos_y - ly * sin_y
            wy = robot_y + lx * sin_y + ly * cos_y

            points.append((wx, wy))

        return points

    def scan_grid(self, x_min, x_max, y_min, y_max, grid_step,
                    scans_per_pos=4, yaws=None):
        """Telegrapie le robot sur une grille et collecte les scans."""
        if yaws is None:
            yaws = [0.0, math.pi/2, math.pi, -math.pi/2]

        x_positions = np.arange(x_min, x_max + grid_step/2, grid_step)
        y_positions = np.arange(y_min, y_max + grid_step/2, grid_step)

        total_positions = len(x_positions) * len(y_positions) * len(yaws)
        self.get_logger().info(
            f'Grille : {len(x_positions)}x{len(y_positions)} positions '
            f'x {len(yaws)} orientations = {total_positions} scans'
        )

        n_processed = 0
        n_failed = 0

        for x in x_positions:
            for y in y_positions:
                for yaw in yaws:
                    n_processed += 1

                    # Teleport
                    if not self.teleport(float(x), float(y), float(yaw)):
                        n_failed += 1
                        continue

                    # Attendre que la position soit confirmee
                    if not self.wait_odom_at(float(x), float(y)):
                        # Le robot ne peut pas etre la (probablement dans un mur)
                        # On enregistre tout de meme la position pour info
                        continue

                    # Robot bien teleporte - cette position est libre
                    self.robot_positions.append((float(x), float(y)))

                    # Petite pause pour stabiliser le scan
                    time.sleep(0.2)

                    # Capturer le scan
                    points = self.capture_scan_at(float(x), float(y), float(yaw))
                    self.scan_points.extend(points)

                    if n_processed % 10 == 0:
                        self.get_logger().info(
                            f'  Progres : {n_processed}/{total_positions} '
                            f'({100*n_processed/total_positions:.0f}%) - '
                            f'{len(self.scan_points)} points scannes, '
                            f'{len(self.robot_positions)} positions libres'
                        )

        self.get_logger().info(
            f'\nScan termine: {len(self.scan_points)} points obstacles, '
            f'{len(self.robot_positions)} positions libres confirmees, '
            f'{n_failed} teleports echoues'
        )


def build_map(scan_points, robot_positions, resolution=0.05, padding=2.0):
    """
    Construit une carte d'occupation a partir de points scannes et de positions
    libres. Style Nav2 : 254=libre, 0=occupe, 205=inconnu.
    """
    if not scan_points:
        print("ERREUR: aucun point scanne")
        sys.exit(1)

    # Bounding box
    all_xs = [p[0] for p in scan_points] + [p[0] for p in robot_positions]
    all_ys = [p[1] for p in scan_points] + [p[1] for p in robot_positions]

    x_min = min(all_xs) - padding
    x_max = max(all_xs) + padding
    y_min = min(all_ys) - padding
    y_max = max(all_ys) + padding

    width_m = x_max - x_min
    height_m = y_max - y_min
    width_px = int(math.ceil(width_m / resolution))
    height_px = int(math.ceil(height_m / resolution))

    print(f"Carte : {width_m:.1f}m x {height_m:.1f}m ({width_px}x{height_px} px)")
    print(f"Origin : ({x_min:.2f}, {y_min:.2f})")

    # 205 = inconnu
    img = np.full((height_px, width_px), 205, dtype=np.uint8)

    def world_to_pixel(x, y):
        col = int((x - x_min) / resolution)
        row = height_px - 1 - int((y - y_min) / resolution)
        return col, row

    # Marquer les positions libres (cercle de rayon ~0.3m)
    free_radius_px = int(0.3 / resolution)
    for x, y in robot_positions:
        col, row = world_to_pixel(x, y)
        for dr in range(-free_radius_px, free_radius_px+1):
            for dc in range(-free_radius_px, free_radius_px+1):
                if dr*dr + dc*dc > free_radius_px*free_radius_px:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < height_px and 0 <= c < width_px:
                    img[r, c] = 254  # libre

    # Marquer les obstacles (epaissir un peu)
    for x, y in scan_points:
        col, row = world_to_pixel(x, y)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = row + dr, col + dc
                if 0 <= r < height_px and 0 <= c < width_px:
                    img[r, c] = 0  # obstacle

    print(f"Pixels libres   : {(img == 254).sum()} ({100*(img == 254).sum()/img.size:.1f}%)")
    print(f"Pixels obstacle : {(img == 0).sum()} ({100*(img == 0).sum()/img.size:.1f}%)")
    print(f"Pixels inconnu  : {(img == 205).sum()} ({100*(img == 205).sum()/img.size:.1f}%)")

    return img, x_min, y_min


def save_map(img, origin_x, origin_y, resolution, output_name):
    pgm_path = f"{output_name}.pgm"
    yaml_path = f"{output_name}.yaml"

    Image.fromarray(img, mode='L').save(pgm_path)
    print(f"PGM sauve : {pgm_path}")

    yaml_content = f"""image: {os.path.basename(pgm_path)}
mode: trinary
resolution: {resolution}
origin: [{origin_x:.3f}, {origin_y:.3f}, 0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"YAML sauve : {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='Auto-scan map generator')
    parser.add_argument('world_name', help='Nom du monde Gazebo (default, hospital, ...)')
    parser.add_argument('output_name', help='Nom de sortie (sans extension)')
    parser.add_argument('--bounds', type=float, nargs=4,
                        default=[-10, 10, -10, 10],
                        metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'),
                        help='Bornes XY du scan en metres (defaut -10 10 -10 10)')
    parser.add_argument('--grid-step', type=float, default=1.5,
                        help='Pas de la grille en metres (defaut 1.5)')
    parser.add_argument('--scans-per-pos', type=int, default=4,
                        help='Orientations par position (defaut 4)')
    parser.add_argument('--resolution', type=float, default=0.05,
                        help='Resolution PGM en m/px (defaut 0.05)')
    parser.add_argument('--padding', type=float, default=1.0,
                        help='Marge autour de la carte en m (defaut 1.0)')
    args = parser.parse_args()

    rclpy.init()
    mapper = AutoScanMapper(world_name=args.world_name)

    print(f"\n=== Configuration ===")
    print(f"Monde         : {args.world_name}")
    print(f"Bornes XY     : x[{args.bounds[0]}, {args.bounds[1]}] "
          f"y[{args.bounds[2]}, {args.bounds[3]}]")
    print(f"Grid step     : {args.grid_step}m")
    print(f"Orientations  : {args.scans_per_pos}")
    print(f"Resolution    : {args.resolution}m/px")
    print()

    # Verifier que /scan publie
    print("Verification : /scan publie ?")
    for _ in range(20):
        rclpy.spin_once(mapper, timeout_sec=0.1)
        if mapper.last_scan is not None:
            print(f"  OK : scan recu ({len(mapper.last_scan.ranges)} rays, "
                  f"range max={mapper.last_scan.range_max}m)")
            break
        time.sleep(0.1)
    else:
        print("ERREUR: /scan ne publie pas. Le bridge tourne-t-il ?")
        sys.exit(1)

    # Lancer le scan
    yaws = [i * 2 * math.pi / args.scans_per_pos for i in range(args.scans_per_pos)]
    mapper.scan_grid(
        x_min=args.bounds[0], x_max=args.bounds[1],
        y_min=args.bounds[2], y_max=args.bounds[3],
        grid_step=args.grid_step,
        yaws=yaws,
    )

    # Construire la carte
    img, origin_x, origin_y = build_map(
        mapper.scan_points, mapper.robot_positions,
        resolution=args.resolution, padding=args.padding,
    )

    save_map(img, origin_x, origin_y, args.resolution, args.output_name)

    mapper.destroy_node()
    rclpy.shutdown()
    print("\nDone.")


if __name__ == '__main__':
    main()

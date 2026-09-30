"""
auto_scan_map_v2.py - Genere une carte d'occupation 2D en teleportant un robot
sur une grille et capturant les scans LiDAR.

VERSION V2 :
  - Verifie la teleportation via /world/<world>/pose/info (pose verite Gazebo)
  - Plus de wait_odom (qui ne refletait jamais les teleports car odom = roues)
  - Beaucoup plus rapide

Pre-requis:
  - Gazebo + bridge ROS<->Gazebo doivent tourner
  - Le robot 'limo' doit etre spawn dans le monde
  - /scan doit publier
  - Le service /world/<world>/set_pose doit fonctionner

Usage:
  python3 auto_scan_map_v2.py <world_name> <output_name> [options]
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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class AutoScanMapperV2(Node):
    def __init__(self, world_name='hospital', model_name='limo'):
        super().__init__('auto_scan_mapper_v2')
        self.world_name = world_name
        self.model_name = model_name
        self.set_pose_service = f'/world/{world_name}/set_pose'

        self.last_scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.scan_points = []
        self.robot_positions = []

        self.get_logger().info(f'V2 pret. Service: {self.set_pose_service}')

    def _scan_cb(self, msg):
        self.last_scan = msg

    def teleport(self, x, y, yaw=0.0):
        """Teleporte le robot via gz service. Retourne True si data:true."""
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        req = (
            f'name: "{self.model_name}", '
            f'position: {{x: {x}, y: {y}, z: 0.2}}, '
            f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}'
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

    def get_robot_pose_from_gz(self):
        """
        Lit la position vraie du robot depuis /world/<world>/pose/info via gz topic.
        Retourne (x, y) ou None.
        """
        try:
            # Lance gz topic en mode echo pour 1 seul message JSON
            cmd = [
                'gz', 'topic',
                '-e', '-t', f'/world/{self.world_name}/pose/info',
                '-n', '1',
                '--json-output',
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=2, text=True)
            output = result.stdout

            # Le format JSON contient une liste de poses, on cherche celle de "limo"
            data = json.loads(output)
            poses = data.get('pose', [])
            for p in poses:
                if p.get('name') == self.model_name:
                    pos = p.get('position', {})
                    return (pos.get('x', 0), pos.get('y', 0))
        except Exception as e:
            return None
        return None

    def verify_teleport(self, target_x, target_y, tolerance=0.5, max_tries=5):
        """Verifie via la pose verite Gazebo que le robot est bien teleporte."""
        for _ in range(max_tries):
            pose = self.get_robot_pose_from_gz()
            if pose is not None:
                dx = pose[0] - target_x
                dy = pose[1] - target_y
                d = math.sqrt(dx*dx + dy*dy)
                if d < tolerance:
                    return True, d
            time.sleep(0.1)
        return False, math.inf

    def capture_scan_at(self, robot_x, robot_y, robot_yaw):
        """Capture un scan LiDAR et convertit en points 2D dans le repere monde."""
        # Drainer les anciens scans
        self.last_scan = None
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.last_scan is not None:
                break
            time.sleep(0.05)

        if self.last_scan is None:
            return []

        scan = self.last_scan
        points = []
        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            if r < scan.range_min or r > scan.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)

            cos_y = math.cos(robot_yaw)
            sin_y = math.sin(robot_yaw)
            wx = robot_x + lx * cos_y - ly * sin_y
            wy = robot_y + lx * sin_y + ly * cos_y

            points.append((wx, wy))

        return points

    def scan_grid(self, x_min, x_max, y_min, y_max, grid_step,
                    yaws=None, scan_settle_time=0.3):
        """Telegrapie le robot sur une grille et collecte les scans."""
        if yaws is None:
            yaws = [0.0, math.pi/2, math.pi, -math.pi/2]

        x_positions = np.arange(x_min, x_max + grid_step/2, grid_step)
        y_positions = np.arange(y_min, y_max + grid_step/2, grid_step)

        total = len(x_positions) * len(y_positions) * len(yaws)
        self.get_logger().info(
            f'Grille : {len(x_positions)}x{len(y_positions)} x {len(yaws)} = {total} scans'
        )

        n = 0
        n_failed_teleport = 0
        n_failed_verify = 0

        for x in x_positions:
            for y in y_positions:
                for yaw in yaws:
                    n += 1

                    # Teleport
                    if not self.teleport(float(x), float(y), float(yaw)):
                        n_failed_teleport += 1
                        continue

                    # Petite pause pour laisser Gazebo appliquer la pose
                    time.sleep(0.1)

                    # Verifier via la pose verite Gazebo
                    ok, d = self.verify_teleport(float(x), float(y), tolerance=0.5)
                    if not ok:
                        # Position non atteinte = probablement dans un mur
                        n_failed_verify += 1
                        continue

                    # Pause stabilisation scan
                    time.sleep(scan_settle_time)

                    # Capture
                    points = self.capture_scan_at(float(x), float(y), float(yaw))
                    for p in points:
                        self.scan_points.append((p[0], p[1]))
                        self.robot_positions.append((x, y))

                    if n % 10 == 0:
                        self.get_logger().info(
                            f'  {n}/{total} ({100*n/total:.0f}%) - '
                            f'{len(self.scan_points)} obstacles, '
                            f'{len(self.robot_positions)} libres, '
                            f'echecs: {n_failed_teleport} teleport, {n_failed_verify} verif'
                        )

        self.get_logger().info(
            f'\nScan termine: {len(self.scan_points)} obstacles, '
            f'{len(self.robot_positions)} positions libres, '
            f'{n_failed_teleport + n_failed_verify} echecs'
        )

def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x1, y1))
    return points

def build_map(scan_points, robot_positions, resolution=0.05, padding=2.0):
    if not scan_points:
        print("ERREUR: aucun point scanne")
        sys.exit(1)

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

    # log-odds simple
    log_odds = np.zeros((height_px, width_px), dtype=np.float32)

    def world_to_pixel(x, y):
        col = int((x - x_min) / resolution)
        row = height_px - 1 - int((y - y_min) / resolution)
        return col, row

    # paramètres (à ajuster si besoin)
    LO_FREE = -1.0
    LO_OCC = 2.0
    LO_MIN = -5
    LO_MAX = 5

    # 🔥 IMPORTANT : ray tracing
    for (rx, ry), (px, py) in zip(robot_positions, scan_points):
        rc, rr = world_to_pixel(rx, ry)
        pc, pr = world_to_pixel(px, py)

        ray = bresenham(rc, rr, pc, pr)

        # libre (tout sauf dernier point)
        for c, r in ray[:-1]:
            if 0 <= r < height_px and 0 <= c < width_px:
                log_odds[r, c] += LO_FREE

        # obstacle (dernier point)
        c, r = ray[-1]
        if 0 <= r < height_px and 0 <= c < width_px:
            log_odds[r, c] += LO_OCC

    # clamp
    log_odds = np.clip(log_odds, LO_MIN, LO_MAX)

    # conversion en image
    img = np.full((height_px, width_px), 205, dtype=np.uint8)

    img[log_odds > 1.0] = 0      # obstacle
    img[log_odds < -1.0] = 254   # libre

    print(f"Pixels libres   : {(img == 254).sum()}")
    print(f"Pixels obstacle : {(img == 0).sum()}")
    print(f"Pixels inconnu  : {(img == 205).sum()}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('world_name')
    parser.add_argument('output_name')
    parser.add_argument('--bounds', type=float, nargs=4,
                        default=[-7, 7, -7, 7],
                        metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'))
    parser.add_argument('--grid-step', type=float, default=1.5)
    parser.add_argument('--scans-per-pos', type=int, default=4)
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--padding', type=float, default=1.0)
    parser.add_argument('--model', default='limo')
    parser.add_argument('--scan-settle', type=float, default=0.3,
                        help='Pause apres teleport avant capture scan (s)')
    args = parser.parse_args()

    rclpy.init()
    mapper = AutoScanMapperV2(world_name=args.world_name, model_name=args.model)

    print(f"\n=== Configuration ===")
    print(f"Monde         : {args.world_name}")
    print(f"Modele        : {args.model}")
    print(f"Bornes XY     : x[{args.bounds[0]}, {args.bounds[1]}] "
          f"y[{args.bounds[2]}, {args.bounds[3]}]")
    print(f"Grid step     : {args.grid_step}m")
    print(f"Orientations  : {args.scans_per_pos}")
    print()

    # Verifier /scan
    print("Verification /scan...")
    for _ in range(20):
        rclpy.spin_once(mapper, timeout_sec=0.1)
        if mapper.last_scan is not None:
            print(f"  OK : {len(mapper.last_scan.ranges)} rays, max={mapper.last_scan.range_max}m")
            break
        time.sleep(0.1)
    else:
        print("ERREUR: /scan absent")
        sys.exit(1)

    # Verifier acces a la pose verite
    print("Verification pose verite Gazebo...")
    pose = mapper.get_robot_pose_from_gz()
    if pose is None:
        print("ERREUR: impossible de lire /world/<>/pose/info")
        print(f"Verifie: gz topic -e -t /world/{args.world_name}/pose/info -n 1")
        sys.exit(1)
    print(f"  OK : robot a ({pose[0]:.2f}, {pose[1]:.2f})")

    # Lancer le scan
    yaws = [i * 2 * math.pi / args.scans_per_pos for i in range(args.scans_per_pos)]
    mapper.scan_grid(
        x_min=args.bounds[0], x_max=args.bounds[1],
        y_min=args.bounds[2], y_max=args.bounds[3],
        grid_step=args.grid_step,
        yaws=yaws,
        scan_settle_time=args.scan_settle,
    )

    # Construire et sauver
    img, ox, oy = build_map(
        mapper.scan_points, mapper.robot_positions,
        resolution=args.resolution, padding=args.padding,
    )
    save_map(img, ox, oy, args.resolution, args.output_name)

    mapper.destroy_node()
    rclpy.shutdown()
    print("\nDone.")


if __name__ == '__main__':
    main()
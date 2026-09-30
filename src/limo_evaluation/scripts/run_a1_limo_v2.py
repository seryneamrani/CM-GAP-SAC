"""
run_a1_limo_v2.py - 15 runs A1 sur monde limo AVEC teleportation Gazebo.

Nouveaute v2 :
  - Teleporte vraiment le robot dans Gazebo entre chaque run via
    le service /world/<world>/set_pose
  - Verifie que /odom reflete bien la nouvelle position avant de continuer

Pre-requis:
  - navigation.launch.py world:=limo map:=limo doit etre lance
  - AMCL doit etre initialise (au moins une fois)
  - Tous les 9 nodes Nav2 doivent etre active

Resultats: /tmp/limo_eval_results/A1_limo/manual_metrics_v2.csv
"""
import os
import csv
import math
import time
import yaml
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose


def yaw_to_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class A1RunnerV2(Node):
    def __init__(self, world_name='default'):
        super().__init__('a1_runner_v2')

        self.world_name = world_name  # nom du monde Gazebo (default)
        self.gz_set_pose_service = f'/world/{world_name}/set_pose'

        # Pubs / clients
        self.initpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Pour metriques
        self.odom_positions = []
        self.scan_distances = []
        self.cmd_vel_count = 0
        self.last_odom = None  # pour verifier la teleportation

        self.create_subscription(Odometry, '/odom', self._odom_cb, 50)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 50)

        self.recording = False

        # Charger waypoints
        from ament_index_python.packages import get_package_share_directory
        pkg = get_package_share_directory('limo_evaluation')
        wp_file = os.path.join(pkg, 'config', 'waypoints.yaml')
        with open(wp_file) as f:
            self.waypoints = yaml.safe_load(f)

        # Output CSV
        self.output_dir = '/tmp/limo_eval_results/A1_limo'
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, 'manual_metrics_v2.csv')

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'run_id', 'pair_id', 'rep_id', 'success', 'duration_s',
                'distance_m', 'optimal_distance_m', 'spl',
                'd_min_obstacle_m', 'cmd_vel_count', 'teleport_ok'
            ])

        self.get_logger().info(f'RunnerV2 pret. CSV: {self.csv_path}')
        self.get_logger().info(f'Service teleport: {self.gz_set_pose_service}')

    def _odom_cb(self, msg):
        self.last_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.recording:
            self.odom_positions.append(self.last_odom)

    def _scan_cb(self, msg):
        if self.recording:
            valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
            if valid:
                self.scan_distances.append(min(valid))

    def _cmd_vel_cb(self, msg):
        if self.recording:
            self.cmd_vel_count += 1

    def reset_recording(self):
        self.odom_positions = []
        self.scan_distances = []
        self.cmd_vel_count = 0

    def teleport_robot_in_gazebo(self, x, y, yaw):
        """Teleporte le robot dans Gazebo via gz service. Retourne True si OK."""
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        # Format de la requete Gazebo Pose
        req = (
            f'name: "limo", '
            f'position: {{x: {x}, y: {y}, z: 0.2}}, '
            f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}'
        )

        cmd = [
            'gz', 'service',
            '-s', self.gz_set_pose_service,
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', req
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=5, text=True
            )
            success = 'data: true' in result.stdout.lower()
            return success
        except subprocess.TimeoutExpired:
            self.get_logger().warn('  Teleport timeout')
            return False
        except Exception as e:
            self.get_logger().warn(f'  Teleport erreur: {e}')
            return False

    def wait_odom_at_position(self, target_x, target_y, tolerance=0.3,
                                timeout=3.0):
        """Attend que /odom reflete la nouvelle position teleportee."""
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.last_odom is not None:
                dx = self.last_odom[0] - target_x
                dy = self.last_odom[1] - target_y
                if math.sqrt(dx*dx + dy*dy) < tolerance:
                    return True
            time.sleep(0.1)
        return False

    def publish_initialpose(self, x, y, yaw, n=5):
        for i in range(n):
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            msg.pose.covariance[0] = 0.25
            msg.pose.covariance[7] = 0.25
            msg.pose.covariance[35] = 0.0685
            self.initpose_pub.publish(msg)
            time.sleep(0.3)

    def send_goal(self, x, y, yaw, timeout=60.0):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        send_future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        gh = send_future.result()
        if not gh or not gh.accepted:
            return ('aborted', 0.0)

        result_future = gh.get_result_async()
        start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.5)
            if result_future.done():
                duration = time.time() - start
                result = result_future.result()
                outcome = 'success' if result.status == 4 else 'failure'
                return (outcome, duration)
            if time.time() - start > timeout:
                cancel_future = gh.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=3.0)
                return ('timeout', timeout)

        return ('aborted', 0.0)

    def compute_metrics(self, run_id, pair_id, rep_id, success, duration,
                        start_pose, goal_pose, teleport_ok):
        # Distance parcourue
        dist = 0.0
        for i in range(1, len(self.odom_positions)):
            dx = self.odom_positions[i][0] - self.odom_positions[i-1][0]
            dy = self.odom_positions[i][1] - self.odom_positions[i-1][1]
            dist += math.sqrt(dx*dx + dy*dy)

        # Distance optimale
        opt_dist = math.sqrt(
            (goal_pose[0] - start_pose[0])**2 +
            (goal_pose[1] - start_pose[1])**2
        )

        # SPL
        spl = 0.0
        if success and dist > 0:
            spl = opt_dist / max(dist, opt_dist)

        # d_min
        d_min = min(self.scan_distances) if self.scan_distances else float('nan')

        return {
            'run_id': run_id,
            'pair_id': pair_id,
            'rep_id': rep_id,
            'success': int(success),
            'duration_s': round(duration, 2),
            'distance_m': round(dist, 3),
            'optimal_distance_m': round(opt_dist, 3),
            'spl': round(spl, 3),
            'd_min_obstacle_m': round(d_min, 3) if not math.isnan(d_min) else 'nan',
            'cmd_vel_count': self.cmd_vel_count,
            'teleport_ok': int(teleport_ok),
        }

    def run_all(self, world='limo', reps=3):
        if world not in self.waypoints['worlds']:
            self.get_logger().error(f'Monde {world} introuvable')
            return

        self.get_logger().info('Attente nav_client...')
        self.nav_client.wait_for_server()

        pairs = self.waypoints['worlds'][world]
        total_runs = len(pairs) * reps
        run_idx = 0

        for pair_name, pair_data in pairs.items():
            pair_id = int(pair_name.split('_')[-1])
            start = pair_data['start']
            goal = pair_data['goal']

            self.get_logger().info(
                f"\n=== Pair {pair_id} : {pair_data.get('description','')} ==="
            )

            for rep in range(1, reps + 1):
                run_idx += 1
                run_id = f"A1_{world}_p{pair_id}_r{rep}"

                self.get_logger().info(
                    f">>> RUN {run_idx}/{total_runs} : {run_id}"
                )

                # 1. TELEPORTER le robot dans Gazebo (NOUVEAU v2)
                self.get_logger().info(
                    f"   Teleport robot -> ({start['x']:.2f}, {start['y']:.2f})"
                )
                teleport_ok = self.teleport_robot_in_gazebo(
                    start['x'], start['y'], start['yaw']
                )

                if teleport_ok:
                    # Attendre que /odom reflete la nouvelle position
                    if self.wait_odom_at_position(start['x'], start['y']):
                        self.get_logger().info("   Teleport confirme via /odom")
                    else:
                        self.get_logger().warn("   /odom ne reflete pas encore la nouvelle pose")
                else:
                    self.get_logger().warn("   Teleport echoue, on continue quand meme")

                # 2. Re-init AMCL sur cette nouvelle position
                self.publish_initialpose(start['x'], start['y'], start['yaw'])
                time.sleep(2.0)  # AMCL converge

                # 3. Reset metriques et commencer recording
                self.reset_recording()
                self.recording = True

                # 4. Envoyer le goal
                outcome, duration = self.send_goal(
                    goal['x'], goal['y'], goal['yaw'], timeout=60.0
                )

                # 5. Stop recording
                self.recording = False

                # 6. Calculer metriques
                metrics = self.compute_metrics(
                    run_id, pair_id, rep,
                    success=(outcome == 'success'),
                    duration=duration,
                    start_pose=(start['x'], start['y']),
                    goal_pose=(goal['x'], goal['y']),
                    teleport_ok=teleport_ok
                )

                # 7. Ecrire CSV
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        metrics['run_id'], metrics['pair_id'], metrics['rep_id'],
                        metrics['success'], metrics['duration_s'],
                        metrics['distance_m'], metrics['optimal_distance_m'],
                        metrics['spl'],
                        metrics['d_min_obstacle_m'], metrics['cmd_vel_count'],
                        metrics['teleport_ok']
                    ])

                self.get_logger().info(
                    f"   [{outcome}] dur={duration:.1f}s "
                    f"dist={metrics['distance_m']}m "
                    f"opt={metrics['optimal_distance_m']}m "
                    f"spl={metrics['spl']} "
                    f"d_min={metrics['d_min_obstacle_m']}m "
                    f"teleport={metrics['teleport_ok']}"
                )

                time.sleep(2.0)

        self.get_logger().info(
            f"\n=== FIN : {total_runs} runs effectues ==="
        )

        # Stats finales
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        n = len(rows)
        n_success = sum(1 for r in rows if r['success'] == '1')
        n_teleport = sum(1 for r in rows if r['teleport_ok'] == '1')

        self.get_logger().info(
            f"Success rate : {n_success}/{n} = {100*n_success/n:.1f}%"
        )
        self.get_logger().info(
            f"Teleport OK  : {n_teleport}/{n}"
        )
        self.get_logger().info(f"CSV: {self.csv_path}")


def main():
    rclpy.init()
    runner = A1RunnerV2(world_name='default')
    try:
        runner.run_all(world='limo', reps=3)
    except KeyboardInterrupt:
        runner.get_logger().info('Interrompu')
    finally:
        runner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

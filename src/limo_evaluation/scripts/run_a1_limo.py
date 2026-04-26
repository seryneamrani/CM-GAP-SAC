"""
run_a1_limo.py - Lance 15 runs A1 sur monde limo en bypass du harness.

Pre-requis:
  - navigation.launch.py world:=limo map:=limo doit etre lance
  - AMCL doit etre initialise et converge
  - Tous les 9 nodes Nav2 doivent etre active

Resultats: /tmp/limo_eval_results/A1_limo/manual_metrics.csv
"""
import os
import csv
import math
import time
import yaml

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose


def yaw_to_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class A1Runner(Node):
    def __init__(self):
        super().__init__('a1_runner_manual')

        # Pubs / subs / clients
        self.initpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Pour metriques
        self.odom_positions = []
        self.scan_distances = []
        self.cmd_vel_count = 0

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
        self.csv_path = os.path.join(self.output_dir, 'manual_metrics.csv')

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'run_id', 'pair_id', 'rep_id', 'success', 'duration_s',
                'distance_m', 'optimal_distance_m', 'spl',
                'd_min_obstacle_m', 'cmd_vel_count'
            ])

        self.get_logger().info(f'Runner pret. CSV: {self.csv_path}')

    def _odom_cb(self, msg):
        if self.recording:
            self.odom_positions.append(
                (msg.pose.pose.position.x, msg.pose.pose.position.y)
            )

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
        """Retourne (outcome, duration) avec outcome dans {success, failure, timeout}."""
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
                        start_pose, goal_pose):
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
        results = []

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

                # 1. Repositionner robot
                self.publish_initialpose(start['x'], start['y'], start['yaw'])
                time.sleep(2.5)  # AMCL converge

                # 2. Reset metriques et commencer recording
                self.reset_recording()
                self.recording = True

                # 3. Envoyer le goal
                outcome, duration = self.send_goal(
                    goal['x'], goal['y'], goal['yaw'], timeout=60.0
                )

                # 4. Stop recording
                self.recording = False

                # 5. Calculer metriques
                metrics = self.compute_metrics(
                    run_id, pair_id, rep,
                    success=(outcome == 'success'),
                    duration=duration,
                    start_pose=(start['x'], start['y']),
                    goal_pose=(goal['x'], goal['y'])
                )

                # 6. Ecrire CSV
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        metrics['run_id'], metrics['pair_id'], metrics['rep_id'],
                        metrics['success'], metrics['duration_s'],
                        metrics['distance_m'], metrics['optimal_distance_m'],
                        metrics['spl'],
                        metrics['d_min_obstacle_m'], metrics['cmd_vel_count']
                    ])

                self.get_logger().info(
                    f"    [{outcome}] dur={duration:.1f}s "
                    f"dist={metrics['distance_m']}m "
                    f"opt={metrics['optimal_distance_m']}m "
                    f"spl={metrics['spl']} "
                    f"d_min={metrics['d_min_obstacle_m']}m"
                )
                results.append(metrics)

                time.sleep(2.0)  # Pause entre runs

        self.get_logger().info(
            f"\n=== FIN : {total_runs} runs effectues ==="
        )
        n_success = sum(1 for r in results if r['success'])
        self.get_logger().info(
            f"Success rate: {n_success}/{total_runs} = {100*n_success/total_runs:.1f}%"
        )
        self.get_logger().info(f"CSV: {self.csv_path}")


def main():
    rclpy.init()
    runner = A1Runner()
    try:
        runner.run_all(world='limo', reps=3)
    except KeyboardInterrupt:
        runner.get_logger().info('Interrompu')
    finally:
        runner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

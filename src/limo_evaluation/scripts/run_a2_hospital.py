"""
run_a2_hospital.py - 15 runs A2 sur monde hospital.

Identique a a1_runner_v3 mais :
  - Sortie : /tmp/limo_eval_results/A2_hospital/manual_metrics_v3.csv
  - Run ID : A2_hospital_pX_rY
  - Suppose que perception_node + obstacle_projector tournent (lances par
    sim.launch.py perception:=true)

Pre-requis :
  ros2 launch limo_description navigation.launch.py \\
      world:=hospital map:=hospital perception:=true
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
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition


def yaw_to_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class A2Runner(Node):
    NODES_TO_ACTIVATE = [
        'planner_server', 'behavior_server', 'bt_navigator',
        'velocity_smoother', 'waypoint_follower',
    ]

    def __init__(self, world_name='hospital', arch='A2'):
        super().__init__('a2_runner')

        self.world_name = world_name
        self.arch = arch
        self.gz_set_pose_service = f'/world/{world_name}/set_pose'

        self.initpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.odom_positions = []
        self.scan_distances = []
        self.cmd_vel_count = 0
        self.last_odom = None
        self.last_amcl_pose = None

        self.create_subscription(Odometry, '/odom', self._odom_cb, 50)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 50)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10)

        self.recording = False

        from ament_index_python.packages import get_package_share_directory
        pkg = get_package_share_directory('limo_evaluation')
        wp_file = os.path.join(pkg, 'config', 'waypoints.yaml')
        with open(wp_file) as f:
            self.waypoints = yaml.safe_load(f)

        # Sortie A2
        self.output_dir = f'/tmp/limo_eval_results/{arch}_{world_name}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, 'manual_metrics_v3.csv')

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'run_id', 'pair_id', 'rep_id', 'success', 'duration_s',
                'distance_m', 'optimal_distance_m', 'spl',
                'd_min_obstacle_m', 'cmd_vel_count', 'teleport_ok',
                'amcl_converged'
            ])

        self.get_logger().info(f'{arch} Runner pret. CSV: {self.csv_path}')

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

    def _amcl_cb(self, msg):
        self.last_amcl_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def reset_recording(self):
        self.odom_positions = []
        self.scan_distances = []
        self.cmd_vel_count = 0

    def activate_lifecycle_nodes(self):
        self.get_logger().info('  Activation lifecycle nodes...')
        n_activated = 0
        for node_name in self.NODES_TO_ACTIVATE:
            client = self.create_client(ChangeState, f'/{node_name}/change_state')
            if not client.wait_for_service(timeout_sec=2.0):
                continue
            req = ChangeState.Request()
            req.transition.id = Transition.TRANSITION_ACTIVATE
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.done() and future.result() and future.result().success:
                n_activated += 1
        self.get_logger().info(f'  {n_activated}/{len(self.NODES_TO_ACTIVATE)} actives')

    def publish_initialpose(self, x, y, yaw, n=8, sleep_between=0.4):
        for _ in range(n):
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.position.x = float(x)
            msg.pose.pose.position.y = float(y)
            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            msg.pose.covariance[0] = 0.25
            msg.pose.covariance[7] = 0.25
            msg.pose.covariance[35] = 0.0685
            self.initpose_pub.publish(msg)
            time.sleep(sleep_between)

    def wait_amcl_converged(self, target_x, target_y, tolerance=0.5, timeout=8.0):
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.last_amcl_pose is not None:
                dx = self.last_amcl_pose[0] - target_x
                dy = self.last_amcl_pose[1] - target_y
                d = math.sqrt(dx*dx + dy*dy)
                if d < tolerance:
                    return True, d
            time.sleep(0.1)
        return False, math.inf

    def teleport_robot_in_gazebo(self, x, y, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        req = (
            f'name: "limo", '
            f'position: {{x: {x}, y: {y}, z: 0.2}}, '
            f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}'
        )
        cmd = ['gz', 'service', '-s', self.gz_set_pose_service,
               '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
               '--timeout', '2000', '--req', req]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5, text=True)
            return 'data: true' in result.stdout.lower()
        except Exception as e:
            self.get_logger().warn(f'Teleport erreur: {e}')
            return False

    def wait_odom_at_position(self, target_x, target_y, tolerance=0.3, timeout=3.0):
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

    def reposition_robot(self, x, y, yaw):
        teleport_ok = self.teleport_robot_in_gazebo(x, y, yaw)
        if not teleport_ok:
            self.get_logger().warn(f'  Teleport echoue')
            return False, False
        self.wait_odom_at_position(x, y, tolerance=0.5, timeout=4.0)
        time.sleep(0.5)
        self.activate_lifecycle_nodes()
        time.sleep(0.5)
        self.publish_initialpose(x, y, yaw, n=8, sleep_between=0.4)
        amcl_ok, dist = self.wait_amcl_converged(x, y, tolerance=0.5, timeout=8.0)
        if amcl_ok:
            self.get_logger().info(f'  AMCL converge (delta={dist:.2f}m)')
        else:
            self.get_logger().warn(f'  AMCL pas converge (delta={dist:.2f}m)')
        return teleport_ok, amcl_ok

    def initial_setup(self):
        self.get_logger().info('=== Setup initial ===')
        self.activate_lifecycle_nodes()
        time.sleep(2.0)
        self.publish_initialpose(0.0, 0.0, 0.0, n=8)
        ok, d = self.wait_amcl_converged(0.0, 0.0, tolerance=0.5, timeout=10.0)
        if ok:
            self.get_logger().info(f'  AMCL initial converge (delta={d:.2f}m)')
        else:
            self.get_logger().warn(f'  AMCL initial pas converge (delta={d:.2f}m)')
        self.get_logger().info('=== Setup termine ===\n')

    def send_goal(self, x, y, yaw, timeout=180.0):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
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
                        start_pose, goal_pose, teleport_ok, amcl_ok):
        dist = 0.0
        for i in range(1, len(self.odom_positions)):
            dx = self.odom_positions[i][0] - self.odom_positions[i-1][0]
            dy = self.odom_positions[i][1] - self.odom_positions[i-1][1]
            dist += math.sqrt(dx*dx + dy*dy)
        opt_dist = math.sqrt(
            (goal_pose[0] - start_pose[0])**2 +
            (goal_pose[1] - start_pose[1])**2)
        spl = 0.0
        if success and dist > 0:
            spl = opt_dist / max(dist, opt_dist)
        d_min = min(self.scan_distances) if self.scan_distances else float('nan')
        return {
            'run_id': run_id, 'pair_id': pair_id, 'rep_id': rep_id,
            'success': int(success),
            'duration_s': round(duration, 2),
            'distance_m': round(dist, 3),
            'optimal_distance_m': round(opt_dist, 3),
            'spl': round(spl, 3),
            'd_min_obstacle_m': round(d_min, 3) if not math.isnan(d_min) else 'nan',
            'cmd_vel_count': self.cmd_vel_count,
            'teleport_ok': int(teleport_ok),
            'amcl_converged': int(amcl_ok),
        }

    def run_all(self, world='hospital', reps=3):
        if world not in self.waypoints['worlds']:
            self.get_logger().error(f'Monde {world} introuvable')
            return
        self.get_logger().info('Attente nav_client...')
        self.nav_client.wait_for_server()
        self.initial_setup()
        pairs = self.waypoints['worlds'][world]
        total_runs = len(pairs) * reps
        run_idx = 0
        for pair_name, pair_data in pairs.items():
            pair_id = int(pair_name.split('_')[-1])
            start = pair_data['start']
            goal = pair_data['goal']
            self.get_logger().info(
                f"\n=== Pair {pair_id} : {pair_data.get('description','')} ===")
            for rep in range(1, reps + 1):
                run_idx += 1
                run_id = f"{self.arch}_{world}_p{pair_id}_r{rep}"
                self.get_logger().info(f">>> RUN {run_idx}/{total_runs} : {run_id}")
                self.get_logger().info(
                    f"  Repositionnement -> ({start['x']:.2f}, {start['y']:.2f}, "
                    f"yaw={math.degrees(start['yaw']):.0f}deg)")
                teleport_ok, amcl_ok = self.reposition_robot(
                    start['x'], start['y'], start['yaw'])
                self.reset_recording()
                self.recording = True
                outcome, duration = self.send_goal(
                    goal['x'], goal['y'], goal['yaw'], timeout=180.0)
                self.recording = False
                metrics = self.compute_metrics(
                    run_id, pair_id, rep,
                    success=(outcome == 'success'), duration=duration,
                    start_pose=(start['x'], start['y']),
                    goal_pose=(goal['x'], goal['y']),
                    teleport_ok=teleport_ok, amcl_ok=amcl_ok)
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        metrics['run_id'], metrics['pair_id'], metrics['rep_id'],
                        metrics['success'], metrics['duration_s'],
                        metrics['distance_m'], metrics['optimal_distance_m'],
                        metrics['spl'],
                        metrics['d_min_obstacle_m'], metrics['cmd_vel_count'],
                        metrics['teleport_ok'], metrics['amcl_converged']
                    ])
                self.get_logger().info(
                    f"  [{outcome}] dur={duration:.1f}s "
                    f"dist={metrics['distance_m']}m "
                    f"opt={metrics['optimal_distance_m']}m "
                    f"spl={metrics['spl']} "
                    f"d_min={metrics['d_min_obstacle_m']}m")
                time.sleep(1.5)
        self.get_logger().info(f"\n=== FIN : {total_runs} runs effectues ===")
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n = len(rows)
        n_success = sum(1 for r in rows if r['success'] == '1')
        n_amcl = sum(1 for r in rows if r['amcl_converged'] == '1')
        self.get_logger().info(f"Success rate    : {n_success}/{n} = {100*n_success/n:.1f}%")
        self.get_logger().info(f"AMCL converged  : {n_amcl}/{n}")
        self.get_logger().info(f"CSV: {self.csv_path}")


def main():
    rclpy.init()
    runner = A2Runner(world_name='hospital', arch='A2')
    try:
        runner.run_all(world='hospital', reps=3)
    except KeyboardInterrupt:
        runner.get_logger().info('Interrompu')
    finally:
        runner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

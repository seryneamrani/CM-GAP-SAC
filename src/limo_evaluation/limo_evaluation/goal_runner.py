"""
goal_runner.py - Envoie automatiquement les goals Nav2 selon waypoints.yaml.

VERSION ROBUSTE: attend la convergence AMCL avant de demarrer les runs.

Setup :
  1. Active manuellement les 5 nodes lifecycle bloques en inactive
  2. Publie /initialpose 5 fois sur 3 secondes
  3. Verifie que TF map->odom existe (= AMCL converge)
  4. Lance les runs

Workflow par run :
  1. Charger waypoint
  2. Re-publier /initialpose pour repositionner le robot
  3. Attendre AMCL converge sur la nouvelle position
  4. Publier /eval/run_config + /eval/start_run
  5. Envoyer le goal via action /navigate_to_pose
  6. Attendre soit succes soit timeout
  7. /eval/end_run_*
  8. Pause + repeter
"""
import os
import time
import math
from typing import Optional

import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


def yaw_to_quaternion(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class GoalRunner(Node):
    def __init__(self):
        super().__init__('goal_runner')

        self.declare_parameter('world', 'limo')
        self.declare_parameter('architecture', 'A1')
        self.declare_parameter('repetitions', 3)
        self.declare_parameter('timeout_per_run', 120.0)
        self.declare_parameter('reset_pause', 3.0)
        self.declare_parameter('waypoints_file', '')

        self.world = self.get_parameter('world').value
        self.arch = self.get_parameter('architecture').value
        self.reps = self.get_parameter('repetitions').value
        self.timeout = self.get_parameter('timeout_per_run').value
        self.pause = self.get_parameter('reset_pause').value
        wp_file = self.get_parameter('waypoints_file').value

        if not wp_file:
            from ament_index_python.packages import get_package_share_directory
            pkg = get_package_share_directory('limo_evaluation')
            wp_file = os.path.join(pkg, 'config', 'waypoints.yaml')

        with open(wp_file) as f:
            self.waypoints = yaml.safe_load(f)

        if self.world not in self.waypoints['worlds']:
            self.get_logger().error(
                f'Monde {self.world} introuvable dans waypoints.yaml'
            )
            raise SystemExit(1)

        # Pubs / clients
        self.config_pub = self.create_publisher(String, '/eval/run_config', 10)
        self.initpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10
        )

        self.client_start = self.create_client(Trigger, '/eval/start_run')
        self.client_end_success = self.create_client(Trigger, '/eval/end_run_success')
        self.client_end_timeout = self.create_client(Trigger, '/eval/end_run_timeout')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # TF listener pour verifier convergence AMCL
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f'GoalRunner pret. Monde={self.world} archi={self.arch} reps={self.reps}'
        )

    def wait_for_services(self):
        self.get_logger().info('Attente des services /eval/* et /navigate_to_pose...')
        self.client_start.wait_for_service()
        self.client_end_success.wait_for_service()
        self.client_end_timeout.wait_for_service()
        self.nav_client.wait_for_server()
        self.get_logger().info('Tous les services disponibles.')

    def activate_lifecycle_nodes(self):
        """Active manuellement les nodes lifecycle qui restent inactifs."""
        nodes_to_activate = [
            'planner_server',
            'behavior_server',
            'bt_navigator',
            'velocity_smoother',
            'waypoint_follower',
        ]

        self.get_logger().info('Activation des nodes lifecycle...')
        for node_name in nodes_to_activate:
            client = self.create_client(ChangeState, f'/{node_name}/change_state')
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f'  {node_name}: service indisponible')
                continue

            req = ChangeState.Request()
            req.transition.id = Transition.TRANSITION_ACTIVATE
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.done() and future.result() and future.result().success:
                self.get_logger().info(f'  {node_name}: activated')
            else:
                # Ce n'est pas grave si c'est deja active - le service retourne un echec
                self.get_logger().info(f'  {node_name}: deja active (ou echec, on continue)')

    def publish_initial_pose(self, x: float, y: float, yaw: float, n_times: int = 5):
        """Publie /initialpose plusieurs fois pour etre sur qu'AMCL recoit."""
        # Attendre que le subscriber AMCL soit la
        for i in range(20):
            if self.initpose_pub.get_subscription_count() > 0:
                break
            time.sleep(0.2)

        for i in range(n_times):
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
            time.sleep(0.5)

    def wait_for_amcl_convergence(self, timeout_sec: float = 15.0) -> bool:
        """Attend que TF map->odom existe (= AMCL a converge)."""
        self.get_logger().info('Attente convergence AMCL (TF map->odom)...')
        start = time.time()
        while time.time() - start < timeout_sec:
            try:
                self.tf_buffer.lookup_transform(
                    'map', 'odom',
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.5)
                )
                self.get_logger().info(
                    f'  AMCL converge en {time.time() - start:.1f}s'
                )
                return True
            except (LookupException, ConnectivityException, ExtrapolationException):
                rclpy.spin_once(self, timeout_sec=0.2)
                time.sleep(0.3)

        self.get_logger().error('  AMCL n\'a pas converge dans le timeout')
        return False

    def setup_navigation_stack(self):
        """Setup complet : lifecycle + initial pose + verification convergence."""
        self.get_logger().info('=== Setup navigation stack ===')

        # 1. Active les nodes lifecycle
        self.activate_lifecycle_nodes()
        time.sleep(2.0)

        # 2. Publie pose initiale au centre (0,0,0)
        self.get_logger().info('Publication initialpose (0, 0, 0)...')
        self.publish_initial_pose(0.0, 0.0, 0.0, n_times=5)

        # 3. Attend convergence AMCL
        if not self.wait_for_amcl_convergence(timeout_sec=15.0):
            self.get_logger().warn(
                'AMCL pas converge - on continue quand meme, '
                'risque d\'echecs sur premiers runs'
            )

        self.get_logger().info('=== Setup termine ===')

    def publish_run_config(self, pair_id: int, rep_id: int, start, goal):
        config_str = (
            f"architecture={self.arch};world={self.world};"
            f"pair_id={pair_id};repetition_id={rep_id};"
            f"start_x={start['x']};start_y={start['y']};start_yaw={start['yaw']};"
            f"goal_x={goal['x']};goal_y={goal['y']};goal_yaw={goal['yaw']}"
        )
        msg = String()
        msg.data = config_str
        self.config_pub.publish(msg)
        time.sleep(0.5)

    def call_trigger(self, client, label: str) -> bool:
        req = Trigger.Request()
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() and future.result().success:
            return True
        self.get_logger().warn(f'Service {label} echoue ou timeout')
        return False

    def send_goal_and_wait(self, x: float, y: float, yaw: float) -> str:
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(f'Envoi goal : ({x:.2f}, {y:.2f}, {yaw:.2f})')
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=10.0)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('Goal refuse')
            return 'aborted'

        result_future = goal_handle.get_result_async()
        start = time.time()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.5)
            if result_future.done():
                result = result_future.result()
                if result.status == 4:
                    return 'success'
                else:
                    return 'failure'

            if time.time() - start > self.timeout:
                self.get_logger().warn(f'Timeout {self.timeout}s atteint, annulation')
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
                return 'timeout'

        return 'aborted'

    def run_all(self):
        self.wait_for_services()
        self.setup_navigation_stack()

        pairs = self.waypoints['worlds'][self.world]
        total_runs = len(pairs) * self.reps
        run_idx = 0

        for pair_name, pair_data in pairs.items():
            pair_id = int(pair_name.split('_')[-1])
            start = pair_data['start']
            goal = pair_data['goal']

            self.get_logger().info(
                f"=== Pair {pair_id} : {pair_data.get('description','')} ==="
            )

            for rep in range(1, self.reps + 1):
                run_idx += 1
                self.get_logger().info(
                    f">>> RUN {run_idx}/{total_runs} : pair={pair_id} rep={rep}"
                )

                # 1. Repositionner robot via initialpose
                self.publish_initial_pose(start['x'], start['y'], start['yaw'], n_times=3)
                time.sleep(2.0)

                # 2. Configurer le run et le demarrer
                self.publish_run_config(pair_id, rep, start, goal)
                if not self.call_trigger(self.client_start, 'start_run'):
                    self.get_logger().error('Echec start_run, on continue')
                    continue

                # 3. Envoyer le goal
                outcome = self.send_goal_and_wait(goal['x'], goal['y'], goal['yaw'])

                # 4. Terminer le run
                if outcome == 'success':
                    self.call_trigger(self.client_end_success, 'end_run_success')
                else:
                    self.call_trigger(self.client_end_timeout, 'end_run_timeout')

                # 5. Pause
                self.get_logger().info(f'Pause {self.pause}s avant prochain run...')
                time.sleep(self.pause)

        self.get_logger().info(
            f'=== TOUS LES RUNS TERMINES ({total_runs}) ==='
        )


def main():
    rclpy.init()
    runner = GoalRunner()
    try:
        runner.run_all()
    except KeyboardInterrupt:
        runner.get_logger().info('Interrompu par utilisateur')
    finally:
        runner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

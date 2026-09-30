"""
evaluation_node.py — Node ROS 2 de collecte de donnees et calcul de metriques.

Lance par evaluation.launch.py. Ecoute en parallele :
  /odom         — trajectoire reelle
  /cmd_vel      — commandes envoyees
  /scan         — pour collisions, d_min, TTC
  /plan         — replanifications globales
  /amcl_pose    — pose localisee

Service /eval/start_run     : commence la collecte pour un nouveau run
Service /eval/end_run       : termine, calcule les metriques, ecrit dans CSV

Topic /eval/metrics_live   : publie les metriques en cours pendant le run
"""
import os
import csv
import math
from typing import Optional, Dict, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger

from limo_evaluation.metrics import aggregate_run_metrics


# CSV columns en sortie (ordre fixe)
CSV_COLUMNS = [
    'run_id', 'architecture', 'world', 'pair_id', 'repetition_id',
    'success', 'failure_reason',
    'duration_s', 'distance_m', 'optimal_distance_m',
    'spl', 'tdi',
    'collisions', 'replanifications',
    'd_min_obstacle_m', 'jerk_mean', 'ttc_min_s',
    'perception_latency_p50_ms', 'perception_latency_p95_ms', 'perception_latency_p99_ms',
]


class EvaluationNode(Node):
    def __init__(self):
        super().__init__('evaluation_node')
        
        # Parametres
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('plan_topic', '/plan')
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('output_dir', '/tmp/limo_eval_results')
        self.declare_parameter('csv_filename', 'metrics.csv')
        self.declare_parameter('collision_distance', 0.05)
        self.declare_parameter('verbose', True)
        
        self.output_dir = self.get_parameter('output_dir').value
        self.csv_filename = self.get_parameter('csv_filename').value
        self.collision_dist = self.get_parameter('collision_distance').value
        self.verbose = self.get_parameter('verbose').value
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, self.csv_filename)
        self._init_csv_if_needed()
        
        # Etat run en cours
        self.recording = False
        self.run_data: Optional[Dict] = None
        self.run_start_time: Optional[float] = None
        
        # QoS pour topics capteurs (BEST_EFFORT typique)
        sensor_qos = QoSProfile(depth=10)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        
        nav_qos = QoSProfile(depth=10)
        nav_qos.reliability = ReliabilityPolicy.RELIABLE
        nav_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        
        # Subscribers
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._odom_cb, 50)
        self.create_subscription(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            self._cmd_vel_cb, 50)
        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self._scan_cb, sensor_qos)
        self.create_subscription(
            Path,
            self.get_parameter('plan_topic').value,
            self._plan_cb, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter('amcl_pose_topic').value,
            self._amcl_cb, 10)
        
        # Publishers et services
        self.metrics_pub = self.create_publisher(String, '/eval/metrics_live', 10)
        
        self.create_service(Trigger, '/eval/start_run', self._cb_start_run)
        self.create_service(Trigger, '/eval/end_run_success', self._cb_end_run_success)
        self.create_service(Trigger, '/eval/end_run_failure', self._cb_end_run_failure)
        self.create_service(Trigger, '/eval/end_run_timeout', self._cb_end_run_timeout)
        
        # Configuration du run courant (parametres ROS via topic ou service)
        self.create_subscription(
            String, '/eval/run_config', self._cb_run_config, 10)
        
        # Timer pour publier les metriques live
        self.create_timer(1.0, self._publish_live_metrics)
        
        self.get_logger().info(
            f'EvaluationNode pret. CSV : {self.csv_path}'
        )
    
    # =========================================================================
    # Initialisation CSV
    # =========================================================================
    def _init_csv_if_needed(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)
            self.get_logger().info(f'CSV cree : {self.csv_path}')
    
    # =========================================================================
    # Configuration du run via topic
    # =========================================================================
    def _cb_run_config(self, msg: String):
        """
        Recoit une config sous forme JSON-like simple :
        "architecture=A1;world=hospital;pair_id=1;repetition_id=2;\
         start_x=0;start_y=0;start_yaw=0;goal_x=8;goal_y=0;goal_yaw=0"
        """
        try:
            cfg = {}
            for part in msg.data.split(';'):
                k, v = part.split('=')
                cfg[k.strip()] = v.strip()
            self._pending_config = cfg
            self.get_logger().info(f'Config run en attente : {cfg}')
        except Exception as e:
            self.get_logger().error(f'Config invalide : {e}')
    
    # =========================================================================
    # Services start/end
    # =========================================================================
    def _cb_start_run(self, request, response):
        if not hasattr(self, '_pending_config'):
            response.success = False
            response.message = 'Aucune config en attente. Publie sur /eval/run_config d\'abord.'
            return response
        
        cfg = self._pending_config
        self.run_data = {
            'run_id': f"{cfg.get('architecture','?')}_{cfg.get('world','?')}_p{cfg.get('pair_id','?')}_r{cfg.get('repetition_id','?')}",
            'architecture': cfg.get('architecture', 'unknown'),
            'world': cfg.get('world', 'unknown'),
            'pair_id': int(cfg.get('pair_id', 0)),
            'repetition_id': int(cfg.get('repetition_id', 0)),
            'start_pose': (
                float(cfg.get('start_x', 0)),
                float(cfg.get('start_y', 0)),
                float(cfg.get('start_yaw', 0)),
            ),
            'goal_pose': (
                float(cfg.get('goal_x', 0)),
                float(cfg.get('goal_y', 0)),
                float(cfg.get('goal_yaw', 0)),
            ),
            'odom_positions': [],
            'odom_history_full': [],
            'cmd_vel_history': [],
            'scan_distances': [],
            'plan_timestamps': [],
            'image_timestamps': [],
            'detection_timestamps': [],
            'success': False,
            'reason': '',
            'duration_seconds': 0.0,
        }
        self.run_start_time = self.get_clock().now().nanoseconds / 1e9
        self.recording = True
        
        self.get_logger().info(
            f"=== RUN START : {self.run_data['run_id']} ==="
        )
        response.success = True
        response.message = f"Run {self.run_data['run_id']} demarre"
        return response
    
    def _end_run(self, success: bool, reason: str):
        if not self.recording or self.run_data is None:
            return False, 'Aucun run en cours'
        
        self.recording = False
        now = self.get_clock().now().nanoseconds / 1e9
        self.run_data['duration_seconds'] = now - self.run_start_time
        self.run_data['success'] = success
        self.run_data['reason'] = reason
        
        # Calcul metriques
        try:
            metrics = aggregate_run_metrics(self.run_data)
        except Exception as e:
            self.get_logger().error(f'Echec calcul metriques : {e}')
            return False, str(e)
        
        # Ecriture CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [metrics.get(col, '') for col in CSV_COLUMNS]
            writer.writerow(row)
        
        self.get_logger().info(
            f"=== RUN END : {self.run_data['run_id']} | "
            f"success={success} | reason={reason} | "
            f"duration={self.run_data['duration_seconds']:.1f}s | "
            f"dist={metrics.get('distance_m', 0):.2f}m | "
            f"collisions={metrics.get('collisions', 0)} ==="
        )
        
        # Reset
        self.run_data = None
        return True, f'Metriques ecrites dans {self.csv_path}'
    
    def _cb_end_run_success(self, request, response):
        ok, msg = self._end_run(success=True, reason='goal_reached')
        response.success = ok
        response.message = msg
        return response
    
    def _cb_end_run_failure(self, request, response):
        ok, msg = self._end_run(success=False, reason='failure')
        response.success = ok
        response.message = msg
        return response
    
    def _cb_end_run_timeout(self, request, response):
        ok, msg = self._end_run(success=False, reason='timeout')
        response.success = ok
        response.message = msg
        return response
    
    # =========================================================================
    # Subscribers callbacks
    # =========================================================================
    def _odom_cb(self, msg: Odometry):
        if not self.recording or self.run_data is None:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v_linear = math.sqrt(
            msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2
        )
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.run_data['odom_positions'].append((x, y))
        self.run_data['odom_history_full'].append((x, y, v_linear, t))
    
    def _cmd_vel_cb(self, msg: Twist):
        if not self.recording or self.run_data is None:
            return
        t = self.get_clock().now().nanoseconds / 1e9
        self.run_data['cmd_vel_history'].append(
            (msg.linear.x, msg.angular.z, t)
        )
    
    def _scan_cb(self, msg: LaserScan):
        if not self.recording or self.run_data is None:
            return
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid:
            self.run_data['scan_distances'].append(min(valid))
    
    def _plan_cb(self, msg: Path):
        if not self.recording or self.run_data is None:
            return
        t = self.get_clock().now().nanoseconds / 1e9
        self.run_data['plan_timestamps'].append(t)
    
    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        # Reserve pour utilisation future (si on veut comparer odom vs amcl)
        pass
    
    # =========================================================================
    # Publication metriques live
    # =========================================================================
    def _publish_live_metrics(self):
        if not self.recording or self.run_data is None:
            return
        
        n_odom = len(self.run_data['odom_positions'])
        n_scan = len(self.run_data['scan_distances'])
        d_min = min(self.run_data['scan_distances']) if n_scan else float('nan')
        
        elapsed = self.get_clock().now().nanoseconds / 1e9 - self.run_start_time
        
        msg = String()
        msg.data = (
            f"run={self.run_data['run_id']} | "
            f"elapsed={elapsed:.1f}s | "
            f"odom_samples={n_odom} | "
            f"d_min={d_min:.2f}m"
        )
        self.metrics_pub.publish(msg)
        
        if self.verbose:
            self.get_logger().info(msg.data, throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""
pose_picker.py — Utilitaire pour capturer les coordonnees du robot.

Usage : on lance ce node, on deplace le robot dans Gazebo (via teleop ou main
clic Gazebo), et a chaque appui sur ENTREE dans le terminal lancant ce node,
il imprime la pose actuelle dans le frame map.

ATTENTION : il faut que AMCL soit initialise (via 2D Pose Estimate dans RViz).

Resultat : tu peux placer le robot a tes 5 positions start et 5 goals par
monde, noter les coordonnees, et les copier-coller dans waypoints.yaml.

Lancement :
  ros2 run limo_evaluation pose_picker
"""
import rclpy
from rclpy.node import Node
import math
import threading

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


def quat_to_yaw(qx, qy, qz, qw):
    """Convertit quaternion en yaw (rad)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PosePicker(Node):
    def __init__(self):
        super().__init__('pose_picker')
        
        self.last_amcl_pose = None
        self.last_odom_pose = None
        
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10
        )
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10
        )
        
        self.get_logger().info(
            'PosePicker pret. Deplace le robot dans Gazebo, '
            'puis appuie sur ENTREE pour capturer la pose.'
        )
        self.get_logger().info(
            'Tape "q" puis ENTREE pour quitter.'
        )
    
    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        yaw = quat_to_yaw(p.orientation.x, p.orientation.y,
                          p.orientation.z, p.orientation.w)
        self.last_amcl_pose = (p.position.x, p.position.y, yaw)
    
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose
        yaw = quat_to_yaw(p.orientation.x, p.orientation.y,
                          p.orientation.z, p.orientation.w)
        self.last_odom_pose = (p.position.x, p.position.y, yaw)
    
    def print_pose(self, label: str = ''):
        if self.last_amcl_pose:
            x, y, yaw = self.last_amcl_pose
            print(f'\n=== POSE CAPTUREE (AMCL/map frame) ===')
            print(f'  {{x: {x:.3f}, y: {y:.3f}, yaw: {yaw:.4f}}}')
            print(f'  yaw degrees: {math.degrees(yaw):.1f}')
        else:
            print('\n[!] Aucune pose AMCL recue. AMCL est-il initialise via RViz ?')
        
        if self.last_odom_pose:
            x, y, yaw = self.last_odom_pose
            print(f'=== POSE ODOM (frame odom, pour reference) ===')
            print(f'  x={x:.3f}, y={y:.3f}, yaw={yaw:.4f}')
        print('')


def input_thread(node):
    counter = 0
    while rclpy.ok():
        cmd = input('>>> Appuie ENTREE pour capturer (ou "q" pour quitter): ')
        if cmd.strip().lower() == 'q':
            rclpy.shutdown()
            break
        counter += 1
        node.print_pose(label=f'capture_{counter}')


def main():
    rclpy.init()
    node = PosePicker()
    t = threading.Thread(target=input_thread, args=(node,), daemon=True)
    t.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""
Relay /model/limo/odometry -> /odom avec correction du child_frame_id.

Gazebo Harmonic prefixe les frame_ids avec le nom du modele (limo/base_link).
Cartographer et Nav2 attendent base_link sans prefixe. On corrige ici.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OdomRelayFixed(Node):
    def __init__(self):
        super().__init__('odom_relay_fixed')
        self.sub = self.create_subscription(
            Odometry, '/model/limo/odometry', self.cb, 10)
        self.pub = self.create_publisher(Odometry, '/odom', 10)

    def cb(self, msg):
        # Reecriture pour enlever le prefixe "limo/"
        if msg.header.frame_id.startswith('limo/'):
            msg.header.frame_id = msg.header.frame_id.replace('limo/', '', 1)
        if msg.child_frame_id.startswith('limo/'):
            msg.child_frame_id = msg.child_frame_id.replace('limo/', '', 1)
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(OdomRelayFixed())


main()

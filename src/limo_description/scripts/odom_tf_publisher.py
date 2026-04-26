import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import tf2_ros
from geometry_msgs.msg import TransformStamped

class OdomTFPublisher(Node):
    def __init__(self):
        super().__init__('odom_tf_publisher')
        self.br = tf2_ros.TransformBroadcaster(self)
        self.sub = self.create_subscription(
            Odometry, '/model/limo/odometry', self.cb, 10)

    def cb(self, msg):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)

def main():
    rclpy.init()
    rclpy.spin(OdomTFPublisher())

main()

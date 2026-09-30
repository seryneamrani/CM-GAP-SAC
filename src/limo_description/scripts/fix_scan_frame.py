import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, MultiEchoLaserScan, LaserEcho


class ScanFrameFixer(Node):
    """
    Rewrites the frame_id of the Gazebo-bridged LaserScan from
    '/model/limo/laser/scan' (frame_id undefined) to 'laser_link'
    (which matches the TF published by robot_state_publisher).

    Also publishes a MultiEchoLaserScan on /horizontal_laser_2d
    for Cartographer compatibility.
    """

    def __init__(self):
        super().__init__('scan_frame_fixer')
        self.sub = self.create_subscription(
            LaserScan, '/model/limo/laser/scan', self.cb, 10)
        self.pub_scan = self.create_publisher(LaserScan, '/scan', 10)
        self.pub_echo = self.create_publisher(
            MultiEchoLaserScan, '/horizontal_laser_2d', 10)

    def cb(self, msg):
        now = self.get_clock().now().to_msg()

        # LaserScan — frame_id doit correspondre au link du LiDAR dans l'URDF
        msg.header.frame_id = 'laser_link'
        msg.header.stamp = now
        self.pub_scan.publish(msg)

        # MultiEchoLaserScan pour Cartographer
        echo_msg = MultiEchoLaserScan()
        echo_msg.header = msg.header  # même frame_id = laser_link
        echo_msg.angle_min = msg.angle_min
        echo_msg.angle_max = msg.angle_max
        echo_msg.angle_increment = msg.angle_increment
        echo_msg.time_increment = msg.time_increment
        echo_msg.scan_time = msg.scan_time
        echo_msg.range_min = msg.range_min
        echo_msg.range_max = msg.range_max
        echo_msg.ranges = [LaserEcho(echoes=[r]) for r in msg.ranges]
        echo_msg.intensities = [LaserEcho(echoes=[i]) for i in msg.intensities]
        self.pub_echo.publish(echo_msg)


def main():
    rclpy.init()
    rclpy.spin(ScanFrameFixer())


main()

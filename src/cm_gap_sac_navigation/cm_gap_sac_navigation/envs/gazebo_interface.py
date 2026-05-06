"""ROS 2 <-> Gazebo Harmonic interface node (v0.2).

Capabilities:
    - Subscribe to /scan        (sensor_msgs/LaserScan)
    - Subscribe to /odom        (nav_msgs/Odometry)
    - Subscribe to /imu         (sensor_msgs/Imu)              <- NEW v0.2
    - Publish    to /cmd_vel    (geometry_msgs/Twist)
    - Call world-control service to reset the simulation
    - Call set_pose service to teleport the robot at episode start

Single responsibility: own the wires to the simulator. The Gym env owns
this node via composition and queries it through clean data-only getters.
No reward, no spaces, no episode logic here.
"""
from __future__ import annotations

from threading import Event, Lock
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from sensor_msgs.msg import Imu, LaserScan

from cm_gap_sac_navigation.utils.geometry import yaw_from_quaternion

try:
    from ros_gz_interfaces.srv import ControlWorld, SetEntityPose  # type: ignore
    from ros_gz_interfaces.msg import Entity, WorldControl  # type: ignore
    HAS_ROS_GZ = True
except ImportError:
    HAS_ROS_GZ = False


class _LatestScan:
    """Thread-safe holder for the most recent LaserScan."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._ranges: Optional[np.ndarray] = None
        self._stamp_s: float = 0.0
        self._range_max: float = 0.0
        self.event = Event()

    def update(self, msg: LaserScan) -> None:
        with self._lock:
            self._ranges = np.asarray(msg.ranges, dtype=np.float32)
            self._stamp_s = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            self._range_max = float(msg.range_max)
        self.event.set()

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._ranges is None else self._ranges.copy()


class _LatestOdom:
    """Thread-safe holder for the most recent Odometry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.stamp_s = 0.0
        self.event = Event()

    def update(self, msg: Odometry) -> None:
        with self._lock:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
            self.v = msg.twist.twist.linear.x
            self.omega = msg.twist.twist.angular.z
            self.stamp_s = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.event.set()

    def snapshot(self) -> tuple[float, float, float, float, float]:
        with self._lock:
            return self.x, self.y, self.yaw, self.v, self.omega


class _LatestImu:                                              # NEW in v0.2
    """Thread-safe holder for the most recent IMU message.

    Stores the raw 6 features [ax, ay, az, wx, wy, wz] in SI units
    (m/s^2 for accelerations, rad/s for angular velocities). Normalization
    is performed by the env, not here, to keep the interface neutral.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._features = np.zeros(6, dtype=np.float32)
        self._stamp_s: float = 0.0
        self.event = Event()

    def update(self, msg: Imu) -> None:
        with self._lock:
            a = msg.linear_acceleration
            w = msg.angular_velocity
            self._features[0] = a.x
            self._features[1] = a.y
            self._features[2] = a.z
            self._features[3] = w.x
            self._features[4] = w.y
            self._features[5] = w.z
            self._stamp_s = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.event.set()

    def get(self) -> np.ndarray:
        with self._lock:
            return self._features.copy()


class GazeboInterface(Node):
    """Node holding all the simulator wires. Owned by the Gym env."""

    def __init__(
        self,
        world_name: str,
        scan_topic: str,
        odom_topic: str,
        imu_topic: str,                                        # NEW in v0.2
        cmd_vel_topic: str,
        reset_service_template: str,
        set_pose_service_template: str,
    ) -> None:
        super().__init__("cm_gap_sac_gazebo_interface")

        self.world_name = world_name

        sensor_qos = QoSProfile(
            depth=5,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Subscriptions ---------------------------------------------------
        self.scan = _LatestScan()
        self.odom = _LatestOdom()
        self.imu  = _LatestImu()                               # NEW in v0.2

        self.create_subscription(
            LaserScan, scan_topic, self.scan.update, sensor_qos,
        )
        self.create_subscription(
            Odometry, odom_topic, self.odom.update, sensor_qos,
        )
        self.create_subscription(
            Imu, imu_topic, self.imu.update, sensor_qos,       # NEW in v0.2
        )

        # Publications ----------------------------------------------------
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # Services --------------------------------------------------------
        if HAS_ROS_GZ:
            reset_srv = reset_service_template.format(world=world_name)
            set_pose_srv = set_pose_service_template.format(world=world_name)
            self._reset_client = self.create_client(ControlWorld, reset_srv)
            self._set_pose_client = self.create_client(SetEntityPose, set_pose_srv)
        else:
            self.get_logger().warn(
                "ros_gz_interfaces not found; reset/teleport will be no-ops. "
                "Install ros-jazzy-ros-gz to enable simulator control."
            )
            self._reset_client = None
            self._set_pose_client = None

        self.get_logger().info(
            f"GazeboInterface up: world={world_name}, "
            f"scan={scan_topic}, odom={odom_topic}, "
            f"imu={imu_topic}, cmd_vel={cmd_vel_topic}"
        )

    # ------------------------------------------------------------------
    # Action publication
    # ------------------------------------------------------------------
    def publish_cmd(self, v: float, omega: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(omega)
        self._cmd_pub.publish(msg)

    def publish_zero_cmd(self) -> None:
        self.publish_cmd(0.0, 0.0)

    # ------------------------------------------------------------------
    # Sensor access (blocking with timeout)
    # ------------------------------------------------------------------
    def wait_for_first_messages(self, timeout_s: float = 5.0) -> bool:
        """Block until first scan, odom, and imu have all been received."""
        ok_scan = self.scan.event.wait(timeout=timeout_s)
        ok_odom = self.odom.event.wait(timeout=timeout_s)
        ok_imu  = self.imu.event.wait(timeout=timeout_s)        # NEW in v0.2
        return ok_scan and ok_odom and ok_imu

    def get_scan(self) -> Optional[np.ndarray]:
        return self.scan.get()

    def get_robot_state(self) -> tuple[float, float, float, float, float]:
        """Returns (x, y, yaw, v, omega)."""
        return self.odom.snapshot()

    def get_imu(self) -> np.ndarray:                            # NEW in v0.2
        """Returns raw IMU [ax, ay, az, wx, wy, wz], shape (6,) float32."""
        return self.imu.get()

    # ------------------------------------------------------------------
    # Simulator control
    # ------------------------------------------------------------------
    def reset_world(self, timeout_s: float = 2.0) -> bool:
        """Reset Gazebo world (time + entity poses).

        Tries the bridged ROS service first; falls back to the `gz` CLI if
        the service is not available (common when ros_gz_bridge is not
        configured to bridge services).
        """
        # Try bridged ROS service first.
        if self._reset_client is not None:
            if self._reset_client.wait_for_service(timeout_sec=timeout_s):
                req = ControlWorld.Request()
                wc = WorldControl()
                wc.reset.all = True
                req.world_control = wc
                future = self._reset_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
                if future.done() and future.result() is not None:
                    return True

        # Fallback: use gz CLI directly. This bypasses the ROS bridge.
        return self._reset_world_gz_cli(timeout_s)

    def _reset_world_gz_cli(self, timeout_s: float = 2.0) -> bool:
        """Reset the world by shelling out to `gz service`.

        Equivalent to:
            gz service -s /world/<world>/control \\
                       --reqtype gz.msgs.WorldControl \\
                       --reptype gz.msgs.Boolean \\
                       --timeout 2000 \\
                       --req 'reset: { all: true }'
        """
        import subprocess
        service = f"/world/{self.world_name}/control"
        try:
            result = subprocess.run(
                ["gz", "service", "-s", service,
                 "--reqtype", "gz.msgs.WorldControl",
                 "--reptype", "gz.msgs.Boolean",
                 "--timeout", str(int(timeout_s * 1000)),
                 "--req", "reset: { all: true }"],
                capture_output=True, text=True, timeout=timeout_s + 1.0,
            )
            if result.returncode == 0 and "data: true" in result.stdout:
                return True
            self.get_logger().warn(
                f"gz service reset returned: stdout={result.stdout!r}, "
                f"stderr={result.stderr!r}"
            )
            return False
        except subprocess.TimeoutExpired:
            self.get_logger().error("gz service reset timed out")
            return False
        except FileNotFoundError:
            self.get_logger().error(
                "`gz` CLI not found in PATH. Source the Gazebo Harmonic env."
            )
            return False

    def set_robot_pose(
        self,
        entity_name: str,
        x: float, y: float, yaw: float,
        timeout_s: float = 2.0,
    ) -> bool:
        """Teleport the robot to a given (x, y, yaw).

        Tries the bridged ROS service first; falls back to the `gz` CLI.
        """
        # Try bridged ROS service first.
        if self._set_pose_client is not None:
            if self._set_pose_client.wait_for_service(timeout_sec=timeout_s):
                req = SetEntityPose.Request()
                req.entity = Entity(name=entity_name, type=Entity.MODEL)
                pose = Pose()
                pose.position.x = float(x)
                pose.position.y = float(y)
                pose.position.z = 0.0
                pose.orientation.z = float(np.sin(yaw / 2.0))
                pose.orientation.w = float(np.cos(yaw / 2.0))
                req.pose = pose
                future = self._set_pose_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
                if future.done() and future.result() is not None:
                    return True

        # Fallback: use gz CLI.
        return self._set_pose_gz_cli(entity_name, x, y, yaw, timeout_s)

    def _set_pose_gz_cli(
        self,
        entity_name: str,
        x: float, y: float, yaw: float,
        timeout_s: float = 2.0,
    ) -> bool:
        """Teleport entity by shelling out to `gz service`.

        Equivalent to:
            gz service -s /world/<world>/set_pose \\
                       --reqtype gz.msgs.Pose \\
                       --reptype gz.msgs.Boolean \\
                       --timeout 2000 \\
                       --req 'name: "<entity>", position: {...}, orientation: {...}'
        """
        import subprocess
        service = f"/world/{self.world_name}/set_pose"
        qz = float(np.sin(yaw / 2.0))
        qw = float(np.cos(yaw / 2.0))
        req = (
            f'name: "{entity_name}", '
            f'position: {{ x: {x}, y: {y}, z: 0.0 }}, '
            f'orientation: {{ x: 0.0, y: 0.0, z: {qz}, w: {qw} }}'
        )
        try:
            result = subprocess.run(
                ["gz", "service", "-s", service,
                 "--reqtype", "gz.msgs.Pose",
                 "--reptype", "gz.msgs.Boolean",
                 "--timeout", str(int(timeout_s * 1000)),
                 "--req", req],
                capture_output=True, text=True, timeout=timeout_s + 1.0,
            )
            if result.returncode == 0 and "data: true" in result.stdout:
                return True
            self.get_logger().warn(
                f"gz service set_pose returned: stdout={result.stdout!r}, "
                f"stderr={result.stderr!r}"
            )
            return False
        except subprocess.TimeoutExpired:
            self.get_logger().error("gz service set_pose timed out")
            return False
        except FileNotFoundError:
            self.get_logger().error("`gz` CLI not found in PATH.")
            return False

"""ROS 2 <-> Gazebo Harmonic interface node (v0.6).

Capabilities:
    - Subscribe to /scan                          (sensor_msgs/LaserScan)
    - Subscribe to /odom                          (nav_msgs/Odometry)   [velocity only]
    - Subscribe to /imu                           (sensor_msgs/Imu)
    - Subscribe to /world/<w>/dynamic_pose/info   (tf2_msgs/TFMessage)  [ground-truth pose]
    - Publish    to /cmd_vel                      (geometry_msgs/Twist)
    - Call world-control service to reset the simulation
    - Call set_pose service to teleport the robot at episode start
    - reset_pedestrians(): parallel-teleport all registered pedestrians to spawn poses

World-frame pose:
    Under accelerated physics, DiffDrive odometry drifts hard from the true pose
    (wheel-slip integration error), so it is NOT used for position. Instead we read
    the simulator's ground-truth pose, bridged from /world/<w>/dynamic_pose/info
    (gz.msgs.Pose_V -> tf2_msgs/TFMessage). This is exact, RTF-proof, and cheap
    (a DDS subscription, not a subprocess). On real hardware this pose is replaced
    by SLAM/AMCL localization. Linear/angular velocities still come from /odom
    (the commanded twist the agent produces).

    Note: GroundTruthTracker (separate node, owned by the env) typically subscribes
    to the same dynamic_pose/info bridge for pedestrian observation. One bridge
    entry therefore serves both nodes simultaneously.

Single responsibility: own the wires to the simulator. The Gym env owns this
node via composition and queries it through clean data-only getters. No reward,
no spaces, no episode logic here.
"""
from __future__ import annotations

import concurrent.futures
from threading import Event, Lock
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from sensor_msgs.msg import Imu, LaserScan
from tf2_msgs.msg import TFMessage

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
    """Thread-safe holder for the most recent Odometry.

    Only the twist (v, omega) is consumed downstream; the integrated position
    drifts under accelerated physics and is intentionally not used as pose.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self.v = 0.0
        self.omega = 0.0
        self.stamp_s = 0.0
        self.event = Event()

    def update(self, msg: Odometry) -> None:
        with self._lock:
            self.v = msg.twist.twist.linear.x
            self.omega = msg.twist.twist.angular.z
            self.stamp_s = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.event.set()

    def twist(self) -> Tuple[float, float]:
        with self._lock:
            return self.v, self.omega


class _LatestRobotPose:
    """Thread-safe holder for the ground-truth robot pose.

    Fed by /world/<world>/dynamic_pose/info bridged to tf2_msgs/TFMessage.

    gz-sim's dynamic_pose/info does NOT populate entity names in the published
    gz.msgs.Pose_V (it uses integer IDs for efficiency), so child_frame_id
    arrives as an empty string after bridging. We identify the robot by its
    chassis height instead:

        LIMO base_link  z ≈  0.15 m  (world frame, set by spawn z: 0.15)
        Pedestrians     z ≈  0.0  m  (ground level)
        Wheel links     z ≈ -0.105 m (relative to base_link, not world)
        Static objects  z =  0.0  m

    The window [robot_z_min, robot_z_max] = [0.05, 0.35] uniquely selects the
    robot body and tolerates chassis tilt/bounce (e.g. during hard braking at
    1 m/s). Pedestrians stay at z < 0.03; wheel joints at z ≈ −0.105.
    """

    def __init__(
        self,
        robot_name: str = "limo",       # kept for API compatibility, not used
        robot_z_min: float = 0.05,
        robot_z_max: float = 0.35,
    ) -> None:
        self._lock = Lock()
        self._z_min = robot_z_min
        self._z_max = robot_z_max
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.stamp_s = 0.0
        self.event = Event()

    def update(self, msg: TFMessage) -> None:
        for tf in msg.transforms:
            t = tf.transform.translation
            if self._z_min <= t.z <= self._z_max:
                q = tf.transform.rotation
                with self._lock:
                    self.x = t.x
                    self.y = t.y
                    self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
                    self.stamp_s = (tf.header.stamp.sec
                                    + 1e-9 * tf.header.stamp.nanosec)
                self.event.set()
                return

    def snapshot(self) -> Tuple[float, float, float]:
        with self._lock:
            return self.x, self.y, self.yaw


class _LatestImu:
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
        imu_topic: str,
        cmd_vel_topic: str,
        # These two are kept for backward compatibility; the CLI path
        # constructs the service strings from world_name directly.
        reset_service_template: Optional[str] = None,
        set_pose_service_template: Optional[str] = None,
        robot_name: str = "limo",
        pose_topic: Optional[str] = None,
        pedestrian_poses: Optional[Dict[str, Tuple]] = None,
    ) -> None:
        super().__init__("cm_gap_sac_gazebo_interface")

        self.world_name = world_name
        self.robot_name = robot_name

        # Pedestrian spawn poses used by reset_pedestrians().
        # Dict {entity_name: (x, y, yaw)}.
        self._pedestrian_poses: Dict[str, Tuple] = pedestrian_poses or {}

        if pose_topic is None:
            pose_topic = f"/world/{world_name}/dynamic_pose/info"

        sensor_qos = QoSProfile(
            depth=5,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        # Bridged TF topic: subscribe RELIABLE to match the ros_gz_bridge
        # default. Switch to sensor_qos if no pose messages arrive.
        pose_qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Subscriptions ---------------------------------------------------
        self.scan = _LatestScan()
        self.odom = _LatestOdom()
        self.imu = _LatestImu()
        self.robot_pose = _LatestRobotPose(robot_name)

        self.create_subscription(LaserScan, scan_topic, self.scan.update, sensor_qos)
        self.create_subscription(Odometry, odom_topic, self.odom.update, sensor_qos)
        self.create_subscription(Imu, imu_topic, self.imu.update, sensor_qos)
        self.create_subscription(TFMessage, pose_topic, self.robot_pose.update, pose_qos)

        # Publications ----------------------------------------------------
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # Services --------------------------------------------------------
        reset_srv = (reset_service_template or "/world/{world}/control").format(
            world=world_name
        )
        set_pose_srv = (set_pose_service_template or "/world/{world}/set_pose").format(
            world=world_name
        )
        if HAS_ROS_GZ:
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
            f"imu={imu_topic}, cmd_vel={cmd_vel_topic}, pose={pose_topic}, "
            f"pedestrians={list(self._pedestrian_poses)}"
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
    def wait_for_first_messages(self, timeout_s: float = 10.0) -> bool:
        """Block until scan, odom, imu, and ground-truth pose are all ready."""
        ok_scan = self.scan.event.wait(timeout=timeout_s)
        ok_odom = self.odom.event.wait(timeout=timeout_s)
        ok_imu = self.imu.event.wait(timeout=timeout_s)
        ok_pose = self.robot_pose.event.wait(timeout=timeout_s)
        if not (ok_scan and ok_odom and ok_imu and ok_pose):
            self.get_logger().error(
                f"Sensor timeout: scan={ok_scan} odom={ok_odom} "
                f"imu={ok_imu} pose={ok_pose}. "
                f"If only pose failed: add /world/{self.world_name}/dynamic_pose/info "
                f"to ros_gz_bridge (gz.msgs.Pose_V -> tf2_msgs/TFMessage) and "
                f"confirm child_frame_id with: "
                f"ros2 topic echo /world/{self.world_name}/dynamic_pose/info --once"
            )
            return False
        return True

    def get_scan(self) -> Optional[np.ndarray]:
        return self.scan.get()

    def get_robot_state(self) -> Tuple[float, float, float, float, float]:
        """Returns (x, y, yaw, v, omega) in the WORLD frame.

        Position/yaw are the simulator ground truth (drift-free, RTF-proof).
        v and omega are the odom twist (the commanded body velocity).
        """
        gx, gy, gyaw = self.robot_pose.snapshot()
        v, omega = self.odom.twist()
        return gx, gy, gyaw, v, omega

    def get_imu(self) -> np.ndarray:
        """Returns raw IMU [ax, ay, az, wx, wy, wz], shape (6,) float32."""
        return self.imu.get()

    # ------------------------------------------------------------------
    # Simulator control
    # ------------------------------------------------------------------
    def reset_world(self, timeout_s: float = 2.0) -> bool:
        """Reset Gazebo world (time + entity poses).

        Tries the bridged ROS service first; falls back to the gz CLI if the
        service is not available (common when ros_gz_bridge is not configured
        to bridge services).
        """
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

        return self._reset_world_gz_cli(timeout_s)

    def _reset_world_gz_cli(self, timeout_s: float = 2.0) -> bool:
        """Reset the world by shelling out to `gz service`."""
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

    def reset_pedestrians(self, timeout_s: float = 0.5) -> int:
        """Teleport all registered pedestrians to their spawn poses.

        Fires up to 4 teleports in parallel so the total wall-clock cost stays
        below ~1.5 s for the typical 7-pedestrian hospital world, regardless of
        the timeout_s budget passed by the env.

        Returns the number of successfully teleported pedestrians.
        """
        if not self._pedestrian_poses:
            return 0

        per_ped_timeout = max(timeout_s * 3, 1.0)
        n_workers = min(4, len(self._pedestrian_poses))

        def _teleport(item: Tuple) -> bool:
            name, pose = item
            x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
            return self._set_pose_gz_cli(name, x, y, yaw, timeout_s=per_ped_timeout)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_teleport, self._pedestrian_poses.items()))

        return sum(results)

    def set_robot_pose(self, entity_name, x, y, yaw, timeout_s=5.0, max_retries=2):
        """Teleport the robot and wait for the ground-truth pose to confirm it.

        The wheels are stopped first so no residual command perturbs the spawn.
        We then poll the bridged GT pose. If it fails to confirm within 2 s,
        we force-write the commanded pose directly — the gz service already
        confirmed the teleport succeeded, the bridge is just slow to reflect it.
        This prevents the first step of the new episode using a stale pose.
        """
        import time

        self.publish_zero_cmd()
        time.sleep(0.1)

        success = self._set_pose_gz_cli(entity_name, x, y, yaw, timeout_s)
        if not success:
            return False

        # Short settle window before polling so the sim has time to apply
        # the teleport before the first dynamic_pose/info arrives.
        time.sleep(0.3)

        t0 = time.time()
        while time.time() - t0 < 2.0:
            gx, gy, _ = self.robot_pose.snapshot()
            if abs(gx - x) < 0.3 and abs(gy - y) < 0.3:
                return True
            time.sleep(0.02)

        # Bridge did not update in time. The set_pose succeeded (gz service
        # returned data: true), so we can safely force the pose so the first
        # observation of the new episode is correct.
        #self.get_logger().warn(
            #f"set_robot_pose: GT bridge did not confirm ({x:.2f}, {y:.2f}) "
            #f"within 2 s; forcing pose to avoid stale first observation."
        #)
        with self.robot_pose._lock:
            self.robot_pose.x = float(x)
            self.robot_pose.y = float(y)
            self.robot_pose.yaw = float(yaw)
        return True

    def _set_pose_gz_cli(
        self,
        entity_name: str,
        x: float, y: float, yaw: float,
        timeout_s: float = 2.0,
    ) -> bool:
        """Teleport entity by shelling out to `gz service`."""
        import subprocess
        service = f"/world/{self.world_name}/set_pose"
        qz = float(np.sin(yaw / 2.0))
        qw = float(np.cos(yaw / 2.0))
        req = (
            f'name: "{entity_name}", '
            f'position: {{ x: {x}, y: {y}, z: 0.15 }}, '
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
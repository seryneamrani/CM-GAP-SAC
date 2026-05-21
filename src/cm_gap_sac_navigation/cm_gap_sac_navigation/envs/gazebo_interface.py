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
from geometry_msgs.msg import Pose, Twist
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
#from tf2_msgs.msg import TFMessage
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

    def __init__(self) -> None:                                       # infeasible >N consecutive steps
 
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
        #self.robot_pose = _LatestRobotPose()

    
        self.create_subscription(
            LaserScan, scan_topic, self.scan.update, sensor_qos,
        )
        self.create_subscription(
            Odometry, odom_topic, self.odom.update, sensor_qos,
        )
        self.create_subscription(
            Imu, imu_topic, self.imu.update, sensor_qos,       # NEW in v0.2
        )
        #self.create_subscription(
            #TFMessage, f"/world/{world_name}/pose/info",
            #self.robot_pose.update, sensor_qos,
        #)
       

        
        
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
        # Background thread polling gz topic for ground-truth pose.
        # Far cheaper than spawning a subprocess every step.
        import threading
        self._gt_x = 0.0
        self._gt_y = 0.0
        self._gt_yaw = 0.0
        self._gt_lock = Lock()
        self._gt_seen = False
        self._gt_thread = threading.Thread(
            target=self._gt_pose_poller, daemon=True,
        )
        self._gt_thread.start()

        self.get_logger().info(
            f"GazeboInterface up: world={world_name}, "
            f"scan={scan_topic}, odom={odom_topic}, "
            f"imu={imu_topic}, cmd_vel={cmd_vel_topic}"
        )

        # World-frame offset tracking for odometry.
        # DiffDrive integrates wheel velocities from spawn, ignoring teleports.
        # We track the teleport target and the odom reading at that moment,
        # then convert odom deltas into world-frame deltas.
        self._teleport_target_x = 0.0
        self._teleport_target_y = 0.0
        self._teleport_target_yaw = 0.0
        self._odom_at_teleport_x = 0.0
        self._odom_at_teleport_y = 0.0
        self._odom_at_teleport_yaw = 0.0


    def _gt_pose_poller(self):
        """Background thread: continuously parse `gz topic` output."""
        import subprocess, re
        try:
            proc = subprocess.Popen(
                ["gz", "topic", "-e", "-t",
                 f"/world/{self.world_name}/pose/info"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except Exception:
            return

        buf = []
        for line in proc.stdout:
            buf.append(line)
            if line.strip() == "---" or len(buf) > 500:
                block = "".join(buf)
                buf = []
                m = re.search(
                    r'name:\s*"limo"\s*\n.*?position\s*\{([^}]+)\}\s*'
                    r'orientation\s*\{([^}]+)\}',
                    block, re.DOTALL
                )
                if m:
                    try:
                        pos = m.group(1)
                        ori = m.group(2)
                        px = float(re.search(r'x:\s*([0-9e.+-]+)', pos).group(1))
                        py = float(re.search(r'y:\s*([0-9e.+-]+)', pos).group(1))
                        qx = float(re.search(r'x:\s*([0-9e.+-]+)', ori).group(1))
                        qy = float(re.search(r'y:\s*([0-9e.+-]+)', ori).group(1))
                        qz = float(re.search(r'z:\s*([0-9e.+-]+)', ori).group(1))
                        qw = float(re.search(r'w:\s*([0-9e.+-]+)', ori).group(1))
                        yaw = yaw_from_quaternion(qx, qy, qz, qw)
                        with self._gt_lock:
                            self._gt_x = px
                            self._gt_y = py
                            self._gt_yaw = yaw
                            self._gt_seen = True
                    except Exception:
                        pass
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
        """Block until scan, odom, imu, and GT pose lock are all ready."""
        import time
        ok_scan = self.scan.event.wait(timeout=timeout_s)
        ok_odom = self.odom.event.wait(timeout=timeout_s)
        ok_imu  = self.imu.event.wait(timeout=timeout_s)
        if not (ok_scan and ok_odom and ok_imu):
            self.get_logger().error(
                f"Sensor timeout: scan={ok_scan} odom={ok_odom} imu={ok_imu}"
            )
            return False

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self._gt_lock:
                if self._gt_seen:
                    self.get_logger().info(
                        f"GT pose locked at ({self._gt_x:.2f}, {self._gt_y:.2f}, "
                        f"yaw={self._gt_yaw:.2f})"
                    )
                    return True
            time.sleep(0.1)

        self.get_logger().error(
            f"GT pose poller did not lock within {timeout_s}s. "
            f"Check that `gz topic -e -t /world/{self.world_name}/pose/info` "
            f"emits a model named 'limo'."
        )
        return False

    def get_scan(self) -> Optional[np.ndarray]:
        return self.scan.get()

    def get_robot_state(self) -> tuple[float, float, float, float, float]:
        """Returns (x, y, yaw, v, omega) in WORLD frame.

        Position/yaw from the gz subprocess poller (identifies the robot by
        `name: "limo"`, no orientation heuristic). Linear/angular velocities
        from /odom because DiffDrive velocities remain correct after teleport
        (only the integrated position drifts).
        """
        with self._gt_lock:
            gx, gy, gyaw, seen = self._gt_x, self._gt_y, self._gt_yaw, self._gt_seen
        ox, oy, oyaw, v, omega = self.odom.snapshot()
        if seen:
            return gx, gy, gyaw, v, omega
        return ox, oy, oyaw, v, omega

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

    def set_robot_pose(self, entity_name, x, y, yaw, timeout_s=5.0, max_retries=2):
        import time
        success = self._set_pose_gz_cli(entity_name, x, y, yaw, timeout_s)
        print(f"[set_pose] target=({x:.2f},{y:.2f},{yaw:.2f}) success={success}", flush=True)
        if not success:
            return False

        # Wait for the GT poller to reflect the new pose (else first obs is stale)
        t0 = time.time()
        while time.time() - t0 < 1.0:
            with self._gt_lock:
                if (self._gt_seen
                    and abs(self._gt_x - x) < 0.3
                    and abs(self._gt_y - y) < 0.3):
                    # Update odom offset tracking using the now-current odom snapshot
                    ox, oy, oyaw, _, _ = self.odom.snapshot()
                    self._odom_at_teleport_x = ox
                    self._odom_at_teleport_y = oy
                    self._odom_at_teleport_yaw = oyaw
                    self._teleport_target_x = float(x)
                    self._teleport_target_y = float(y)
                    self._teleport_target_yaw = float(yaw)
                    print(f"[set_pose] GT confirmed at ({self._gt_x:.2f}, {self._gt_y:.2f})", flush=True)
                    return True
            time.sleep(0.05)

        self.get_logger().warn(
            f"set_robot_pose: GT poller did not confirm teleport to "
            f"({x:.2f}, {y:.2f}) within 1s; proceeding anyway"
        )
        return True   # set_pose itself succeeded, just stale poller
    
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

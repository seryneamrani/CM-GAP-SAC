#!/usr/bin/env python3
"""
obstacle_projector.py — A2 projection layer

Projette les detections trackees (bbox 2D + ID) vers le monde 3D via la depth camera.

Subscribe :
  /tracked_obstacles      (vision_msgs/Detection2DArray)
  /camera/depth/image_raw (sensor_msgs/Image, depth en metres)
  /camera/camera_info     (sensor_msgs/CameraInfo, intrinseques fx/fy/cx/cy)

Publish :
  /dynamic_obstacles          (geometry_msgs/PoseArray)        — visualisation, frame=map
  /dynamic_obstacles_cloud    (sensor_msgs/PointCloud2)        — Nav2 obstacle_layer, frame=camera optique
  /dynamic_obstacles_markers  (visualization_msgs/MarkerArray) — RViz, frame=map

Strategie depth-sampling : mediane sur une fenetre 5x5 au centre de la bbox
(robuste aux pixels NaN/Inf et aux artefacts depth).

Run :
  python3 obstacle_projector.py
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header

from cv_bridge import CvBridge
import tf2_ros
from tf2_geometry_msgs import do_transform_pose


class ObstacleProjector(Node):
    def __init__(self):
        super().__init__('obstacle_projector')

        # Parametres
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('fallback_frame', 'odom')
        self.declare_parameter('depth_sample_size', 5)
        self.declare_parameter('min_range', 0.3)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('cloud_radius', 0.30)   # rayon disque (m)
        self.declare_parameter('cloud_points', 12)     # points par obstacle dans le cloud
        self.declare_parameter('cloud_height', 1.5)    # hauteur (z) du disque

        self.target_frame = self.get_parameter('target_frame').value
        self.fallback_frame = self.get_parameter('fallback_frame').value
        self.n_sample = int(self.get_parameter('depth_sample_size').value)
        self.min_range = self.get_parameter('min_range').value
        self.max_range = self.get_parameter('max_range').value
        self.cloud_radius = self.get_parameter('cloud_radius').value
        self.cloud_points = int(self.get_parameter('cloud_points').value)
        self.cloud_height = self.get_parameter('cloud_height').value

        # Etat
        self.bridge = CvBridge()
        self.K = None                # 3x3 intrinsics
        self.depth_image = None      # np.float32, HxW, metres
        self.depth_frame = None      # ex: limo/base_link/depth_camera_rgbd

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS RELIABLE pour matcher ros_gz_bridge
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(CameraInfo, '/camera/camera_info', self.cb_info, qos)
        self.create_subscription(Image, '/camera/depth/image_raw', self.cb_depth, qos)
        self.create_subscription(Detection2DArray, '/tracked_obstacles', self.cb_tracks, 10)

        self.pub_poses = self.create_publisher(PoseArray, '/dynamic_obstacles', 10)
        self.pub_cloud = self.create_publisher(PointCloud2, '/dynamic_obstacles_cloud', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/dynamic_obstacles_markers', 10)

        # Stats
        self.n_published = 0
        self.create_timer(5.0, self._log_stats)

        self.get_logger().info('ObstacleProjector pret — attente de CameraInfo et depth')

    # -------- Callbacks ----------------------------------------------------

    def cb_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(
                f'Intrinseques recus : fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} '
                f'cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}'
            )

    def cb_depth(self, msg: Image):
        self.depth_frame = msg.header.frame_id
        try:
            if msg.encoding == '32FC1':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            elif msg.encoding == '16UC1':
                # Convention millimetres → metres
                img = self.bridge.imgmsg_to_cv2(msg, '16UC1').astype(np.float32) / 1000.0
                self.depth_image = img
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg).astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f'Depth conversion echec : {e}')
            self.depth_image = None

    def cb_tracks(self, msg: Detection2DArray):
        if self.K is None or self.depth_image is None or self.depth_frame is None:
            return
        if len(msg.detections) == 0:
            # On publie quand meme des messages vides pour clearing rapide
            self._publish_empty()
            return

        # TF camera_optical → target
        target = self.target_frame
        tf_to_target = self._lookup_tf(target)
        if tf_to_target is None:
            target = self.fallback_frame
            tf_to_target = self._lookup_tf(target)
            if tf_to_target is None:
                return  # logged dans _lookup_tf

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        h, w = self.depth_image.shape[:2]

        now_stamp = self.get_clock().now().to_msg()

        pose_array = PoseArray()
        pose_array.header.stamp = now_stamp
        pose_array.header.frame_id = target

        marker_array = MarkerArray()
        cloud_points_list = []  # liste de (x, y, z) en frame camera_optical
        marker_id = 0

        for det in msg.detections:
            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)

            # Echantillonnage depth robuste autour de (u, v)
            n = self.n_sample
            u_int = int(np.clip(u, n, w - n - 1))
            v_int = int(np.clip(v, n, h - n - 1))
            patch = self.depth_image[v_int - n:v_int + n + 1, u_int - n:u_int + n + 1]

            valid = patch[
                (patch > self.min_range)
                & (patch < self.max_range)
                & np.isfinite(patch)
            ]
            if valid.size < 3:
                continue

            z = float(np.median(valid))

            # Back-projection en frame camera_optical (REP-103: x droite, y bas, z avant)
            X = (u - cx) * z / fx
            Y = (v - cy) * z / fy
            Z = z

            # --- 1) Cloud (frame camera_optical, sans transform) ---
            cloud_points_list.extend(
                self._gen_disk_points(X, Y, Z, self.cloud_radius,
                                       self.cloud_points, self.cloud_height)
            )

            # --- 2) Pose dans target frame (pour viz) ---
            p_cam = Pose()
            p_cam.position.x = X
            p_cam.position.y = Y
            p_cam.position.z = Z
            p_cam.orientation.w = 1.0

            try:
                p_target = do_transform_pose(p_cam, tf_to_target)
            except Exception:
                continue

            pose_array.poses.append(p_target)

            # --- 3) Marker RViz ---
            mk = Marker()
            mk.header = pose_array.header
            mk.ns = 'dynamic_obstacles'
            mk.id = marker_id
            marker_id += 1
            mk.type = Marker.CYLINDER
            mk.action = Marker.ADD
            mk.pose = p_target
            mk.scale.x = self.cloud_radius * 2.0
            mk.scale.y = self.cloud_radius * 2.0
            mk.scale.z = self.cloud_height
            mk.color.r = 1.0
            mk.color.g = 0.4
            mk.color.b = 0.0
            mk.color.a = 0.7
            mk.lifetime.sec = 1
            marker_array.markers.append(mk)

        # Publish PoseArray + Markers
        self.pub_poses.publish(pose_array)
        self.pub_markers.publish(marker_array)

        # Publish PointCloud2 dans le frame camera_optical
        # → permet a Nav2 de raytrace correctement depuis l'origine camera
        cloud_header = Header()
        cloud_header.stamp = now_stamp
        cloud_header.frame_id = self.depth_frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = point_cloud2.create_cloud(cloud_header, fields, cloud_points_list)
        self.pub_cloud.publish(cloud)

        self.n_published += len(pose_array.poses)

    # -------- Helpers ------------------------------------------------------

    def _gen_disk_points(self, x_center, y_center, z_center, radius, n_points, height):
        """Genere n_points autour d'un disque vertical centre sur (x, y, z) en frame camera optique.
        En frame camera optique : Y est vers le bas, donc 'hauteur' s'etale sur Y negatif."""
        pts = []
        # Cercle autour de l'axe vertical (Y dans optical = vers le bas du monde)
        for k in range(n_points):
            ang = 2.0 * math.pi * k / n_points
            dx = radius * math.cos(ang)
            dz = radius * math.sin(ang)
            # 3 niveaux verticaux (haut, milieu, bas) pour epaissir le cylindre
            for dy in (-height / 2.0, 0.0, height / 2.0):
                pts.append((x_center + dx, y_center + dy, z_center + dz))
        return pts

    def _lookup_tf(self, target):
        try:
            return self.tf_buffer.lookup_transform(
                target, self.depth_frame, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF {self.depth_frame} -> {target} indispo : {e}',
                throttle_duration_sec=2.0,
            )
            return None

    def _publish_empty(self):
        now = self.get_clock().now().to_msg()
        target = self.target_frame
        # PoseArray vide
        pa = PoseArray()
        pa.header.stamp = now
        pa.header.frame_id = target
        self.pub_poses.publish(pa)
        # Cloud vide (camera frame)
        if self.depth_frame is not None:
            cloud_header = Header(stamp=now, frame_id=self.depth_frame)
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            self.pub_cloud.publish(point_cloud2.create_cloud(cloud_header, fields, []))
        # Markers : on emet un marker DELETEALL pour nettoyer RViz
        ma = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self.pub_markers.publish(ma)

    def _log_stats(self):
        self.get_logger().info(
            f'Projector : {self.n_published} obstacles projetes (cumul 5s)'
        )
        self.n_published = 0


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleProjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

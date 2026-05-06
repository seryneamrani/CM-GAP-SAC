#!/usr/bin/env python3
"""
obstacle_projector.py -- A2 projection layer (V2)

Projette les detections trackees (bbox 2D + ID) vers le monde 3D via la depth camera.

Subscribe :
  /tracked_obstacles      (vision_msgs/Detection2DArray)
  /camera/depth/image_raw (sensor_msgs/Image, depth en metres)
  /camera/camera_info     (sensor_msgs/CameraInfo, intrinseques fx/fy/cx/cy)

Publish :
  /dynamic_obstacles          (geometry_msgs/PoseArray)        -- viz, frame=map
  /dynamic_obstacles_cloud    (sensor_msgs/PointCloud2)        -- Nav2 obstacle_layer
  /dynamic_obstacles_markers  (visualization_msgs/MarkerArray) -- RViz, frame=map
  /tracked_obstacles_3d       (vision_msgs/Detection3DArray)   -- AVEC track ID, frame=map
                                                                  consomme par trajectory_predictor

NOUVEAUTE V2 : preservation du track ID DeepSORT vers le monde 3D
=> permet le buffer glissant par track pour la prediction de trajectoire (Social-LSTM Lite).

Strategie depth-sampling : mediane sur une fenetre 5x5 au centre de la bbox.
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D, Detection3DArray,
    ObjectHypothesisWithPose,
)
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
        self.declare_parameter('cloud_radius', 0.30)
        self.declare_parameter('cloud_points', 12)
        self.declare_parameter('cloud_height', 1.5)

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
        self.K = None
        self.depth_image = None
        self.depth_frame = None

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        # NOUVEAU : Detection3DArray avec track IDs preserves
        self.pub_dets3d = self.create_publisher(Detection3DArray, '/tracked_obstacles_3d', 10)

        # Stats
        self.n_published = 0
        self.create_timer(5.0, self._log_stats)

        self.get_logger().info('ObstacleProjector V2 pret -- attente CameraInfo + depth')

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
            self._publish_empty()
            return

        target = self.target_frame
        tf_to_target = self._lookup_tf(target)
        if tf_to_target is None:
            target = self.fallback_frame
            tf_to_target = self._lookup_tf(target)
            if tf_to_target is None:
                return

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        h, w = self.depth_image.shape[:2]

        now_stamp = self.get_clock().now().to_msg()

        pose_array = PoseArray()
        pose_array.header.stamp = now_stamp
        pose_array.header.frame_id = target

        det3d_array = Detection3DArray()
        det3d_array.header.stamp = now_stamp
        det3d_array.header.frame_id = target

        marker_array = MarkerArray()
        cloud_points_list = []
        marker_id = 0

        for det in msg.detections:
            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)

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

            X = (u - cx) * z / fx
            Y = (v - cy) * z / fy
            Z = z

            # 1) Cloud (frame camera_optical)
            cloud_points_list.extend(
                self._gen_disk_points(X, Y, Z, self.cloud_radius,
                                       self.cloud_points, self.cloud_height)
            )

            # 2) Pose en camera puis transform vers target
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

            # 3) Marker RViz
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

            # 4) NOUVEAU : Detection3D avec track ID preserve
            det3d = Detection3D()
            det3d.header = det3d_array.header
            det3d.id = det.id  # track ID (str) propage depuis perception_node
            det3d.bbox.center = p_target  # BoundingBox3D.center est un Pose
            det3d.bbox.size.x = self.cloud_radius * 2.0
            det3d.bbox.size.y = self.cloud_radius * 2.0
            det3d.bbox.size.z = self.cloud_height
            # Preserver classe + score depuis l'hypothese 2D
            if len(det.results) > 0:
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = det.results[0].hypothesis.class_id
                hyp.hypothesis.score = det.results[0].hypothesis.score
                det3d.results.append(hyp)
            det3d_array.detections.append(det3d)

        # Publish
        self.pub_poses.publish(pose_array)
        self.pub_markers.publish(marker_array)
        self.pub_dets3d.publish(det3d_array)

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
        pts = []
        for k in range(n_points):
            ang = 2.0 * math.pi * k / n_points
            dx = radius * math.cos(ang)
            dz = radius * math.sin(ang)
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
        # Detection3DArray vide
        d3 = Detection3DArray()
        d3.header.stamp = now
        d3.header.frame_id = target
        self.pub_dets3d.publish(d3)
        # Cloud vide
        if self.depth_frame is not None:
            cloud_header = Header(stamp=now, frame_id=self.depth_frame)
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            self.pub_cloud.publish(point_cloud2.create_cloud(cloud_header, fields, []))
        # Markers : DELETEALL pour clean RViz
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

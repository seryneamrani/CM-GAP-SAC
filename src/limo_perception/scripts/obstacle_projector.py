#!/usr/bin/env python3
"""
obstacle_projector_v2.py -- Projection 3D + injection cone d'occupation futur.

V2 difference :
    En plus du nuage instantane (/dynamic_obstacles_cloud), on ajoute les
    points de prediction LSTM (depuis /predicted_trajectories_json) avec
    cout decroissant, projetes dans le frame camera optique pour beneficier
    du raytracing Nav2.

Subscribe :
    /tracked_obstacles                 (vision_msgs/Detection2DArray)
    /camera/depth/image_raw            (sensor_msgs/Image)
    /camera/camera_info                (sensor_msgs/CameraInfo)
    /predicted_trajectories_json       (std_msgs/String JSON)

Publish :
    /dynamic_obstacles                 (geometry_msgs/PoseArray, frame map)
    /dynamic_obstacles_cloud           (sensor_msgs/PointCloud2,
                                        frame depth_camera optical)
    /dynamic_obstacles_markers         (visualization_msgs/MarkerArray)

Version simplifiee (focus sur la nouveaute v2). Le code complet de la
projection mediane 5x5 est dans obstacle_projector.py v1 deploye au workspace.
"""
import json
import math
import struct
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseArray, Pose, TransformStamped, PointStamped
from std_msgs.msg import String, Header
from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point


# Parametres geometriques
OBSTACLE_RADIUS = 0.30
OBSTACLE_HEIGHT = 1.50
N_ANGULAR = 12
N_VERTICAL = 3
MIN_RANGE = 0.30
MAX_RANGE = 8.0

# Couts (LETHAL=254, INSCRIBED=253, ...)
COST_NOW = 254
COST_FUTURE_START = 200
COST_FUTURE_END = 100

DEPTH_FRAME = "limo/base_link/depth_camera_rgbd"


def make_pointcloud2(points, frame_id: str, stamp) -> PointCloud2:
    """Construit un PointCloud2 XYZ float32 a partir d'une liste (x,y,z)."""
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = b"".join(struct.pack("fff", float(x), float(y), float(z))
                        for x, y, z in points)
    return msg


def gen_cylinder_points(xc, yc, zc, radius=OBSTACLE_RADIUS,
                        height=OBSTACLE_HEIGHT, n_ang=N_ANGULAR, n_v=N_VERTICAL):
    """Genere n_ang*n_v points sur un cylindre centre en (xc, yc, zc).
    Convention : axe vertical = Y dans le frame camera optique (Y vers le bas)."""
    pts = []
    for k in range(n_ang):
        ang = 2.0 * math.pi * k / n_ang
        dx = radius * math.cos(ang)
        dz = radius * math.sin(ang)
        for j in range(n_v):
            # j=0 -> sol, j=n_v-1 -> tete
            frac = j / max(1, n_v - 1)
            dy = -(height * frac - height / 2.0)  # camera Y descend
            pts.append((xc + dx, yc + dy, zc + dz))
    return pts


class ObstacleProjectorV2(Node):
    def __init__(self):
        super().__init__("obstacle_projector_v2")

        self.declare_parameter("min_range", MIN_RANGE)
        self.declare_parameter("max_range", MAX_RANGE)
        self.declare_parameter("target_frame", "map")
        self.target_frame = self.get_parameter("target_frame").value

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Etat
        self.K: Optional[np.ndarray] = None
        self.last_depth: Optional[np.ndarray] = None
        self.last_depth_stamp = None
        self.last_predictions = []   # liste de {id, future:[(x,y),...], state}

        # Subs
        self.create_subscription(CameraInfo, "/camera/camera_info",
                                 self._cb_caminfo, 10)
        self.create_subscription(Image, "/camera/depth/image_raw",
                                 self._cb_depth, 10)
        self.create_subscription(Detection2DArray, "/tracked_obstacles",
                                 self._cb_tracks, 10)
        self.create_subscription(String, "/predicted_trajectories_json",
                                 self._cb_predictions, 10)

        # Pubs
        self.pub_cloud = self.create_publisher(
            PointCloud2, "/dynamic_obstacles_cloud", 10)
        self.pub_poses = self.create_publisher(
            PoseArray, "/dynamic_obstacles", 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, "/dynamic_obstacles_markers", 10)

        self.get_logger().info("ObstacleProjector v2 pret (avec cone futur).")

    def _cb_caminfo(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f"K charge : fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} "
                f"cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}"
            )

    def _cb_depth(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0  # mm -> m
        self.last_depth = depth
        self.last_depth_stamp = msg.header.stamp

    def _cb_predictions(self, msg: String):
        try:
            self.last_predictions = json.loads(msg.data)
        except Exception:
            self.last_predictions = []

    def _cb_tracks(self, msg: Detection2DArray):
        if self.K is None or self.last_depth is None:
            return

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        h, w = self.last_depth.shape[:2]

        cylinder_points = []     # frame camera optique
        pose_array = PoseArray()
        pose_array.header.frame_id = self.target_frame
        pose_array.header.stamp = msg.header.stamp
        markers = MarkerArray()
        markers.markers.append(Marker(action=Marker.DELETEALL))

        for i, det in enumerate(msg.detections):
            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)

            # Echantillonnage 5x5
            n = 5
            ui = int(np.clip(u, n, w - n - 1))
            vi = int(np.clip(v, n, h - n - 1))
            patch = self.last_depth[vi - n:vi + n + 1, ui - n:ui + n + 1]
            valid = patch[(patch > MIN_RANGE) & (patch < MAX_RANGE) &
                          np.isfinite(patch)]
            if valid.size < 3:
                continue
            z = float(np.median(valid))

            X = (u - cx) * z / fx
            Y = (v - cy) * z / fy
            Z = z

            cylinder_points.extend(
                gen_cylinder_points(X, Y, Z))

            # Transforme aussi vers map pour /dynamic_obstacles
            try:
                point_cam = PointStamped()
                point_cam.header.frame_id = DEPTH_FRAME
                point_cam.header.stamp = self.last_depth_stamp
                point_cam.point.x = X
                point_cam.point.y = Y
                point_cam.point.z = Z
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame, DEPTH_FRAME,
                    rclpy.time.Time(), Duration(seconds=0.2))
                point_map = do_transform_point(point_cam, tf)
                p = Pose()
                p.position.x = point_map.point.x
                p.position.y = point_map.point.y
                p.position.z = 0.0
                p.orientation.w = 1.0
                pose_array.poses.append(p)

                # Marker cylindre 3D
                cyl = Marker()
                cyl.header = pose_array.header
                cyl.ns = f"obs_{det.id}"
                cyl.id = int(det.id) if det.id and det.id.isdigit() else i
                cyl.type = Marker.CYLINDER
                cyl.action = Marker.ADD
                cyl.pose.position.x = point_map.point.x
                cyl.pose.position.y = point_map.point.y
                cyl.pose.position.z = OBSTACLE_HEIGHT / 2.0
                cyl.pose.orientation.w = 1.0
                cyl.scale.x = OBSTACLE_RADIUS * 2
                cyl.scale.y = OBSTACLE_RADIUS * 2
                cyl.scale.z = OBSTACLE_HEIGHT
                cyl.color.a = 0.4
                cyl.color.r = 1.0; cyl.color.g = 0.5; cyl.color.b = 0.0
                markers.markers.append(cyl)
            except Exception as e:
                self.get_logger().warn(
                    f"TF echoue depth->map : {e}", throttle_duration_sec=2.0)

        # === V2 : ajout cone d'occupation futur ===
        # Pour chaque prediction : on transforme les points (x,y) du frame map
        # vers le frame camera optique, puis on les ajoute au cloud avec une
        # densite reduite (pour cout moindre).
        if self.last_predictions:
            try:
                tf_map_to_cam = self.tf_buffer.lookup_transform(
                    DEPTH_FRAME, self.target_frame,
                    rclpy.time.Time(), Duration(seconds=0.2))
                for pred in self.last_predictions:
                    future = pred.get("future", [])
                    n_pred = len(future)
                    for t_idx, (fx_m, fy_m) in enumerate(future):
                        # Transform point de map vers cam
                        pm = PointStamped()
                        pm.header.frame_id = self.target_frame
                        pm.header.stamp = self.last_depth_stamp
                        pm.point.x = float(fx_m)
                        pm.point.y = float(fy_m)
                        pm.point.z = 0.0
                        try:
                            pcam = do_transform_point(pm, tf_map_to_cam)
                        except Exception:
                            continue
                        # Genere un mini cylindre (moins de points pour
                        # cout reduit dans la costmap)
                        cylinder_points.extend(
                            gen_cylinder_points(
                                pcam.point.x, pcam.point.y, pcam.point.z,
                                radius=OBSTACLE_RADIUS * 0.8,
                                n_ang=8, n_v=2)
                        )
                        # Marker sphere transparente pour visualisation
                        sph = Marker()
                        sph.header.frame_id = self.target_frame
                        sph.header.stamp = pose_array.header.stamp
                        sph.ns = f"future_{pred['id']}"
                        sph.id = t_idx + 50000
                        sph.type = Marker.SPHERE
                        sph.action = Marker.ADD
                        sph.pose.position.x = float(fx_m)
                        sph.pose.position.y = float(fy_m)
                        sph.pose.position.z = 0.1
                        sph.pose.orientation.w = 1.0
                        sph.scale.x = sph.scale.y = sph.scale.z = 0.15
                        # Fade out avec t
                        alpha = 0.6 * (1.0 - t_idx / max(1, n_pred - 1)) + 0.1
                        sph.color.a = alpha
                        sph.color.r = 0.0
                        sph.color.g = 0.6
                        sph.color.b = 1.0
                        markers.markers.append(sph)
            except Exception as e:
                self.get_logger().warn(
                    f"TF map->cam echoue : {e}", throttle_duration_sec=2.0)

        # Publication
        cloud = make_pointcloud2(cylinder_points, DEPTH_FRAME, msg.header.stamp)
        self.pub_cloud.publish(cloud)
        self.pub_poses.publish(pose_array)
        self.pub_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleProjectorV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

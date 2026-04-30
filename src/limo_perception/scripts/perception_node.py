#!/usr/bin/env python3
"""
perception_node.py — A2 perception layer (YOLO + DeepSORT)

Pipeline : /camera/image_raw → YOLOv8-nano → DeepSORT → tracks publiés

Subscribe : /camera/image_raw          (sensor_msgs/Image)
Publish   : /detections                (vision_msgs/Detection2DArray)
            /tracked_obstacles         (vision_msgs/Detection2DArray) — avec track ID stable
            /perception/debug_image    (sensor_msgs/Image) — bboxes + IDs

Diff vs v1 :
  - Ajout DeepSORT (deep_sort_realtime)
  - Nouveau topic /tracked_obstacles avec ID persistant
  - Debug image affiche maintenant l'ID du track

Run :
  python3 perception_node.py
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


DYNAMIC_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
}


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("max_age", 30)
        self.declare_parameter("n_init", 3)
        self.declare_parameter("max_iou_distance", 0.7)
        self.declare_parameter("nms_max_overlap", 1.0)

        model_path = self.get_parameter("model_path").value
        self.conf_threshold = self.get_parameter("conf_threshold").value
        self.iou_threshold = self.get_parameter("iou_threshold").value
        self.publish_debug = self.get_parameter("publish_debug_image").value
        device = self.get_parameter("device").value

        if device == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA demandé mais indisponible — fallback CPU")
            device = "cpu"
        self.device = device

        self.get_logger().info(f"Chargement YOLO : {model_path} sur {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.get_logger().info("Modèle YOLO chargé")

        self.get_logger().info("Initialisation DeepSORT...")
        self.tracker = DeepSort(
            max_age=self.get_parameter("max_age").value,
            n_init=self.get_parameter("n_init").value,
            max_iou_distance=self.get_parameter("max_iou_distance").value,
            nms_max_overlap=self.get_parameter("nms_max_overlap").value,
            embedder="mobilenet",
            half=True,
            embedder_gpu=(self.device == "cuda"),
        )
        self.get_logger().info("DeepSORT prêt")

        self.bridge = CvBridge()

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_image = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, image_qos
        )

        self.pub_detections = self.create_publisher(
            Detection2DArray, "/detections", 10
        )
        self.pub_tracks = self.create_publisher(
            Detection2DArray, "/tracked_obstacles", 10
        )
        if self.publish_debug:
            self.pub_debug = self.create_publisher(
                Image, "/perception/debug_image", 10
            )

        self.frame_count = 0
        self.track_count_total = 0
        self.last_log_time = self.get_clock().now()
        self.create_timer(5.0, self.log_stats)

        self.get_logger().info("PerceptionNode prêt — en attente de /camera/image_raw")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Conversion image échouée : {e}")
            return

        results = self.model(
            cv_image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )

        deepsort_input = []
        det_array = Detection2DArray()
        det_array.header = msg.header

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                class_id = int(cls[i])
                if class_id not in DYNAMIC_CLASSES:
                    continue

                x1, y1, x2, y2 = xyxy[i]
                w = float(x2 - x1)
                h = float(y2 - y1)
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                deepsort_input.append((
                    [float(x1), float(y1), w, h],
                    float(conf[i]),
                    DYNAMIC_CLASSES[class_id],
                ))

                det = Detection2D()
                det.header = msg.header
                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.size_x = w
                det.bbox.size_y = h
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(class_id)
                hyp.hypothesis.score = float(conf[i])
                det.results.append(hyp)
                det_array.detections.append(det)

        # DeepSORT update
        tracks = self.tracker.update_tracks(deepsort_input, frame=cv_image)

        track_array = Detection2DArray()
        track_array.header = msg.header
        active_tracks = 0

        for track in tracks:
            if not track.is_confirmed():
                continue
            active_tracks += 1

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = ltrb
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            w = float(x2 - x1)
            h = float(y2 - y1)
            class_name = track.get_det_class() if track.get_det_class() else "unknown"

            det = Detection2D()
            det.header = msg.header
            det.id = str(track_id)
            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.size_x = w
            det.bbox.size_y = h
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_name
            hyp.hypothesis.score = float(track.get_det_conf() or 0.0)
            det.results.append(hyp)
            track_array.detections.append(det)

            if self.publish_debug:
                color = self._color_from_id(int(track_id))
                cv2.rectangle(
                    cv_image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                )
                label = f"ID:{track_id} {class_name}"
                cv2.putText(
                    cv_image,
                    label,
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        self.pub_detections.publish(det_array)
        self.pub_tracks.publish(track_array)

        if self.publish_debug:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            debug_msg.header = msg.header
            self.pub_debug.publish(debug_msg)

        self.frame_count += 1
        self.track_count_total += active_tracks

    def _color_from_id(self, track_id: int):
        np.random.seed(track_id)
        c = np.random.randint(0, 255, size=3).tolist()
        return (int(c[0]), int(c[1]), int(c[2]))

    def log_stats(self):
        now = self.get_clock().now()
        dt = (now - self.last_log_time).nanoseconds / 1e9
        if dt > 0:
            fps = self.frame_count / dt
            avg_tracks = self.track_count_total / max(self.frame_count, 1)
            self.get_logger().info(
                f"Perception : {self.frame_count} frames / {dt:.1f}s → {fps:.1f} FPS "
                f"| tracks moyens/frame : {avg_tracks:.2f}"
            )
        self.frame_count = 0
        self.track_count_total = 0
        self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
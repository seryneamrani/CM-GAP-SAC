#!/usr/bin/env python3
"""
trajectory_predictor_node.py -- A2 trajectory prediction layer

Subscribe :
  /tracked_obstacles_3d  (vision_msgs/Detection3DArray, frame=map)
                         publie par obstacle_projector V2 avec track IDs preserves

Publish :
  /predicted_trajectories  (visualization_msgs/MarkerArray)
                           LINE_STRIP par track + SPHERE au point final

Strategie : buffer glissant de obs_len positions par track_id. Quand un track
a obs_len observations, on lance Social-LSTM Lite pour predire pred_len pas.
Tracks non vus depuis max_track_age_s sont purges.

Run :
  ros2 run limo_perception trajectory_predictor_node \
      --ros-args -p model_path:=$HOME/social_lstm_lite/checkpoints/best_eth_ucy.pt
"""
import os
from collections import defaultdict, deque

import numpy as np
import torch

import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from limo_perception.models.social_lstm_lite import SocialLSTMLite


class TrajectoryPredictorNode(Node):
    def __init__(self):
        super().__init__('trajectory_predictor')

        self.declare_parameter('model_path', '')
        self.declare_parameter('obs_len', 8)
        self.declare_parameter('pred_len', 12)
        self.declare_parameter('hidden_size', 32)
        self.declare_parameter('embedding_dim', 16)
        self.declare_parameter('grid_size', 4)
        self.declare_parameter('neighborhood', 2.0)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('target_classes', ['person'])
        self.declare_parameter('max_track_age_s', 1.0)
        self.declare_parameter('device', 'cuda')

        model_path = self.get_parameter('model_path').value
        self.obs_len = int(self.get_parameter('obs_len').value)
        self.pred_len = int(self.get_parameter('pred_len').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.target_classes = list(self.get_parameter('target_classes').value)
        self.max_track_age = float(self.get_parameter('max_track_age_s').value)
        device_param = self.get_parameter('device').value

        if device_param == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA demande mais indisponible -- fallback CPU')
            device_param = 'cpu'
        self.device = torch.device(device_param)

        self.model = SocialLSTMLite(
            embedding_dim=int(self.get_parameter('embedding_dim').value),
            hidden_size=int(self.get_parameter('hidden_size').value),
            grid_size=int(self.get_parameter('grid_size').value),
            neighborhood=float(self.get_parameter('neighborhood').value),
            pred_len=self.pred_len,
        ).to(self.device)

        if model_path and os.path.isfile(model_path):
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.get_logger().info(f'Modele charge : {model_path}')
            except Exception as e:
                self.get_logger().error(f'Echec chargement modele : {e}')
        else:
            self.get_logger().warn(
                f'Pas de checkpoint a "{model_path}" : '
                'modele non entraine, predictions aleatoires'
            )

        self.model.eval()

        # Buffer glissant : track_id (str) -> deque[(x, y)]
        self.track_buffers = defaultdict(lambda: deque(maxlen=self.obs_len))
        # Derniere mise a jour : track_id -> ros time (secondes)
        self.last_seen = {}

        self.sub = self.create_subscription(
            Detection3DArray, '/tracked_obstacles_3d', self.cb_tracks, 10
        )
        self.pub = self.create_publisher(
            MarkerArray, '/predicted_trajectories', 10
        )

        self.create_timer(1.0, self._cleanup_old_tracks)

        self.n_inferences = 0
        self.create_timer(5.0, self._log_stats)

        self.get_logger().info(
            f'TrajectoryPredictor pret -- obs_len={self.obs_len} '
            f'pred_len={self.pred_len} classes={self.target_classes} '
            f'device={self.device}'
        )

    def cb_tracks(self, msg: Detection3DArray):
        now = self.get_clock().now().nanoseconds * 1e-9

        ready_ids = []
        for det in msg.detections:
            track_id = det.id
            if not track_id:
                continue

            class_name = ''
            if len(det.results) > 0:
                class_name = det.results[0].hypothesis.class_id
            if self.target_classes and class_name not in self.target_classes:
                continue

            x = float(det.bbox.center.position.x)
            y = float(det.bbox.center.position.y)

            self.track_buffers[track_id].append((x, y))
            self.last_seen[track_id] = now

            if len(self.track_buffers[track_id]) == self.obs_len:
                ready_ids.append(track_id)

        if not ready_ids:
            self._publish_empty()
            return

        # Construction du tenseur (obs_len, N, 2)
        obs_np = np.stack(
            [np.array(list(self.track_buffers[tid]), dtype=np.float32)
             for tid in ready_ids],
            axis=1,
        )
        obs = torch.from_numpy(obs_np).to(self.device)

        with torch.no_grad():
            pred = self.model(obs, pred_len=self.pred_len)
        pred_np = pred.cpu().numpy()
        self.n_inferences += 1

        # Publication MarkerArray
        marker_array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        stamp = self.get_clock().now().to_msg()

        for i, tid in enumerate(ready_ids):
            try:
                tid_int = int(tid)
            except (TypeError, ValueError):
                tid_int = abs(hash(tid)) % 100000

            color = self._color_from_id(tid_int)

            # LINE_STRIP : derniere obs + futur predit
            line = Marker()
            line.header.stamp = stamp
            line.header.frame_id = self.frame_id
            line.ns = 'predicted_traj'
            line.id = tid_int
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.05
            line.color = color
            line.pose.orientation.w = 1.0
            line.lifetime.sec = 1

            obs_last = self.track_buffers[tid][-1]
            line.points.append(Point(
                x=float(obs_last[0]), y=float(obs_last[1]), z=0.05
            ))
            for t in range(self.pred_len):
                line.points.append(Point(
                    x=float(pred_np[t, i, 0]),
                    y=float(pred_np[t, i, 1]),
                    z=0.05,
                ))
            marker_array.markers.append(line)

            # SPHERE au point final
            tip = Marker()
            tip.header.stamp = stamp
            tip.header.frame_id = self.frame_id
            tip.ns = 'predicted_tip'
            tip.id = tid_int
            tip.type = Marker.SPHERE
            tip.action = Marker.ADD
            tip.pose.position.x = float(pred_np[-1, i, 0])
            tip.pose.position.y = float(pred_np[-1, i, 1])
            tip.pose.position.z = 0.1
            tip.pose.orientation.w = 1.0
            tip.scale.x = 0.15
            tip.scale.y = 0.15
            tip.scale.z = 0.15
            tip.color = color
            tip.lifetime.sec = 1
            marker_array.markers.append(tip)

        self.pub.publish(marker_array)

    def _publish_empty(self):
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        self.pub.publish(ma)

    def _cleanup_old_tracks(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        stale = [tid for tid, t in self.last_seen.items()
                 if (now - t) > self.max_track_age]
        for tid in stale:
            self.track_buffers.pop(tid, None)
            self.last_seen.pop(tid, None)

    def _color_from_id(self, tid: int) -> ColorRGBA:
        rng = np.random.default_rng(tid)
        c = rng.random(3)
        return ColorRGBA(
            r=float(c[0]), g=float(c[1]), b=float(c[2]), a=0.9
        )

    def _log_stats(self):
        self.get_logger().info(
            f'Predictor : {self.n_inferences} inferences (5s) | '
            f'tracks actifs : {len(self.track_buffers)}'
        )
        self.n_inferences = 0


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

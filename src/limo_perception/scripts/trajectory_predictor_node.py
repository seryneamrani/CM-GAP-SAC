"""
Node ROS 2 Jazzy : prédiction de trajectoire à partir des tracks DeepSORT.

Topic d'entrée  : /tracked_persons (vision_msgs/Detection3DArray ou custom)
Topic de sortie : /predicted_trajectories (nav_msgs/Path par track)

Maintient un buffer glissant de obs_len frames par track. Quand un track
a au moins obs_len observations, on lance l'inférence Social-LSTM Lite et
on publie pred_len positions futures.

À placer dans ton package ROS 2 (par ex. limo_perception/scripts/).

Lance avec :
    ros2 run limo_perception trajectory_predictor_node \
        --ros-args -p model_path:=/path/to/best_eth_ucy.pt
"""

import rclpy
from rclpy.node import Node
from collections import defaultdict, deque
import numpy as np
import torch
import sys
import os

# Adapter le chemin selon ton workspace
sys.path.insert(0, os.path.expanduser("~/social_lstm_lite"))
from models.social_lstm_lite import SocialLSTMLite

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header

# Si tu utilises un msg custom pour les tracks, importe-le ici.
# Exemple générique avec Detection3DArray :
from vision_msgs.msg import Detection3DArray


class TrajectoryPredictorNode(Node):
    def __init__(self):
        super().__init__("trajectory_predictor")

        self.declare_parameter("model_path", "")
        self.declare_parameter("obs_len", 8)
        self.declare_parameter("pred_len", 12)
        self.declare_parameter("hidden_size", 32)
        self.declare_parameter("embedding_dim", 16)
        self.declare_parameter("frame_id", "map")

        model_path = self.get_parameter("model_path").value
        self.obs_len = self.get_parameter("obs_len").value
        self.pred_len = self.get_parameter("pred_len").value
        self.frame_id = self.get_parameter("frame_id").value

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SocialLSTMLite(
            embedding_dim=self.get_parameter("embedding_dim").value,
            hidden_size=self.get_parameter("hidden_size").value,
            grid_size=4,
            neighborhood=2.0,
            pred_len=self.pred_len,
        ).to(self.device)

        if model_path and os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.get_logger().info(f"Modèle chargé : {model_path}")
        else:
            self.get_logger().warn("Pas de checkpoint, modèle non entraîné.")

        self.model.eval()

        # Buffer glissant : track_id -> deque de positions (x, y)
        self.track_buffers = defaultdict(lambda: deque(maxlen=self.obs_len))

        self.sub = self.create_subscription(
            Detection3DArray, "/tracked_persons", self.tracks_callback, 10
        )
        self.pub = self.create_publisher(Path, "/predicted_trajectories", 10)

        self.get_logger().info("TrajectoryPredictorNode prêt.")

    def tracks_callback(self, msg: Detection3DArray):
        # 1. Mettre à jour les buffers
        current_ids = []
        for det in msg.detections:
            track_id = det.id  # adapte selon ton msg
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            self.track_buffers[track_id].append((x, y))
            current_ids.append(track_id)

        # 2. Sélectionner les tracks avec obs_len positions complètes
        ready_ids = [tid for tid in current_ids
                     if len(self.track_buffers[tid]) == self.obs_len]
        if not ready_ids:
            return

        # 3. Construire le tenseur d'entrée (obs_len, N, 2)
        obs_np = np.stack(
            [np.array(self.track_buffers[tid]) for tid in ready_ids], axis=1
        )  # (obs_len, N, 2)
        obs = torch.from_numpy(obs_np).float().to(self.device)

        # 4. Inférence
        with torch.no_grad():
            pred = self.model(obs, pred_len=self.pred_len)  # (pred_len, N, 2)
        pred_np = pred.cpu().numpy()

        # 5. Publier un Path par track (un seul topic ici, pour démarrer
        #    on publie le premier track ; à adapter en MarkerArray multi-track)
        for i, tid in enumerate(ready_ids):
            path = Path()
            path.header = Header()
            path.header.stamp = self.get_clock().now().to_msg()
            path.header.frame_id = self.frame_id
            for t in range(self.pred_len):
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = float(pred_np[t, i, 0])
                ps.pose.position.y = float(pred_np[t, i, 1])
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.pub.publish(path)


def main():
    rclpy.init()
    node = TrajectoryPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

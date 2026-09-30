#!/usr/bin/env python3
"""
trajectory_predictor_node.py -- Inference Social-LSTM Lite sur tracks dynamiques.

Subscribe :
    /track_states    (std_msgs/String JSON) -- etats + historique 8 frames

Publish :
    /predicted_trajectories          (nav_msgs/Path[]) -- une Path par track
    /predicted_trajectories_markers  (visualization_msgs/MarkerArray) pour RViz

Strategie :
    - Pour chaque track classe "dynamic" ou "starts_moving" avec history >= 8,
      on extrait (x, y, vx, vy)_t pour t in [-7..0].
    - On normalise par la derniere position (origine).
    - Inference => (dx, dy)_t pour t in [+1..+6] (1.5s a 0.25s d'intervalle).
    - On reconstruit les positions absolues et publie.
"""
import json
import os

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time as TimeMsg

# Import du modele (place dans le meme repertoire scripts/)
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from social_lstm_lite_model import load_model, SocialLSTMLite


T_OBS = 8         # frames d'observation
T_PRED = 6        # frames de prediction
DT_PRED = 0.25    # seconds entre predictions


class TrajectoryPredictorNode(Node):
    def __init__(self):
        super().__init__("trajectory_predictor_node")

        self.declare_parameter(
            "model_path",
            os.path.expanduser(
                "~/limo_jazzy_ws/src/limo_perception/models/social_lstm_lite.pt"
            ),
        )
        self.declare_parameter("device", "cuda")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("predict_states",
                               ["dynamic", "starts_moving"])

        model_path = self.get_parameter("model_path").value
        device = self.get_parameter("device").value
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.frame_id = self.get_parameter("frame_id").value
        self.predict_states = list(self.get_parameter("predict_states").value)

        # Chargement du modele
        if not os.path.exists(model_path):
            self.get_logger().error(
                f"Modele introuvable : {model_path}. "
                "Le node tournera en idle (pas de prediction)."
            )
            self.model = None
        else:
            try:
                self.model = load_model(model_path, device=self.device)
                n_params = sum(p.numel() for p in self.model.parameters())
                self.get_logger().info(
                    f"LSTM charge depuis {model_path} ({n_params:,} params, {self.device})"
                )
            except Exception as e:
                self.get_logger().error(f"Echec chargement modele : {e}")
                self.model = None

        # Subs / pubs
        self.create_subscription(String, "/track_states",
                                 self._cb_states, 10)
        self.pub_paths = self.create_publisher(
            Path, "/predicted_trajectories", 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, "/predicted_trajectories_markers", 10)

        # Aussi : on publie une concatenation indexee par id en JSON
        self.pub_json = self.create_publisher(
            String, "/predicted_trajectories_json", 10)

        self.get_logger().info("TrajectoryPredictor pret.")

    def _cb_states(self, msg: String):
        if self.model is None:
            return
        try:
            payload = json.loads(msg.data)
        except Exception:
            return

        # Filtre les tracks a predire
        batch_inputs = []
        batch_meta = []
        for t in payload:
            if t.get("state") not in self.predict_states:
                continue
            xs = t.get("history_x", [])
            ys = t.get("history_y", [])
            vxs = t.get("history_vx", [])
            vys = t.get("history_vy", [])
            # On a besoin de 8 positions et 8 vitesses (derniere v est
            # calculee entre x[-2] et x[-1])
            if len(xs) < T_OBS or len(vxs) < T_OBS - 1:
                continue
            # Aligne : on prend les T_OBS derniers points
            xs = xs[-T_OBS:]
            ys = ys[-T_OBS:]
            # On padd vxs/vys au debut (premiere vitesse = vitesse second pt)
            vxs = ([vxs[0]] + vxs[-(T_OBS - 1):]) if vxs else [0.0] * T_OBS
            vys = ([vys[0]] + vys[-(T_OBS - 1):]) if vys else [0.0] * T_OBS
            # Normalisation : origine sur la derniere position
            ox, oy = xs[-1], ys[-1]
            x_norm = [xx - ox for xx in xs]
            y_norm = [yy - oy for yy in ys]
            features = np.stack([x_norm, y_norm, vxs, vys], axis=1)  # (T_OBS, 4)
            batch_inputs.append(features)
            batch_meta.append({"id": t["id"], "ox": ox, "oy": oy,
                               "class": t.get("class", "unknown"),
                               "state": t["state"]})

        if not batch_inputs:
            # Publie un MarkerArray vide pour clear RViz
            self.pub_markers.publish(MarkerArray(markers=[Marker(
                action=Marker.DELETEALL)]))
            self.pub_json.publish(String(data="[]"))
            return

        # Inference batch
        x = torch.tensor(np.stack(batch_inputs), dtype=torch.float32,
                         device=self.device)
        with torch.no_grad():
            y = self.model(x).cpu().numpy()    # (B, T_PRED, 2) -- deltas

        # Construction des sorties
        markers = MarkerArray()
        # Clear precedent
        markers.markers.append(Marker(action=Marker.DELETEALL))
        json_out = []
        now = self.get_clock().now().to_msg()

        for i, meta in enumerate(batch_meta):
            ox, oy = meta["ox"], meta["oy"]
            preds = y[i]  # (T_PRED, 2)
            future = []
            for t_idx in range(T_PRED):
                fx = ox + float(preds[t_idx, 0])
                fy = oy + float(preds[t_idx, 1])
                future.append((fx, fy))
            json_out.append({"id": meta["id"], "class": meta["class"],
                             "state": meta["state"], "future": future,
                             "dt": DT_PRED})

            # Marker LINE_STRIP pour visualisation
            m = Marker()
            m.header = Header(frame_id=self.frame_id, stamp=now)
            m.ns = f"pred_{meta['id']}"
            m.id = int(meta["id"]) if meta["id"].isdigit() else hash(meta["id"]) % 1000
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color.a = 0.9
            m.color.r = 1.0 if meta["state"] == "dynamic" else 1.0
            m.color.g = 0.6 if meta["state"] == "dynamic" else 1.0
            m.color.b = 0.0
            # Point depart : derniere obs
            p0 = Point(); p0.x = ox; p0.y = oy; p0.z = 0.05
            m.points.append(p0)
            for fx, fy in future:
                p = Point(); p.x = fx; p.y = fy; p.z = 0.05
                m.points.append(p)
            markers.markers.append(m)

            # Aussi : sphere a +1.5s pour souligner l'arrivee
            sph = Marker()
            sph.header = Header(frame_id=self.frame_id, stamp=now)
            sph.ns = f"pred_end_{meta['id']}"
            sph.id = m.id + 10000
            sph.type = Marker.SPHERE
            sph.action = Marker.ADD
            sph.pose.position.x = future[-1][0]
            sph.pose.position.y = future[-1][1]
            sph.pose.position.z = 0.1
            sph.pose.orientation.w = 1.0
            sph.scale.x = sph.scale.y = sph.scale.z = 0.2
            sph.color.a = 0.7
            sph.color.r = 1.0
            sph.color.g = 0.2
            sph.color.b = 0.2
            markers.markers.append(sph)

            # Publie une Path par track (utile si Nav2 ou autre la consomme)
            path = Path()
            path.header = Header(frame_id=self.frame_id, stamp=now)
            for fx, fy in future:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = fx
                ps.pose.position.y = fy
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.pub_paths.publish(path)

        self.pub_markers.publish(markers)
        self.pub_json.publish(String(data=json.dumps(json_out)))


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


if __name__ == "__main__":
    main()

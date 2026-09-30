#!/usr/bin/env python3
"""
track_classifier_node.py -- Classifie chaque track DeepSORT en 4 etats.

Etats :
    static            -- vitesse < 0.10 m/s sur 8 frames, classe non-humaine
    probably_static   -- vitesse < 0.10 m/s sur 8 frames, classe humaine/animale
    starts_moving     -- vitesse instantanee 0.10-0.30 m/s, accel positive
    dynamic           -- vitesse > 0.30 m/s sur 8 frames

Hysteresis :
    Pour eviter les oscillations, un track classe dynamic ne repasse
    en static qu'apres 15 frames sous le seuil 0.10 m/s.

Subscribe :
    /tracked_obstacles    (vision_msgs/Detection2DArray)  -- ID + classe
    /dynamic_obstacles    (geometry_msgs/PoseArray)        -- positions monde

Publish :
    /track_states         (limo_msgs/TrackStateArray ou std_msgs/String JSON)
                          Note : on publie en JSON via std_msgs/String pour
                          eviter de creer un nouveau msg type.

Format JSON publie :
    [
      {"id": "3", "class": "person", "state": "dynamic",
       "speed": 0.62, "history_x": [...], "history_y": [...],
       "history_vx": [...], "history_vy": [...]},
      ...
    ]
"""
import json
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseArray


HUMAN_LIKE_CLASSES = {"person", "cat", "dog", "bicycle", "motorcycle"}

V_STATIC_MAX = 0.10        # m/s
V_DYNAMIC_MIN = 0.30       # m/s
HISTORY_LEN = 8            # nombre de frames pour la moyenne
HYSTERESIS_FRAMES = 15     # frames sous seuil avant repasse en static


class TrackHistory:
    def __init__(self, track_id: str, class_name: str):
        self.id = track_id
        self.class_name = class_name
        self.x = deque(maxlen=HISTORY_LEN)
        self.y = deque(maxlen=HISTORY_LEN)
        self.vx = deque(maxlen=HISTORY_LEN)
        self.vy = deque(maxlen=HISTORY_LEN)
        self.last_t = None
        self.state = "static"
        self.frames_below_static = 0
        self.last_seen = 0  # frame counter de last update

    def update(self, x: float, y: float, t: float, frame_counter: int):
        if self.last_t is not None and t > self.last_t:
            dt = t - self.last_t
            if dt > 1e-3 and len(self.x) > 0:
                vx_inst = (x - self.x[-1]) / dt
                vy_inst = (y - self.y[-1]) / dt
                self.vx.append(vx_inst)
                self.vy.append(vy_inst)
        self.x.append(x)
        self.y.append(y)
        self.last_t = t
        self.last_seen = frame_counter

    def speed_mean(self) -> float:
        if not self.vx:
            return 0.0
        n = len(self.vx)
        s = 0.0
        for i in range(n):
            s += (self.vx[i] ** 2 + self.vy[i] ** 2) ** 0.5
        return s / n

    def speed_inst(self) -> float:
        if not self.vx:
            return 0.0
        return (self.vx[-1] ** 2 + self.vy[-1] ** 2) ** 0.5

    def accel_positive(self) -> bool:
        """Vrai si la vitesse moyenne sur les 4 dernieres frames > 4 precedentes."""
        if len(self.vx) < 8:
            return False
        recent = sum(
            (self.vx[i] ** 2 + self.vy[i] ** 2) ** 0.5
            for i in range(-4, 0)
        ) / 4.0
        prev = sum(
            (self.vx[i] ** 2 + self.vy[i] ** 2) ** 0.5
            for i in range(-8, -4)
        ) / 4.0
        return recent > prev * 1.1

    def classify(self) -> str:
        """Retourne le nouvel etat avec hysteresis."""
        v_mean = self.speed_mean()
        v_inst = self.speed_inst()
        is_human = self.class_name in HUMAN_LIKE_CLASSES

        # Hysteresis : si actuellement dynamic, on demande N frames sous seuil
        if self.state == "dynamic":
            if v_mean < V_STATIC_MAX:
                self.frames_below_static += 1
                if self.frames_below_static >= HYSTERESIS_FRAMES:
                    self.frames_below_static = 0
                    return "probably_static" if is_human else "static"
                return "dynamic"
            self.frames_below_static = 0
            return "dynamic"

        # Sinon, classification standard
        if v_mean > V_DYNAMIC_MIN:
            self.frames_below_static = 0
            return "dynamic"

        if V_STATIC_MAX <= v_inst <= V_DYNAMIC_MIN and self.accel_positive():
            return "starts_moving"

        if v_mean < V_STATIC_MAX:
            return "probably_static" if is_human else "static"

        return self.state  # fallback : pas de changement


class TrackClassifierNode(Node):
    def __init__(self):
        super().__init__("track_classifier_node")

        # Map track_id -> TrackHistory
        self.tracks: dict = {}
        self.track_classes: dict = {}     # id -> class_name (depuis /tracked_obstacles)
        self.frame_counter = 0
        self.cleanup_after = 30           # supprimer track non vu depuis N frames

        self.create_subscription(
            Detection2DArray, "/tracked_obstacles", self._cb_tracked, 10)
        self.create_subscription(
            PoseArray, "/dynamic_obstacles", self._cb_poses, 10)

        self.pub = self.create_publisher(String, "/track_states", 10)

        self.create_timer(0.5, self._publish_states)
        self.get_logger().info("TrackClassifier pret (4 etats + hysteresis).")

    def _cb_tracked(self, msg: Detection2DArray):
        """Maj des classes par track_id."""
        for det in msg.detections:
            if not det.id:
                continue
            tid = str(det.id)
            if det.results:
                self.track_classes[tid] = det.results[0].hypothesis.class_id

    def _cb_poses(self, msg: PoseArray):
        """Maj position monde de chaque track. msg.poses[i] suppose dans
        l'ordre des detections /tracked_obstacles. On utilise le header
        timestamp pour calculer les vitesses."""
        self.frame_counter += 1
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Sans correlation explicite poses<->tracks, on associe par ordre.
        # Pour etre robuste, le projecteur publie aussi un PoseArray annote
        # mais ici on travaille avec ce qu'on a : on suppose que les ids
        # vivants sont les memes que dans /tracked_obstacles recu juste avant.
        # Un workaround simple : on nomme les tracks par leur ordre.
        live_ids = list(self.track_classes.keys())
        for i, pose in enumerate(msg.poses):
            if i >= len(live_ids):
                break
            tid = live_ids[i]
            class_name = self.track_classes.get(tid, "unknown")
            if tid not in self.tracks:
                self.tracks[tid] = TrackHistory(tid, class_name)
            else:
                self.tracks[tid].class_name = class_name
            self.tracks[tid].update(pose.position.x, pose.position.y, t,
                                     self.frame_counter)
            self.tracks[tid].state = self.tracks[tid].classify()

        # Cleanup des tracks non vus
        to_remove = [
            tid for tid, h in self.tracks.items()
            if self.frame_counter - h.last_seen > self.cleanup_after
        ]
        for tid in to_remove:
            del self.tracks[tid]
            self.track_classes.pop(tid, None)

    def _publish_states(self):
        payload = []
        for tid, h in self.tracks.items():
            payload.append({
                "id": tid,
                "class": h.class_name,
                "state": h.state,
                "speed": round(h.speed_mean(), 3),
                "speed_inst": round(h.speed_inst(), 3),
                "history_x": [round(v, 3) for v in h.x],
                "history_y": [round(v, 3) for v in h.y],
                "history_vx": [round(v, 3) for v in h.vx],
                "history_vy": [round(v, 3) for v in h.vy],
            })
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrackClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
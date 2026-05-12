#!/usr/bin/env python3
import math
import yaml
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped


DYNAMICITY_PROFILES = {
    "low":    {"speed": 0.4, "pause": 4.0},
    "medium": {"speed": 0.7, "pause": 1.5},
    "high":   {"speed": 1.0, "pause": 0.3},
}


class Pedestrian:
    """Pieton avec liste de waypoints, boucle infinie ou aller simple."""

    def __init__(self, cfg, profile, speed_scale, tol):
        self.name      = cfg["name"]
        self.behavior  = cfg.get("behavior", "loop")
        self.speed     = cfg.get("speed", profile["speed"]) * speed_scale
        self.pause     = cfg.get("pause_at_endpoints", profile["pause"])
        self.tol       = tol

        # Support 2 formats : waypoints OU start+goal (retro-compat)
        if "waypoints" in cfg:
            self.waypoints = [tuple(wp) for wp in cfg["waypoints"]]
        elif "start" in cfg and "goal" in cfg:
            self.waypoints = [tuple(cfg["start"]), tuple(cfg["goal"])]
        else:
            raise ValueError(
                f"{self.name}: il faut 'waypoints' OU 'start'+'goal'")

        if len(self.waypoints) < 2:
            raise ValueError(f"{self.name}: au moins 2 waypoints requis")

        # Etat courant
        self.x, self.y = self.waypoints[0]
        self.idx       = 1            # index du prochain waypoint
        self.direction = 1            # 1 = avant, -1 = retour (ping_pong)
        self.paused    = False
        self.pause_t0  = None
        self.done      = False

    def step(self, now_sec, dt):
        cmd = Twist()
        if self.done:
            return cmd

        # Gestion de la pause aux waypoints
        if self.paused:
            if self.pause_t0 is None:
                self.pause_t0 = now_sec
            if (now_sec - self.pause_t0) >= self.pause:
                self.paused = False
                self.pause_t0 = None
            else:
                return cmd

        target = self.waypoints[self.idx]
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = math.hypot(dx, dy)

        if dist < self.tol:
            # Waypoint atteint
            self.paused = True
            self._advance_index()
            return cmd

        # Distance de freinage : a moins de 0.5 m, on ralentit lineairement
        brake_dist = 0.5
        if dist < brake_dist:
            speed_now = self.speed * max(dist / brake_dist, 0.2)
        else:
            speed_now = self.speed

        cmd.linear.x = speed_now * (dx / dist)
        cmd.linear.y = speed_now * (dy / dist)

        # Integration locale
        self.x += cmd.linear.x * dt
        self.y += cmd.linear.y * dt

        return cmd

    def _advance_index(self):
        n = len(self.waypoints)
        if self.behavior == "loop":
            self.idx = (self.idx + 1) % n
        elif self.behavior == "ping_pong":
            self.idx += self.direction
            if self.idx >= n:
                self.idx = n - 2
                self.direction = -1
            elif self.idx < 0:
                self.idx = 1
                self.direction = 1
        elif self.behavior == "one_shot":
            if self.idx + 1 < n:
                self.idx += 1
            else:
                self.done = True


class PedestrianManager(Node):

    def __init__(self):
        super().__init__("pedestrian_manager")

        self.declare_parameter("config_file", "")
        cfg_path = self.get_parameter("config_file").value
        if not cfg_path:
            cfg_path = str(Path(__file__).resolve().parent.parent
                           / "config" / "pedestrians.yaml")

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        g = cfg["global"]
        self.world_name = g["world_name"]
        self.rate_hz    = float(g["update_rate_hz"])
        self.dt         = 1.0 / self.rate_hz
        tol             = float(g.get("arrival_tolerance", 0.10))

        profile = DYNAMICITY_PROFILES[g["dynamicity"]]
        speed_scale = float(g["speed_scale"])

        qos_cmd = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_pose = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=10)

        self.peds = []
        self.pubs = {}
        for pcfg in cfg["pedestrians"]:
            ped = Pedestrian(pcfg, profile, speed_scale, tol)
            self.pubs[ped.name] = self.create_publisher(
                Twist, f"/model/{ped.name}/cmd_vel", qos_cmd)

            def make_cb(p):
                def cb(msg):
                    p.x = msg.pose.position.x
                    p.y = msg.pose.position.y
                return cb


            self.create_subscription(
                PoseStamped, f"/model/{ped.name}/pose",
                make_cb(ped), qos_pose)

            self.peds.append(ped)
            self.get_logger().info(
                f"{ped.name}: {len(ped.waypoints)} waypoints, "
                f"behavior={ped.behavior}, v={ped.speed:.2f} m/s, "
                f"pause={ped.pause:.1f} s")

        self.t0 = self.get_clock().now()
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"PedestrianManager: {len(self.peds)} pietons, "
            f"world={self.world_name}, rate={self.rate_hz} Hz")

    def _tick(self):
        now = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        for ped in self.peds:
            self.pubs[ped.name].publish(ped.step(now, self.dt))


def main():
    rclpy.init()
    node = PedestrianManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        for pub in node.pubs.values():
            pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
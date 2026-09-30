#!/usr/bin/env python3
"""
inspect_real_tracker.py -- test de calibration du RealPerceptionTracker.

Instancie le tracker, le fait tourner, et affiche get_tracks_robot_frame()
une fois par seconde. Sert au test des 2-3 m du README :

    Place une personne a une distance connue PILE DEVANT le robot.
    Le tracker doit sortir x ~= distance, y ~= 0.

    Si y est biaise -> ajuste --cam-yaw-offset.
    Si x est faux  -> probleme de FoV ou de LiDAR (mauvais beam).

Usage:
    export ROS_DOMAIN_ID=20
    python3 inspect_real_tracker.py \
        --tracks /tracked_obstacles \
        --scan /model/limo/laser/scan \
        --hfov 71.0 \
        --img-width 640 \
        --cam-yaw-offset 0.0

ATTENTION au scan_topic : en sim c'est /model/limo/laser/scan (voir ton
sim.launch.py bridge), PAS /scan. Sur le robot reel ce sera /scan.
"""
import argparse
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from limo_perception.real_perception_tracker import RealPerceptionTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="/tracked_obstacles")
    ap.add_argument("--scan", default="/model/limo/laser/scan")
    ap.add_argument("--hfov", type=float, default=71.0,
                    help="FoV horizontal camera en degres. "
                         "Ton xacro: horizontal_fov=1.239 rad = 71.0 deg.")
    ap.add_argument("--img-width", type=int, default=640)
    ap.add_argument("--cam-yaw-offset", type=float, default=0.0,
                    help="rad. Ajuste si y biaise au test pile-devant.")
    args = ap.parse_args()

    rclpy.init()
    tracker = RealPerceptionTracker(
        tracks_topic=args.tracks,
        scan_topic=args.scan,
        image_width=args.img_width,
        h_fov_deg=args.hfov,
        cam_yaw_offset=args.cam_yaw_offset,
    )

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(tracker)

    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print(f"[inspect] tracker up. tracks={args.tracks} scan={args.scan} "
          f"hfov={args.hfov} deg yaw_offset={args.cam_yaw_offset} rad")
    print("[inspect] place une personne PILE DEVANT a distance connue.")
    print("[inspect] attendu: x ~= distance, y ~= 0\n")

    try:
        while True:
            time.sleep(1.0)
            tracks = tracker.get_tracks_robot_frame()
            if not tracks:
                print("[inspect] (aucun track -- personne vue ? scan recu ?)")
                continue
            for t in tracks:
                x, y = t.position_robot
                vx, vy = t.velocity_robot
                d = (x**2 + y**2) ** 0.5
                print(f"  id={t.track_id:<3d} "
                      f"pos_robot=({x:+.2f}, {y:+.2f})  "
                      f"dist={d:.2f}m  "
                      f"vel=({vx:+.2f}, {vy:+.2f})  "
                      f"age={t.age_frames}")
            print()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

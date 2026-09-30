# test_tracker.py
import rclpy, time, threading
from cm_gap_sac_navigation.perception.ground_truth_provider import GroundTruthTracker

rclpy.init()
t = GroundTruthTracker()
threading.Thread(target=lambda: rclpy.spin(t), daemon=True).start()

for _ in range(10):
    time.sleep(0.5)
    tracks = t.get_tracks()
    print(f"{len(tracks)} tracks:",
          [(tr.track_id, tr.position_world.round(2), tr.velocity_world.round(2))
           for tr in tracks])
rclpy.shutdown()
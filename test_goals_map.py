import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cm_gap_sac_navigation.training.train import (
    _sample_navigable_point, _sample_goal_point, WALLS, STATIC_OBSTACLES,
)

rng = np.random.default_rng(0)
spawn_pts = [p for p in (_sample_navigable_point(rng) for _ in range(3000)) if p]
goal_pts  = [p for p in (_sample_goal_point(rng) for _ in range(3000)) if p]
print(f"spawn kept {len(spawn_pts)}, goal kept {len(goal_pts)}")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, pts, title in [(axes[0], spawn_pts, "SPAWN (margin 0.25)"),
                       (axes[1], goal_pts, "GOAL (clearance 0.55)")]:
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, s=2)
    for xmin, xmax, ymin, ymax in WALLS:
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                   color='red', alpha=0.3))
    for ox, oy, orad in STATIC_OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), orad, color='orange', alpha=0.3))
    ax.set_aspect('equal'); ax.set_xlim(-8.5, 8.5); ax.set_ylim(-8.5, 8.5)
    ax.set_title(title); ax.grid(True, alpha=0.2)
plt.savefig("goals_map.png", dpi=100)
print("saved goals_map.png")
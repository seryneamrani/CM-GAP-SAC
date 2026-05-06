# cm_gap_sac_navigation (v0.2)

CM-GAP_SAC : Cross-Modal Gated Attention Prioritized Soft Actor-Critic with
Safety Shield, for socially-aware navigation on the AgileX LIMO Pro under
ROS 2 Jazzy + Gazebo Harmonic.

---

## 1. What is in this package

```
cm_gap_sac_navigation/
├── package.xml              # ROS 2 manifest
├── setup.py                 # ament_python setup
├── setup.cfg                # script destinations
├── README.md                # this file
│
├── config/
│   └── cm_gap_sac.yaml      # ALL hyperparameters live here
│
├── cm_gap_sac_navigation/   # the Python package
│   ├── utils/
│   │   ├── geometry.py            # frame transforms, lidar downsample
│   │   └── config_loader.py       # typed YAML -> dataclasses
│   │
│   ├── perception/
│   │   ├── track_state.py         # pedestrian set assembly with K_max + mask
│   │   ├── pedestrian_tracker.py  # subscriber to YOLO+DeepSORT output
│   │   └── ground_truth_provider.py  # GT pedestrians for ablation A5
│   │
│   ├── envs/
│   │   ├── reward.py              # PURE 6-term reward function (no ROS)
│   │   ├── gazebo_interface.py    # ROS 2 <-> Gazebo wires (scan, odom, imu, cmd_vel)
│   │   └── limo_gazebo_env.py     # Gymnasium env wrapping the above
│   │
│   ├── models/
│   │   ├── encoders.py            # Lidar/Pedestrian/IMU/Goal + MultiModalEncoder
│   │   ├── attention.py           # Self-attn + Cross-attn + TriSourceGate
│   │   ├── actor.py               # Squashed Gaussian SAC actor
│   │   ├── critic.py              # Twin Q-networks + soft update
│   │   └── policy.py              # End-to-end CmGapSacPolicy wrapper
│   │
│   ├── rl/                        # (next iteration)
│   │   ├── per_buffer.py          # SumTree + entropy-weighted PER
│   │   ├── sac.py                 # SAC update loop with auto-α
│   │   └── safety_shield.py       # OSQP-based CBF QP
│   │
│   └── training/                  # (next iteration)
│       ├── train.py               # full training loop
│       └── eval.py                # ablations A1..A5
│
├── launch/
│   └── gym_bringup.launch.py      # ROS 2 + Gazebo + tracker stack
│
├── worlds/                        # Gazebo Harmonic SDF worlds
│   ├── hospital.sdf
│   ├── warehouse.sdf
│   └── hybrid.sdf
│
└── test/                          # 74 unit tests, no ROS required
    ├── test_geometry.py
    ├── test_perception.py
    ├── test_encoders.py
    ├── test_attention_actor_critic.py
    └── test_reward.py
```

---

## 2. Installation

### 2.1 System requirements

- Kubuntu 24.04 (or any Ubuntu 24.04 derivative)
- ROS 2 Jazzy
- Gazebo Harmonic (installed via `ros-jazzy-ros-gz`)
- NVIDIA GPU + driver (RTX A5000 in the lab)
- CUDA 12.4 (matches the PyTorch wheel `cu124`)
- Python 3.12

### 2.2 Step-by-step setup

```bash
# 1. Source ROS 2
source /opt/ros/jazzy/setup.bash

# 2. Create or open your colcon workspace
mkdir -p ~/cm_gap_ws/src
cd ~/cm_gap_ws/src

# 3. Drop the package here
# (unzip cm_gap_sac_navigation_v02.zip into src/)

# 4. Install Python deps
pip install --break-system-packages \
    numpy scipy gymnasium pyyaml \
    torch --index-url https://download.pytorch.org/whl/cu124 \
    osqp

# 5. Install ROS 2 deps via rosdep
cd ~/cm_gap_ws
rosdep install --from-paths src --ignore-src -r -y

# 6. Build
colcon build --packages-select cm_gap_sac_navigation --symlink-install
source install/setup.bash
```

---

## 3. How to run, layer by layer

The code is built so each layer can be tested **independently**. Always go from
the bottom up: never debug the training loop before the unit tests pass.

### Layer 0: unit tests (no ROS, no Gazebo)

```bash
cd ~/cm_gap_ws/src/cm_gap_sac_navigation
PYTHONPATH=. python -m pytest test/ -v
```

Expected: `74 passed`. If any test fails, **stop** and fix that first.
Going further with broken unit tests is wasted time.

### Layer 1: encoder/attention/actor smoke test (no ROS)

```bash
PYTHONPATH=. python -c "
from cm_gap_sac_navigation.models.policy import build_policy_from_config
from cm_gap_sac_navigation.utils.config_loader import load_config
import torch

cfg = load_config('config/cm_gap_sac.yaml')
policy = build_policy_from_config(cfg)
out = policy(
    lidar=torch.randn(1, 36),
    pedestrians=torch.randn(1, 5, 5),
    ped_mask=torch.ones(1, 5),
    imu=torch.randn(1, 6),
    goal=torch.randn(1, 4),
)
print('action:', out['action'])
print('log_prob:', out['log_prob'])
print('entropy:', out['ped_attn_entropy'])
"
```

Expected: prints an action vector inside `[-0.3, 0.6] x [-1, 1]`, a finite
log-prob, and an entropy in `[0, 1]`. If torch is not installed correctly,
this is where it shows.

### Layer 2: launch Gazebo + ROS bridge

In one terminal:

```bash
ros2 launch ros_gz_sim gz_sim.launch.py \
    gz_args:="-r src/cm_gap_sac_navigation/worlds/warehouse.sdf"
```

In another:

```bash
ros2 launch cm_gap_sac_navigation gym_bringup.launch.py
```

Verify topics are flowing:

```bash
ros2 topic hz /scan      # ~10-20 Hz expected
ros2 topic hz /odom      # ~50 Hz
ros2 topic hz /imu       # ~50-100 Hz   <-- v0.2 addition
```

### Layer 3: env smoke test (ROS + Gazebo + Gym)

```bash
ros2 run cm_gap_sac_navigation gym_smoke_test --steps 50
```

Expected output:

```
[smoke] reset OK. obs keys: ['lidar', 'pedestrians', 'ped_mask', 'imu', 'goal']
[smoke] lidar shape: (36,), ped shape: (5, 5), imu shape: (6,), goal: [...]
[smoke] total reward: -X.XX
```

This is the integration boundary: if Layer 0 passes but Layer 3 doesn't,
the bug is in the ROS wiring, not in the algorithm.

### Layer 4: training (next iteration)

```bash
ros2 run cm_gap_sac_navigation train --config config/cm_gap_sac.yaml
tensorboard --logdir ./runs/cm_gap_sac
```

---

## 4. How to spot an error (diagnostic playbook)

A common pitfall when building a system this large is debugging at the wrong
layer. Use the **bottom-up rule**: a failure at layer N means the bug is at
layer N or below, never above.

### 4.1 Tests fail at Layer 0

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: cm_gap_sac_navigation` | wrong PYTHONPATH | `cd` to package root, `export PYTHONPATH=.` |
| `ModuleNotFoundError: torch` | torch not installed | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| `feature_dim mismatch` (encoders) | YAML mismatch between lidar_cnn, pedestrian_mlp, imu_mlp | all three must share the same `feature_dim` (default 64) |
| `divisible by n_heads` (attention) | bad config | feature_dim must be divisible by attention.n_heads |
| permutation invariance test fails | bug in PedestrianEncoder or masking | inspect `test_pedestrian_encoder_permutation_equivariance` |
| reward total ≠ sum of terms | floating-point ordering | compare with `pytest.approx`, not `==` |

### 4.2 Layer 1 (policy smoke test) fails

| Symptom | Likely cause |
|---|---|
| `CUDA out of memory` | batch too large; in smoke test should be impossible at B=1; check no stale processes (`nvidia-smi`) |
| `nan` in log_prob | log_std clipping not applied; check `log_std_min`/`log_std_max` in config |
| action outside bounds | actor scale/bias not registered as buffers; check `register_buffer` calls |
| entropy > 1 | normalization bug; `pedestrian_attention_entropy` clamps to `[0, 1]` |

### 4.3 Layer 2 (ROS topics) silent

```bash
ros2 topic list                     # are the topics there at all?
ros2 topic echo /imu --once         # is IMU publishing?
ros2 topic info /scan -v            # is anyone subscribing?
ros2 node list
ros2 node info /cm_gap_sac_gazebo_interface
```

| Symptom | Likely cause |
|---|---|
| `/imu` not listed | Gazebo IMU sensor not in the SDF, or `ros_gz_bridge` rule missing |
| `/scan` published but not in our env | QoS mismatch (BEST_EFFORT vs RELIABLE); we use BEST_EFFORT |
| `/cmd_vel` published but robot doesn't move | bridge direction wrong (`@` vs `]`), or differential drive plugin missing |
| services `/world/.../control` unreachable | wrong world name in YAML |

### 4.4 Layer 3 (env) hangs

```python
# In limo_gazebo_env.py, this line will hang if any sensor never publishes:
self._gz.wait_for_first_messages(timeout_s=10.0)
```

If it raises `RuntimeError: Gazebo did not produce /scan, /odom or /imu within 10 s`,
check **which** sensor is missing:

```bash
ros2 topic hz /scan      # if 0.0, lidar sensor is broken in the SDF
ros2 topic hz /odom      # if 0.0, diff drive plugin missing
ros2 topic hz /imu       # if 0.0, IMU sensor missing in the SDF
```

In the v0.2 we added IMU as a hard dependency: the env will refuse to start
without it. To run the env temporarily without an IMU sensor, set the
`/imu` topic to point to a constantly-zero publisher.

### 4.5 Layer 4 (training, next iteration)

| Symptom | Likely cause | What to check |
|---|---|---|
| reward stays at -5 forever | episode timeouts only; no progress | r_progress sign, prev_d_goal updates |
| reward jumps to +200 then crashes | terminal reward dominates; entropy collapse | log α (auto-temperature), should stabilize > 0.01 |
| critic loss diverges | learning rate too high, or τ too high | lr=3e-4, τ=0.005 (Haarnoja defaults) |
| ε intrusion frequent (d_min_ped < 0.45) | proxemic penalty not biting | r_prox magnitude relative to r_progress |
| collision rate stays high after 100k steps | CBF disabled or under-tuned | check cbf.enabled=true, r_safe=0.30 |
| hangs at training start | replay buffer fill blocking; environment not stepping | watch `ros2 topic hz /cmd_vel` |

### 4.6 Useful debug recipes

```bash
# Watch IMU values in real time
ros2 topic echo /imu --field linear_acceleration

# Watch the reward breakdown via the env logger node (next iteration)
ros2 topic echo /perception_log --no-arr

# Inspect the network visually (next iteration with TensorBoard)
tensorboard --logdir ./runs/cm_gap_sac --port 6006

# Profile a forward pass
PYTHONPATH=. python -m cProfile -o profile.out -m pytest \
    test/test_attention_actor_critic.py::test_policy_end_to_end_shapes
python -c "import pstats; pstats.Stats('profile.out').sort_stats('cumulative').print_stats(20)"
```

---

## 5. Module-by-module summary

| Module | Inputs | Outputs | Reference |
|---|---|---|---|
| `LidarEncoder` | `(B, n_beams)` | `(B, M, d)` | Tai et al. 2017 |
| `PedestrianEncoder` | `(B, K, F), (B, K)` | `(B, K, d)` | Zaheer et al. 2017 (Deep Sets) |
| `ImuEncoder` | `(B, 6)` | `(B, d)` | standard MLP |
| `GoalEncoder` | `(B, 4)` | `(B, d/2)` | standard MLP |
| `LidarSelfAttention` | `(B, M, d)` | `(B, d)` | Vaswani et al. 2017 |
| `PedestrianCrossAttention` | `(B, K, d), (B, K), goal, imu` | `(B, d), (B, K)` | Liu et al. 2020 (RGL), our IMU extension |
| `TriSourceGate` | three `(B, d)` + goal | three `(B, d)` summing to 1 | Srivastava et al. 2015 (Highway), softmax variant |
| `CrossModalAttention` | latent dict | `xi (B, d+d/2), alpha_ped (B, K)` | originality 1 |
| `SquashedGaussianActor` | `xi` | action, log_prob | Haarnoja et al. 2018 |
| `TwinCritic` | `(xi, action)` | `q1, q2` | Fujimoto et al. 2018 |
| `compute_reward` | obs, action, prev_action, prev_d, terminated, info | per-term dict | 6 terms, see §10.4 architecture doc |

---

## 6. Reproducibility

- Random seed: `training.seed` in YAML (default 42)
- Deterministic actor: `policy.act_from_obs(obs, deterministic=True)`
- Hyperparameters traceable to one of: Zhang 2025, Haarnoja 2018,
  Schaul 2016, Hall 1966, Lee 2020, Ng 1999. Each YAML field has an inline
  comment with its source.

---

## 7. Citation

If this code is used in another work, cite:

```
Amrani, S. (2026). CM-GAP_SAC: Cross-Modal Gated Attention Prioritized SAC
with Safety Shield for Socially-Aware Navigation. Master's thesis,
ESTIN Béjaïa, LITAN Laboratory.
```

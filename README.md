# CM-GAP-SAC

**Cross-Modal Gated-Attention Prioritized Soft Actor-Critic**

A safe multi-modal deep reinforcement learning policy for social navigation in dynamic indoor environments, deployed on the AgileX LIMO Pro.

<p align="center">
  <img src="assets/limo_robot.jpg" alt="AgileX LIMO Pro" width="45%"/>
  &nbsp;
  <img src="assets/hospital_env.png" alt="Simulated hospital environment in Gazebo Harmonic" width="45%"/>
</p>

## Demo

<!-- Drag & drop your .mp4 here in the GitHub web editor and delete this comment. GitHub uploads it and generates the embed automatically. -->

---

## Overview

Mobile robots operating in indoor spaces shared with people have to move safely among pedestrians who walk, stop, and change direction without warning. Reinforcement learning policies that learn to navigate rarely come with any guarantee that they will not collide.

CM-GAP-SAC is a local navigation policy trained with Soft Actor-Critic that fuses LiDAR, tracked pedestrians, inertial data and the goal through a gated cross-attention mechanism, and enforces hard safety through a Control Barrier Function shield solved as a quadratic program. It runs on ROS 2 Jazzy and is trained in a Gazebo simulation of a hospital-like world with dynamic pedestrian agents.

This repository contains the implementation of my engineering and Master's thesis, *Intelligent Perception and Navigation for Mobile Robots in Dynamic Environments*, defended at ESTIN (Béjaïa, Algeria) under the supervision of Dr. Ali Djenadi.

## Contributions

**1. Tri-source gated-softmax fusion with an IMU-augmented attention query.**
The policy fuses three sensing sources (LiDAR, tracked pedestrians, inertial data) together with the goal. A per-dimension gated softmax weighs the three contexts, and the cross-attention query over pedestrians is built from the concatenation of the goal and IMU latents, so social attention adapts to the robot's own physical state. This extends the LiDAR-only gated attention of GAP_SAC to a genuinely multi-modal state.

**2. Attention-entropy prioritized replay.**
Each stored transition is prioritized by a signed TD error scaled by a factor derived from the entropy of the pedestrian attention distribution. When the attention concentrates on a single pedestrian, the moment is treated as a critical decision and replayed more often. Clipping the TD error at zero keeps terminal collisions from dominating the buffer and pushing the critic toward pessimism. Setting the attention weight to zero reduces the scheme to standard PER.

**3. Control Barrier Function safety shield.**
A CBF-QP layer solved with OSQP filters the commanded linear velocity at every control step, projecting an unsafe action onto the closest safe one and keeping the robot inside a forward-invariant safe set. Safety radii are set per obstacle class (0.25 m static, 0.40 m pedestrians in training, 0.60 m on the real robot). The QP was feasible on every step of the full evaluation campaign.

## Architecture

```
                          Nav2 global planner (NavFn / Dijkstra, 1x per episode)
                                        │
   LiDAR 360°      RGB camera          IMU              Goal
   (720 beams)   YOLOv8n + DeepSORT   accel + gyro    (d, theta, v, w)
        │               │                │                │
     Conv1D        Deep Sets MLP        MLP              MLP
   (circular pad)  (perm-equivariant)
        │               │                │                │
   self-attention  cross-attention   ────┘                │
   (4 heads)       Q = f(goal, imu)                       │
        │          K, V = pedestrians                     │
        └───────────────┴──── tri-source softmax gate ────┘
                                        │
                              SAC actor (tanh-Gaussian)
                                        │
                              CBF-QP safety shield (OSQP)
                                        │
                              /cmd_vel (v, omega) @ 20 Hz
```

**Observation space**

| Modality | Shape | Content |
|---|---|---|
| `lidar` | (360,) | 720 beams downsampled to 360, clipped to [0.05, 6.0] m |
| `pedestrians` | (k_max, 5) | relative position, relative velocity, normalized track age |
| `ped_mask` | (k_max,) | 1 for a real slot, 0 for padding |
| `imu` | (6,) | linear acceleration and angular velocity, normalized |
| `goal` | (4,) | distance, bearing, own linear and angular velocity |

**Action space**: continuous, `v` in [-0.3, 0.5] m/s and `omega` in [-1.0, 1.0] rad/s.

## Results

Evaluated over 500 episodes per (method, scenario) pair, on the same spawn-goal seeds across all methods. Success rate in percent, Wilson 95% confidence intervals in brackets.

| Scenario | CM-GAP-SAC | Nav2 + DWB | SAC (ablation) |
|---|---|---|---|
| S1 static only | 83.2 [79.7, 86.2] | **94.0** [91.6, 95.8] | 7.6 [5.6, 10.3] |
| S2 3 peds @ 0.3 m/s | 79.6 [75.8, 82.9] | **87.0** [83.8, 89.7] | 5.2 [3.6, 7.5] |
| S4 5 peds @ 0.3 m/s | **77.8** [74.0, 81.2] | 77.2 [73.3, 80.7] | 4.8 [3.2, 7.0] |
| S6 7 peds @ 0.3 m/s | **79.4** [75.6, 82.7] | 72.0 [67.9, 75.8] | 4.6 [3.1, 6.8] |
| S7 7 peds @ 0.5 m/s | **73.3** [69.3, 77.0] | 69.6 [65.4, 73.5] | 4.2 [2.8, 6.3] |

Three findings:

- **Graceful degradation under density.** From 3 to 7 pedestrians, CM-GAP-SAC stays flat (79.6 to 79.4) while Nav2 loses fifteen points. The classical planner's advantage lives in exact static geometry and does not grow with scenario difficulty; the learned policy's advantage lives in dynamic and social behavior and does grow.
- **Asymmetric safety profile.** Across the four dynamic scenarios, CM-GAP-SAC averages 2.65% pedestrian collisions against 8.6% for Nav2, up to 6.6x fewer on S4. Nav2 treats a pedestrian as frozen at their last detected position; CM-GAP-SAC uses per-pedestrian velocities to steer around them early.
- **The three contributions matter jointly.** Removing the fusion, the attention-weighted replay and the shield at the same training budget drops the success rate by a factor of eleven to eighteen.

The safety shield intervenes on roughly 86% of steps in dynamic scenarios but with a mean correction magnitude of 0.03, small relative to the action bounds. The policy and the shield co-operate: the shield is active often, and its corrections are small.

The main remaining failure mode is doorway passages under fast pedestrian traffic. Intra-zone success stays between 85% and 91% across all scenarios, while cross-zone success (requiring at least one doorway crossing) drops sharply.

## Stack

- **ROS 2 Jazzy Jalisco** for the middleware, with **Nav2** as the global planner
- **Gazebo Harmonic** for the simulated hospital environment (16 m x 16 m, one corridor and four rooms)
- **PyTorch** for the encoders, the attention fusion and the SAC actor-critic
- **OSQP** for the CBF quadratic program
- **YOLOv8-nano** and **DeepSORT** for the pedestrian perception front-end
- **RViz2** for visualization
- Target hardware: **AgileX LIMO Pro** with an **NVIDIA Jetson Orin Nano**

## Getting started

### Prerequisites

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic
- Python 3.12, PyTorch with CUDA recommended

### Setup

```bash
git clone https://github.com/seryneamrani/CM-GAP-SAC.git
cd CM-GAP-SAC
pip install -r requirements.txt
```

### Launch the ROS 2 bridges

Nav2 must be running before training, since the environment queries its planner at each reset.

```bash
./run_bridges.sh
```

### Train

```bash
python src/train.py
```

### Evaluate

```bash
python src/evaluate.py --checkpoint <path/to/checkpoint.pt>
```

## Citation

```bibtex
@mastersthesis{amrani2026cmgapsac,
  author  = {Amrani, Seryne Fettouma},
  title   = {Intelligent Perception and Navigation for Mobile Robots
             in Dynamic Environments},
  school  = {École supérieure en Sciences et Technologies de l'Informatique
             et du Numérique (ESTIN)},
  year    = {2026},
  address = {Béjaïa, Algeria},
  note    = {Supervised by Dr. Ali Djenadi, LITAN laboratory}
}
```

## Acknowledgments

Thesis supervised by Dr. Ali Djenadi at ESTIN. This work builds on the gated-attention SAC design of Zhang et al. (2025), which it extends to a multi-modal state with a hard safety layer.

## Contact

**Seryne Fettouma Amrani**
[amrani.seryne@gmail.com](mailto:amrani.seryne@gmail.com)

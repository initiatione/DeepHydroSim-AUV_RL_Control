# DeepHydroSim — RL-based AUV Control

A sim-to-real reinforcement learning framework for Autonomous Underwater Vehicle (AUV)
navigation that bridges high-fidelity hydrodynamic simulation with scalable RL training.

**Core idea.** [Stonefish](https://github.com/patrykcieslak/stonefish) provides physically
accurate 6-DOF hydrodynamic data for a GIRONA500-class AUV. An MLP (`DeepHydroMLP`) learns
the nonlinear fluid dynamics from that data. The trained MLP is embedded as a custom plugin
inside MuJoCo, enabling fast RL training with realistic hydrodynamics — without paying the
runtime cost of a full CFD simulator.

<p align="center">
  <img src="assets/teacher_student_simulators.png" width="88%">
</p>
<p align="center"><sub>(a) <b>Teacher</b> — Stonefish (NED frame): high-fidelity data collection &nbsp;|&nbsp; (b) <b>Student</b> — MuJoCo (ENU frame): scalable RL training with learned hydrodynamics</sub></p>

> **This repository is a fork with a full reproduction pass.**
> On top of the released code it adds a corrected hydrodynamic plant, the
> `PhysicsFeature-MLP` from the paper, an evaluation/replay toolchain, and a written
> reproduction report. Every headline number below is reproducible from this tree —
> see [`REPRODUCTION.md`](REPRODUCTION.md) for the protocol, the commands and the
> results that could *not* be reproduced.

---

## Reproduction highlights

| Item | Paper | This repository |
|---|---|---|
| Hydro-MLP accuracy (pooled RMSE) | 11.81 | **2.25** (Baseline) / **2.06** (PhysFeature)¹ |
| Open-loop replay vs Stonefish (surge RMSE) | 1.784 → 0.996 m/s | **2.391 → 0.095 m/s**² |
| Stage 1 — point-to-point | 100% | **100%** (85 steps) |
| Stage 2 — sonar obstacle avoidance | 100% | **100% / 80% / 80%**³ |
| Stage 3 — A\*-guided domain navigation | 95% | **90%** (786 steps) |

¹ The paper's numbers come from a label bug: `Thruster::getSetpoint()` returns the target
rotor speed (rad/s), but it was summed as if it were thrust (N). Recording
`Thruster::getThrust()` and re-deriving the labels drops the error by 84%.
See `REPRODUCTION.md` §3.6.

² Nine 10 s segments (90 s total) replayed open-loop through MuJoCo; the two columns are
no-hydrodynamic-model vs. learned plant. `scripts/replay_table3.py`.

³ Curriculum levels 2 / 3 / 4 (level 4 = fixed 8-pillar slalom). The paper's 100% was
obtained on the old plant, whose surge force was hard-coded as `30|u| + 12u²` and reached
20.6 m/s at full thrust; the corrected plant tops out at 1.36 m/s, so the same task is
substantially harder.

### What did *not* reproduce

The paper argues that `PhysicsFeature-MLP` prevents closed-loop divergence caused by the
plain MLP using the acceleration channel as a shortcut for the nonlinear drag law. That
divergence turned out to be an implementation bug — the Kalman filter that estimates
acceleration was stepped with the physics timestep (`0.002 s`) instead of the control
period (`0.1 s`), inflating the estimate by 50×. With the timestep fixed, the plain
Baseline also reaches 100%. PhysFeature consequently shows no measurable advantage in
closed loop (95% vs. 100%, single seed) or in open-loop replay (0.095 vs. 0.095 m/s),
despite being 8.8% better offline.

---

## Architecture

```
┌─────────────────────┐      CSV logs       ┌─────────────────────┐
│   Stonefish C++     │ ─────────────────▶  │    hydro_MLP        │
│   (cpp_sim/)        │                     │  (train DeepHydro   │
│  GIRONA500 AUV      │                     │   MLP from data)    │
│  8-thruster layout  │                     └──────────┬──────────┘
│  HybridDataCollect  │                                │  .pth weights
└─────────────────────┘                                ▼
                                            ┌─────────────────────┐
                                            │    mujoco_sim       │
                                            │  HydroInference     │
                                            │  plugin (ENU↔NED)   │
                                            │  SAC · curriculum   │
                                            │  A* path planning   │
                                            └─────────────────────┘
```

The teacher simulates the **GIRONA500** (`sf::FeatherstoneRobot`, 8 thrusters); the student
uses `mujoco_sim/assets/yuyuan.xml`, which mirrors the same mass, inertia and thruster
layout.

---

## Repository structure

```
.
├── cpp_sim/                        # Stonefish data-collection application
│   └── src/
│       ├── MyAUVManager.{h,cpp}    # Simulation manager: robot, scene, data loop
│       ├── MyAUVApp.{h,cpp}        # OpenGL GUI / HUD overlay
│       ├── controllers/
│       │   ├── MotionController.h  # 6-DOF → 8-thruster mixing
│       │   └── MotionStrategy.h    # PID, MaxThrust, HybridDataCollection, OU noise
│       └── utils/
│           ├── DataCollector.h     # CSV logger (state + setpoints + true thrust)
│           └── analysing_damping.py
│
├── hydro_MLP/                      # MLP training pipeline
│   ├── config.yaml                 # Model / training configuration
│   └── src/
│       ├── model.py                # DeepHydroMLP, PhysFeatureMLP, build_model()
│       ├── dataset.py              # Contiguous-block split (no temporal leakage)
│       ├── data_process.py         # Inverse-dynamics label extraction
│       ├── prepare_data.py         # Merge raw Stonefish logs
│       ├── downsample.py           # 200 Hz → 50 Hz
│       ├── data_analysis.py
│       ├── train.py
│       ├── evaluate.py             # Aggregate metrics
│       └── evaluate_detail.py      # Per-DOF / per-sample analysis
│
├── mujoco_sim/                     # MuJoCo RL environment
│   ├── configs/
│   │   ├── config.yaml             # SAC hyperparameters, Hydra defaults
│   │   ├── env/default.yaml        # Legacy plant (setpoint-as-force labels)
│   │   ├── env/thrust.yaml         # Corrected plant (true-thrust labels)
│   │   └── task/
│   │       ├── stage1_navigate.yaml
│   │       ├── stage2_avoidance.yaml
│   │       └── stage3_domain_navigation.yaml
│   ├── scripts/
│   │   ├── train.py                # SAC training with curriculum & WandB
│   │   ├── eval_policy.py          # Deterministic N-episode evaluation
│   │   ├── bench_control.py        # Closed-loop tracking / terminal velocity
│   │   ├── replay_stonefish.py     # Replay recorded thrust through MuJoCo
│   │   ├── replay_table3.py        # Paper Table III protocol (9 × 10 s)
│   │   ├── evaluate.py             # Passive-viewer evaluation
│   │   ├── enjoy_rl.py             # Interactive rollout viewer
│   │   └── check_env.py
│   └── src/
│       ├── core/
│       │   ├── hydro_plugin.py     # MLP inference inside the MuJoCo step loop
│       │   ├── models.py           # Online mirror of the MLP architectures
│       │   ├── robot.py            # Kinematics, thruster mixing
│       │   ├── sensors.py          # IMU, DVL, sonar wrappers
│       │   └── scene_builder.py
│       ├── envs/
│       │   ├── auv_base_env.py     # Gymnasium base environment
│       │   └── tasks/
│       │       ├── navigation_task.py
│       │       ├── avoidance_task.py
│       │       └── domain_navigation_task.py
│       └── utils/
│           ├── kalman_filter.py    # KalmanFilter6D acceleration estimation
│           ├── astar_planner.py    # A* on occupancy grid → arc-length path
│           └── ...
│
├── REPRODUCTION.md                 # Full reproduction report (Chinese)
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 (CUDA recommended) |
| MuJoCo | ≥ 3.1 |
| Stonefish | Built from source — see [Stonefish docs](https://stonefish.readthedocs.io) |
| C++ compiler | GCC ≥ 11 or Clang ≥ 14, CMake ≥ 3.16 |

## Installation

```bash
git clone https://github.com/initiatione/DeepHydroSim-AUV_RL_Control.git
cd DeepHydroSim-AUV_RL_Control
pip install -r requirements.txt
```

Build the Stonefish data-collection application:

```bash
cd cpp_sim
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## Workflow

### Step 1 — Collect hydrodynamic data (Stonefish)

`HybridDataCollectionStrategy` runs nine deterministic manoeuvres (surge, heave, roll,
pitch, yaw, two circles, and three coupled patterns) followed by stochastic
Ornstein–Uhlenbeck exploration:

```bash
./cpp_sim/build/MyAUVApp
```

Logs are written to `cpp_sim/logs/GeneralMission_<timestamp>.csv`. Each row records both
the commanded setpoint (`M_*`) and the **true propeller thrust** (`T_*`, N).

`SF_MAX_TIME` stops the run at a fixed simulated time, which makes long collections
terminate deterministically instead of relying on an external timeout:

```bash
SF_MAX_TIME=17151 ./cpp_sim/build/MyAUVApp
```

### Step 2 — Train the hydrodynamic MLP

```bash
cd hydro_MLP
python src/prepare_data.py        # merge raw logs
python src/data_process.py        # inverse-dynamics labels (uses T_* when present)
python src/downsample.py          # 200 Hz → 50 Hz (4× decimation)
python src/train.py               # model.type = baseline | physfeature
python src/evaluate.py            # aggregate metrics
python src/evaluate_detail.py     # per-DOF analysis
```

Weights and scalers are written to `hydro_MLP/models/`.

**Model:** `DeepHydroMLP` — 12 → 128 → 256 → 128 → 6 (71 K parameters).
`PhysFeatureMLP` expands the 12-D input to a 33-D basis
`[ν, ν̇, ν|ν|, νᵢνⱼ]` before the same hourglass.

**Label generation:**

```
F_fluid = τ_prop − M_total · a̅
```

where `τ_prop` comes from the recorded thruster thrust, `M_total` is the rigid-body +
added-mass diagonal, and `a̅` is smoothed with Savitzky–Golay + finite differences.

### Step 3 — Train the RL agent (MuJoCo)

Copy the trained MLP into `mujoco_sim/weights/` (or `weights_thrust_baseline/`) and point
`configs/env/<profile>.yaml → weights` at it. **Model weights are not committed to this
repository** (`*.pth` and `*.pkl` are gitignored) — train them with Step 2 first.

```bash
cd mujoco_sim

# corrected plant (true-thrust weights)
python scripts/train.py env=thrust task=stage1_navigate
python scripts/train.py env=thrust task=stage2_avoidance
python scripts/train.py env=thrust task=stage3_domain_navigation

# legacy plant (released behaviour)
python scripts/train.py env=default task=stage1_navigate
```

Training uses **SAC** (Stable Baselines3) with `VecNormalize`, curriculum advancement on a
rolling success-rate threshold, WandB + TensorBoard logging, and periodic checkpoints.

**Warm-starting a stage is not just loading the weights.** A policy trained on a different
reward scale needs the inherited `VecNormalize` statistics and replay buffer, otherwise the
critic's TD targets jump by the ratio of the return scales and the actor is blown away:

```bash
python scripts/train.py env=thrust task=stage2_avoidance total_timesteps=600000 \
  hyperparams.learning_rate=1e-4 \
  pretrained.model_path=outputs/<stage1>/final_model.zip \
  pretrained.vecnorm_path=outputs/<stage1>/vec_normalize.pkl \
  pretrained.load_buffer=true \
  pretrained.buffer_path=outputs/<stage1>/final_replay_buffer.pkl
```

### Step 4 — Evaluate and replay

```bash
cd mujoco_sim

# deterministic evaluation of a checkpoint
python scripts/eval_policy.py task=stage2_avoidance env=thrust \
  pretrained.model_path=outputs/<run>/final_model.zip \
  +eval_episodes=20 +eval_stage=4

# plant fidelity: replay recorded Stonefish thrust through MuJoCo
python scripts/replay_stonefish.py

# paper Table III protocol (nine 10 s segments, five plant variants)
python scripts/replay_table3.py

# closed-loop tracking and terminal-velocity comparison
python scripts/bench_control.py
```

---

## Configuration

All hyperparameters are managed with **Hydra**. Key settings in `mujoco_sim/configs/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 5 000 000 | Total SAC training steps |
| `num_envs` | 1 | Parallel environments |
| `batch_size` | 512 | SAC mini-batch size |
| `learning_rate` | 3e-4 | Adam learning rate |
| `eval_freq` | 10 000 | Steps between evaluations |
| `gradient_steps` | `${num_envs}` | UTD ratio kept at 1:1 |

Override any parameter on the command line:

```bash
python scripts/train.py total_timesteps=10000000 hyperparams.batch_size=256
```

> **Note on evaluation noise.** `EvalCallback` uses Stable Baselines3's default of **5
> episodes** per evaluation. At a true success rate of 0.8 the standard error of a 5-episode
> estimate is 0.18, so swings between 0.60 and 1.00 are expected for an unchanged policy —
> and `best_model` can be selected by noise. Judge convergence with 20+ episodes
> (`eval_policy.py`) rather than the training curve.

### Plant profiles

| Profile | Weights | Notes |
|---|---|---|
| `env/default` | `weights/` | Released behaviour: rotor setpoints treated as thrust, hard-coded `30\|u\| + 12u²` surge override, `mlp_drag_scale=0.7` |
| `env/thrust` | `weights_thrust_baseline/` | Corrected: true-thrust labels, no override, `mlp_drag_scale=1.0`, `force_clip=2000`, `(v/v_max)²` extrapolation, neutral buoyancy at 1806.9 N |

---

## Curriculum stages

| Stage | Task | Key challenge |
|-------|------|---------------|
| 1 — `navigate` | Point-to-point (fixed start/goal) | Stable 6-DOF control |
| 2 — `avoidance` | Navigate with static obstacles | Collision penalty shaping, sonar-only |
| 3 — `domain_navigation` | A\*-guided path in a cluttered domain | Arc-length progress reward + PBRS |

Stage 3 uses an arc-length coordinate along the A\* waypoint path, giving a monotonically
increasing progress signal that avoids local reward plateaus.

<p align="center">
  <img src="assets/auv_navigation_demo.jpg" width="55%">
</p>
<p align="center"><sub>Stage 3 — GIRONA500 navigating a cluttered domain. Orange solid line: executed trajectory. Orange dashed line: A* reference path.</sub></p>

---

## Coordinate frames

The MLP is trained on Stonefish data logged in **NED** (North-East-Down); MuJoCo uses
**ENU** (East-North-Up). The `HydroInference` plugin applies a sign-flip mask before
inference and after the output:

```python
COORD_MASK = [+1, -1, -1, +1, -1, -1]   # [Surge, Sway, Heave, Roll, Pitch, Yaw]
```

---

## Ablation

Setting `mlp_drag_scale: 0.0` zeros the learned drag term while keeping added mass,
buoyancy and every other dynamic term intact — a pure rigid-body + added-mass baseline.
`simplified_mode: true` does the same via the added-mass-only branch.

---

## Known issues

- **Weights are not versioned.** `*.pth` / `*.pkl` are gitignored, so a fresh clone has no
  trained plant. Run Step 2, or fetch the weights separately.
- **`EvalCallback` evaluates 5 episodes.** See the note above; use `eval_policy.py` for
  reported numbers.
- **Stage 2 level 4 is sensitive to the episode budget.** Successful rollouts take
  589–1188 control steps, so a 650-step training budget truncates correct behaviour into a
  timeout penalty. Train and evaluate with `env.max_steps=1200`.
- **Single-environment, CPU-bound training.** `mj_step` dominates (~52% of a step) and the
  SAC update adds another ~65% of wall-clock; GPU utilisation stays near 4%. Parallelising
  environments alone yields little because the update cost scales with the sample count.

---

## Citation

If you use this work, please cite the original project:

```bibtex
@misc{deephyrosim2025,
  author  = {Ray Wu},
  title   = {AUV Deep RL Navigation — DeepHydroSim},
  year    = {2025},
  url     = {https://github.com/RayH-Wu/DeepHydroSim-AUV_RL_Control}
}
```

## Acknowledgements

- [Stonefish](https://github.com/patrykcieslak/stonefish) — high-fidelity underwater simulator
- [MuJoCo](https://mujoco.org) — physics engine for RL training
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — SAC implementation
- GIRONA500 AUV from the University of Girona

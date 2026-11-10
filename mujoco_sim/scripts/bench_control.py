"""Closed-loop control-accuracy benchmark for the two hydro configurations.

Compares the legacy (setpoint-as-force) plant against the thrust-corrected
plant with a *fixed* velocity-tracking PI controller.  The point is not to
tune the controller — it is to show how much the plant gain error costs a
controller that was designed against the legacy model.

Usage:
    cd mujoco_sim
    python scripts/bench_control.py
"""

import os
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.envs.auv_base_env import AUVGymEnv


def build(cfg_name: str):
    cfg_env = OmegaConf.load(os.path.join("configs/env", f"{cfg_name}.yaml"))
    cfg_task = OmegaConf.load("configs/task/stage1_navigate.yaml")
    cfg_env.randomization.pos_noise = [0.0, 0.0, 0.0]
    cfg_env.randomization.angle_noise = [0.0, 0.0, 0.0]
    env = AUVGymEnv(cfg_env, cfg_task)
    return env


def reset_level(env):
    env.reset(seed=0)
    env.data.qpos[:] = 0.0
    env.data.qpos[2] = 10.0
    env.data.qpos[3] = 1.0          # identity quaternion
    env.data.qvel[:] = 0.0
    env.hydro.reset()


def track_profile(env, profile, kp=2.0, ki=1.5, dt=0.1):
    """PI velocity tracking on surge; heading and depth held by P loops."""
    reset_level(env)
    integ = 0.0
    log = []
    for u_ref, n_steps in profile:
        for _ in range(n_steps):
            vel, rot = env.robot.get_body_state()
            err = u_ref - vel[0]
            integ = np.clip(integ + err * dt, -2.0, 2.0)
            surge = np.clip(kp * err + ki * integ, -1.0, 1.0)
            # Heading: body-x should stay aligned with world +x (yaw ~ 0)
            yaw = np.arctan2(rot[1, 0], rot[0, 0])
            yaw_rate = vel[5]
            yaw_cmd = np.clip(-1.5 * yaw - 0.8 * yaw_rate, -1.0, 1.0)
            # Depth hold at 10 m (positive heave = up)
            depth_err = 10.0 - env.data.qpos[2]
            heave = np.clip(0.6 * depth_err - 0.5 * vel[2], -1.0, 1.0)
            action = np.array([surge, 0.0, heave, 0.0, 0.0, yaw_cmd], dtype=np.float32)
            env.step(action)
            vel, _ = env.robot.get_body_state()
            log.append((u_ref, vel[0], env.data.qpos[2], surge))
    return np.array(log)


def summarise(name, log, profile, dt=0.1):
    ref, u, z, cmd = log[:, 0], log[:, 1], log[:, 2], log[:, 3]
    print(f"\n{name}")
    print(f"  speed tracking RMSE : {np.sqrt(np.mean((u - ref) ** 2)):.3f} m/s")
    print(f"  speed tracking MAE  : {np.mean(np.abs(u - ref)):.3f} m/s")
    print(f"  depth error (mean)  : {np.mean(np.abs(z - 10.0)):.3f} m")
    print(f"  mean |surge cmd|    : {np.mean(np.abs(cmd)):.3f}")
    print(f"  saturated fraction  : {np.mean(np.abs(cmd) >= 0.999) * 100:.1f} %")
    i = 0
    for lvl, n in profile:
        seg = u[i:i + n]
        tail = seg[int(0.6 * len(seg)):]
        print(f"    ref={lvl:4.2f}  settled u={tail.mean():5.2f}  "
              f"overshoot={max(0.0, seg.max() - lvl):5.2f}  "
              f"cmd={np.mean(np.abs(cmd[i:i + n])):.2f}")
        i += n
    print(f"  peak speed          : {u.max():.2f} m/s")


def top_speed(env, seconds=20.0):
    """Open-loop: hold full surge (with depth hold) and report terminal speed."""
    reset_level(env)
    u = 0.0
    for _ in range(int(seconds / 0.1)):
        vel, _ = env.robot.get_body_state()
        depth_err = 10.0 - env.data.qpos[2]
        heave = np.clip(0.6 * depth_err - 0.5 * vel[2], -1.0, 1.0)
        env.step(np.array([1.0, 0, heave, 0, 0, 0], dtype=np.float32))
        u = env.robot.get_body_state()[0][0]
    return u


def main():
    profile = [(0.3, 100), (0.6, 100), (1.0, 100), (0.5, 100), (0.0, 100)]
    for name in ("default", "thrust"):
        env = build(name)
        log = track_profile(env, profile)
        label = ("legacy plant (default.yaml)" if name == "default"
                 else "thrust-corrected plant (thrust.yaml)")
        summarise(label, log, profile)
        print(f"  terminal speed @ full surge (20 s): {top_speed(env):.2f} m/s")
        env.close()


if __name__ == "__main__":
    main()

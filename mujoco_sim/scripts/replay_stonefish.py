"""Replay recorded Stonefish thrust through the MuJoCo plant.

The Stonefish log contains the *actual* per-thruster thrust (T_*, N) and the
resulting body velocity.  Feeding that thrust into MuJoCo makes the plant
directly comparable: whichever hydro model reproduces the recorded surge
response is the one that matches the source simulator.

Only the surge axis is scored, and only level-attitude segments are used, so
the result is not contaminated by the (slightly negative) buoyancy trim of the
MuJoCo model or by the vertical thruster sign convention.

Usage:
    cd mujoco_sim
    python scripts/replay_stonefish.py [--segments 40] [--seconds 8]
"""

import argparse
import os
import sys

import mujoco
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.envs.auv_base_env import AUVGymEnv

# Log order -> MuJoCo actuator order (t0_hfr, t1_hfl, t2_hrr, t3_hrl,
# t4_vfr, t5_vfl, t6_vrr, t7_vrl)
LOG_TO_MJC = [1, 0, 3, 2, 5, 4, 7, 6]
HORIZONTAL = slice(0, 4)


def load_segments(csv_path, seconds, n_seg, seed=0):
    """Return the frame, segment start rows (200 Hz indexing) and span.

    The log is 200 Hz; the MuJoCo control period is frame_skip * timestep =
    0.1 s, so every 20th log row is one control step.
    """
    df = pd.read_csv(csv_path)
    need = ["u(m/s)", "Roll(rad)", "Pitch(rad)", "T_FL", "T_FR", "T_RL", "T_RR"]
    df = df.dropna(subset=need)
    n = int(seconds * 200)
    level = (df["Roll(rad)"].abs() < 0.15) & (df["Pitch(rad)"].abs() < 0.15)
    starts = np.where(level.values[: len(df) - n])[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(starts, size=min(n_seg, len(starts)), replace=False)
    return df, sorted(pick), n


def replay(env, df, start, n):
    seg = df.iloc[start:start + n:20]          # 200 Hz -> 10 Hz control
    env.reset(seed=0)
    env.data.qpos[:] = 0.0
    env.data.qpos[2] = 10.0
    env.data.qpos[3] = 1.0
    env.data.qvel[:] = 0.0
    env.data.qvel[0] = seg["u(m/s)"].iloc[0]
    env.hydro.reset()

    thrust = seg[["T_FL", "T_FR", "T_RL", "T_RR",
                  "T_VFL", "T_VFR", "T_VRL", "T_VRR"]].values[:, LOG_TO_MJC]
    ctrl = np.clip(thrust / 155.0, -1.0, 1.0)

    pred = []
    fs = int(env.cfg_env.frame_skip)
    for k in range(len(ctrl)):
        env.robot.set_thrusters(ctrl[k])
        env.hydro.apply_hydrodynamics(env.robot)
        cache = env.data.xfrc_applied.copy()
        for _ in range(fs):
            env.data.xfrc_applied = cache
            mujoco.mj_step(env.model, env.data)
        pred.append(env.robot.get_body_state()[0][0])
    pred = np.array(pred)
    ref = seg["u(m/s)"].values
    # pred[i] is the state 0.1 s after applying ctrl[i] -> compare with ref[i+1]
    return pred[:-1], ref[1:len(pred)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../hydro_MLP/data/phy_processed/"
                                    "mission_log_processed_thrust.csv")
    ap.add_argument("--segments", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    df, starts, n = load_segments(args.csv, args.seconds, args.segments)
    print(f"{len(starts)} level segments x {args.seconds:.0f} s from {args.csv}")

    for name in ("default", "thrust"):
        cfg_env = OmegaConf.load(os.path.join("configs/env", f"{name}.yaml"))
        cfg_task = OmegaConf.load("configs/task/stage1_navigate.yaml")
        cfg_env.randomization.pos_noise = [0.0, 0.0, 0.0]
        cfg_env.randomization.angle_noise = [0.0, 0.0, 0.0]
        env = AUVGymEnv(cfg_env, cfg_task)

        errs, final = [], []
        for s in starts:
            pred, ref = replay(env, df, s, n)
            errs.append(np.sqrt(np.mean((pred - ref) ** 2)))
            final.append(abs(pred[-1] - ref[-1]))
        env.close()
        label = "legacy plant " if name == "default" else "thrust plant "
        print(f"{label} surge RMSE {np.mean(errs):6.3f} m/s   "
              f"median {np.median(errs):6.3f}   "
              f"final |du| {np.mean(final):6.3f} m/s")


if __name__ == "__main__":
    main()

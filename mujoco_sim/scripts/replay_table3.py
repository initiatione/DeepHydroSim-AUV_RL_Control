"""Open-loop replay of the paper's Table III protocol.

Table III scores the surge-velocity error of the learned hydrodynamic plant
when the *recorded* Stonefish thrust is replayed through MuJoCo.  The paper
reports 90 s of command (nine 10 s segments); we build the same shape from the
nine deterministic manoeuvres of ``HybridDataCollectionStrategy``.

Each segment is replayed independently from the recorded state, so the metric
is a pure one-step-ahead rollout of the plant, not a free-running simulation.

Usage:
    cd mujoco_sim
    python scripts/replay_table3.py [--csv PATH] [--models baseline_thrust]
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

# Log order -> MuJoCo actuator order
LOG_TO_MJC = [1, 0, 3, 2, 5, 4, 7, 6]

T_NAMES = ["T_FL", "T_FR", "T_RL", "T_RR", "T_VFL", "T_VFR", "T_VRL", "T_VRR"]
VEL_NAMES = ["u(m/s)", "v(m/s)", "w(m/s)", "p(rad/s)", "q(rad/s)", "r(rad/s)"]

# Deterministic manoeuvre windows in simulation time (s).  Derived from
# HybridDataCollectionStrategy: 5 s settle, then 180 s per axis / 180 s per
# circle / 270 s per coupled manoeuvre, with a 5 s rest between each.
MANOEUVRES = [
    ("Surge",     5,    185),
    ("Heave",     190,  370),
    ("Roll",      375,  555),
    ("Pitch",     560,  740),
    ("Yaw",       745,  925),
    ("CircleCW",  930,  1110),
    ("VertWave",  1300, 1570),
    ("Corkscrew", 1575, 1845),
    ("DriftTurn", 1850, 2120),
]

SEG_SECONDS = 10.0
LOG_HZ = 200.0

MODELS = {
    "no_hydro":        dict(model_type="baseline",   mlp="",                                         drag_scale=0.0),
    "baseline_old":    dict(model_type="baseline",   mlp="weights/best_hydro_model.pth",              drag_scale=1.0),
    "baseline_thrust": dict(model_type="baseline",   mlp="weights_thrust_baseline/best_hydro_model.pth",    drag_scale=1.0),
    "phys_old":        dict(model_type="physfeature", mlp="weights_physfeature/best_hydro_model.pth",      drag_scale=1.0),
    "phys_thrust":     dict(model_type="physfeature", mlp="weights_thrust_physfeature/best_hydro_model.pth", drag_scale=1.0),
}


def load_log(csv_path, max_time):
    cols = ["Time(s)"] + VEL_NAMES + T_NAMES
    df = pd.read_csv(csv_path, usecols=cols, nrows=int(max_time * LOG_HZ) + 10)
    return df.dropna(subset=["Time(s)"] + VEL_NAMES + T_NAMES)


def pick_windows(df):
    """Largest-surge 10 s window inside each manoeuvre."""
    n = int(SEG_SECONDS * LOG_HZ)
    out = []
    for name, a, b in MANOEUVRES:
        seg = df[(df["Time(s)"] >= a) & (df["Time(s)"] < b)]
        best, best_std = None, -1.0
        for start in range(0, max(len(seg) - n, 1), int(1.0 * LOG_HZ)):
            w = seg.iloc[start:start + n]
            s = w["u(m/s)"].std()
            if s > best_std:
                best, best_std = w, s
        out.append((name, best, best_std))
    return out


def build_env(weights_dir, model_type, drag_scale):
    cfg = OmegaConf.load("configs/env/thrust.yaml")
    task = OmegaConf.load("configs/task/stage1_navigate.yaml")
    cfg.randomization.pos_noise = [0.0, 0.0, 0.0]
    cfg.randomization.angle_noise = [0.0, 0.0, 0.0]
    cfg.mlp_drag_scale = drag_scale
    if weights_dir:
        cfg.weights.model_type = model_type
        cfg.weights.mlp = os.path.join(weights_dir, "best_hydro_model.pth")
        cfg.weights.scaler_x = os.path.join(weights_dir, "scaler_X.pkl")
        cfg.weights.scaler_y = os.path.join(weights_dir, "scaler_Y.pkl")
    return AUVGymEnv(cfg, task)


def replay(env, seg):
    seg = seg.iloc[::int(LOG_HZ / 10.0)]        # 200 Hz -> 10 Hz control
    env.reset(seed=0)
    env.data.qpos[:] = 0.0
    env.data.qpos[2] = 20.0
    env.data.qpos[3] = 1.0                      # level attitude
    env.data.qvel[:] = 0.0
    env.data.qvel[0] = seg["u(m/s)"].iloc[0]
    env.hydro.reset()

    thrust = seg[T_NAMES].values[:, LOG_TO_MJC]
    ctrl = np.clip(thrust / 155.0, -1.0, 1.0)
    sat = float(np.mean(np.abs(thrust) > 155.0))

    fs = int(env.cfg_env.frame_skip)
    pred = []
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
    return pred[:-1], ref[1:len(pred)], sat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../hydro_MLP/data/phy_processed/"
                                    "mission_log_processed_Physics_thrust.csv")
    ap.add_argument("--models", default="all")
    args = ap.parse_args()

    max_time = max(b for _, _, b in MANOEUVRES)
    df = load_log(args.csv, max_time)
    windows = pick_windows(df)
    print(f"{len(windows)} segments x {SEG_SECONDS:.0f} s = "
          f"{len(windows) * SEG_SECONDS:.0f} s from {os.path.basename(args.csv)}\n")

    names = list(MODELS) if args.models == "all" else args.models.split(",")
    results = {}
    for key in names:
        spec = MODELS[key]
        env = build_env(spec["mlp"].rsplit("/", 1)[0] if spec["mlp"] else "",
                        spec["model_type"], spec["drag_scale"])
        row = {}
        for name, seg, _ in windows:
            pred, ref, sat = replay(env, seg)
            row[name] = float(np.sqrt(np.mean((pred - ref) ** 2)))
        results[key] = row
        print(f"[{key}] done")

    hdr = f"{'segment':11s} " + " ".join(f"{k:>16s}" for k in names)
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, _, ustd in windows:
        line = f"{name:11s} " + " ".join(f"{results[k][name]:16.3f}" for k in names)
        print(line)
    print("-" * len(hdr))
    pooled = {k: float(np.sqrt(np.mean([v ** 2 for v in results[k].values()])))
              for k in names}
    print(f"{'pooled':11s} " + " ".join(f"{pooled[k]:16.3f}" for k in names))
    print(f"\n(units m/s; reference u std per segment: "
          + ", ".join(f"{n}={s:.2f}" for n, _, s in windows) + ")")


if __name__ == "__main__":
    main()

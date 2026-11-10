"""
Headless policy evaluation — runs a trained SAC policy for N deterministic
episodes and reports success / collision / timeout statistics.

Unlike ``enjoy_rl.py`` this needs no display, so it can be used to produce the
numbers quoted in REPRODUCTION.md.

Usage:
    cd mujoco_sim
    python scripts/eval_policy.py task=stage3_domain_navigation \
        pretrained.model_path=outputs/.../final_model.zip \
        pretrained.vecnorm_path=outputs/.../vec_normalize.pkl \
        +eval_episodes=20 +eval_stage=4
"""

import os
import sys

import hydra
import hydra.utils
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.envs.auv_base_env import AUVGymEnv


def _abs(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(hydra.utils.get_original_cwd(), path)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    model_path = _abs(cfg.pretrained.model_path)
    vecnorm_path = _abs(cfg.pretrained.vecnorm_path)
    episodes = int(cfg.get("eval_episodes", 20))
    stage = cfg.get("eval_stage", None)
    if stage is None:
        stage = cfg.task.curriculum.initial_stage if "curriculum" in cfg.task else 0

    if not model_path or not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Observation normalisation is off in this project, so the VecNormalize
    # file only carries reward statistics and inference does not need it.
    # Assert that instead of silently ignoring a file the user passed.
    if vecnorm_path and os.path.exists(vecnorm_path):
        import pickle
        with open(vecnorm_path, "rb") as fh:
            vn = pickle.load(fh)
        if getattr(vn, "norm_obs", False):
            print("WARNING: this VecNormalize normalises observations; "
                  "raw-env evaluation will differ from training.")

    env = AUVGymEnv(cfg.env, cfg.task)
    env.set_stage(int(stage))
    model = SAC.load(model_path, device=cfg.device)

    results = {"success": 0, "collision": 0, "out_of_bounds": 0, "timeout": 0}
    rewards, steps, final_dists, ctes = [], [], [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        total, n = 0.0, 0
        info = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            n += 1
            if terminated or truncated:
                break
        reason = info.get("termination_reason") or "timeout"
        results[reason] = results.get(reason, 0) + 1
        rewards.append(total)
        steps.append(n)
        body_id = env.model.body("yuyuan").id
        final_dists.append(float(np.linalg.norm(env.data.xpos[body_id] - env.target_pos)))
        if "state/cross_track_err" in info:
            ctes.append(float(info["state/cross_track_err"]))
        print(f"  ep {ep + 1:3d}/{episodes}: {reason:<14s} steps={n:4d} "
              f"reward={total:9.2f} dist={final_dists[-1]:5.2f}")

    print("\n=== summary ===")
    print(OmegaConf.to_yaml({"stage": int(stage), "episodes": episodes}))
    for k, v in results.items():
        print(f"  {k:<14s} {v:3d}/{episodes}  ({100.0 * v / episodes:5.1f}%)")
    print(f"  mean reward    {np.mean(rewards):9.2f} +/- {np.std(rewards):.2f}")
    print(f"  mean steps     {np.mean(steps):9.1f}")
    print(f"  mean final dist{np.mean(final_dists):9.2f} m")
    if ctes:
        print(f"  mean final CTE {np.mean(ctes):9.2f} m")
    env.close()


if __name__ == "__main__":
    main()

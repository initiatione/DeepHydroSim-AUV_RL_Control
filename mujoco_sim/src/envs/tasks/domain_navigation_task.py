import numpy as np
from .base_task import BaseTask
from src.utils.astar_planner import AStarPlanner


class DomainNavigationTask(BaseTask):
    """
    Stage 3 task: A*-guided navigation with obstacle avoidance.

    The task uses a 1D progress coordinate along the planned path for reward
    shaping (monotonic progress prevents reward hacking from backtracking) and
    exposes a cross-track-error (CTE) vector in the observation so the agent
    can correct lateral drift from the A* route.
    """

    def __init__(self, config):
        super().__init__(config)
        self.cfg = self.config.reward_weights
        self.goals = self.config.goals

        # Reward weight scalars with safe defaults
        self.w_cte = getattr(self.cfg, 'w_cte', 15.0)
        self.w_collision_step = getattr(self.cfg, 'w_collision_step', 25.0)
        self.w_final_bonus = getattr(self.cfg, 'w_final_bonus', 500.0)
        # Distance-potential scale (same structure as stage 1/2)
        self.phi_dist = getattr(self.cfg, 'phi_dist', 500.0)
        self.last_potential = 0.0

        # Safety thresholds (read from config, fall back to safe defaults)
        safety_cfg = getattr(self.config, 'safety', None)
        self.warning_distance = getattr(safety_cfg, 'warning_distance', 4.0) if safety_cfg else 4.0
        self.critical_distance = getattr(safety_cfg, 'critical_distance', 0.4) if safety_cfg else 0.4

        # Water-column bounds. Without them the policy can fly above the
        # surface (z > 15) and farm a long timeout with a huge cross-track
        # error, which makes SAC collapse instead of following the path.
        self.z_min = getattr(getattr(config, 'safety', None), 'z_min', 0.5)
        self.z_max = getattr(getattr(config, 'safety', None), 'z_max', 14.5)
        self.w_out_of_bounds = getattr(getattr(config, 'safety', None), 'w_out_of_bounds', 500.0)
        self.max_cte_cost = getattr(self.cfg, 'max_cte_cost', 20.0)

        # obs = [pos(3), vel(3), gyro(3), quat(4), gravity(3), depth(1),
        #        sonar(15), alt(1), accel(3), cte(3)] = 39
        self.obs_dim = 39

        # debug=False: the planner runs on every reset, so writing the debug
        # map each episode would dominate the training loop's wall time.
        self.planner = AStarPlanner(resolution=0.05, safe_margin=1.7, debug=False)

        # Path state — fully initialised here, reset() overwrites each episode
        self.waypoints = []
        self.wp_cum_dists = [0.0]
        self.total_path_length = 0.0
        self.current_seg_idx = 0

        # Progress tracking (monotonic: never decreases within an episode)
        self.max_s_reached = 0.0

        # Smoothed look-ahead point for heading alignment
        self.smoothed_lookahead_pt = None
        self.lookahead_distance = 0.7
        self.search_window = 3

        # Per-step state
        self.last_action = None
        self.current_is_collision = False

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def get_obs_dim(self):
        return self.obs_dim

    def reset(self, env):
        self.last_action = np.zeros(env.action_space.shape)
        self.current_is_collision = False
        self.max_s_reached = 0.0
        self.smoothed_lookahead_pt = None

        env.target_pos = getattr(env, 'fixed_target_pos', np.array([18.0, 0.0, 10.0]))
        active_obstacles = env.scene_builder.get_active_obstacles()
        start_pos = env.data.xpos[env.robot.body_id].copy()

        # Plan A* path; fall back to straight line if planner fails
        self.waypoints = self.planner.plan(start_pos, env.target_pos, active_obstacles)
        if not self.waypoints or len(self.waypoints) == 0:
            self.waypoints = [start_pos, env.target_pos]

        # Pre-compute cumulative arc-length along waypoints (1-D S coordinate)
        self.wp_cum_dists = [0.0]
        for i in range(1, len(self.waypoints)):
            seg_len = np.linalg.norm(self.waypoints[i] - self.waypoints[i - 1])
            self.wp_cum_dists.append(self.wp_cum_dists[-1] + seg_len)
        self.total_path_length = self.wp_cum_dists[-1] if len(self.waypoints) > 1 else 0.0

        self.current_seg_idx = 0
        self._get_projection_status(start_pos)
        self.last_potential = -(np.linalg.norm(start_pos - env.target_pos)
                                / self.goals.max_dist) * self.phi_dist

        # Force-initialise smooth look-ahead so the first frame has no lag
        self.smoothed_lookahead_pt = self._map_1d_s_to_3d_pos(
            min(self.max_s_reached + self.lookahead_distance, self.total_path_length)
        )

    # ------------------------------------------------------------------
    # Path utilities
    # ------------------------------------------------------------------

    def _get_projection_status(self, pos):
        """Project AUV onto the planned path, returning 1-D progress S and lateral CTE."""
        if len(self.waypoints) < 2:
            return 0.0, 0.0

        end_idx = min(self.current_seg_idx + self.search_window, len(self.waypoints) - 1)
        min_dist = float('inf')
        best_i = self.current_seg_idx
        best_t = 0.0

        for i in range(self.current_seg_idx, end_idx):
            A = self.waypoints[i]
            B = self.waypoints[i + 1]
            AB = B - A
            len_sq = np.dot(AB, AB)
            if len_sq < 1e-6:
                continue
            t = np.clip(np.dot(pos - A, AB) / len_sq, 0.0, 1.0)
            proj = A + t * AB
            d = np.linalg.norm(pos - proj)
            if d < min_dist:
                min_dist = d
                best_i = i
                best_t = t

        if best_i > self.current_seg_idx:
            self.current_seg_idx = best_i

        seg_len = np.linalg.norm(self.waypoints[best_i + 1] - self.waypoints[best_i])
        s_current = self.wp_cum_dists[best_i] + best_t * seg_len
        self.max_s_reached = max(self.max_s_reached, s_current)
        return s_current, min_dist

    def _map_1d_s_to_3d_pos(self, s_target):
        """Map a 1-D arc-length coordinate back to a 3-D world position."""
        if s_target <= 0.0:
            return self.waypoints[0]
        if s_target >= self.total_path_length:
            return self.waypoints[-1]
        for i in range(len(self.waypoints) - 1):
            if self.wp_cum_dists[i] <= s_target <= self.wp_cum_dists[i + 1] + 1e-5:
                seg_start_s = self.wp_cum_dists[i]
                seg_len = self.wp_cum_dists[i + 1] - seg_start_s
                if seg_len < 1e-6:
                    return self.waypoints[i]
                t = (s_target - seg_start_s) / seg_len
                return self.waypoints[i] + t * (self.waypoints[i + 1] - self.waypoints[i])
        return self.waypoints[-1]

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(self, env, action, obs):
        raw = env.sensors.get_raw_data()
        body_id = env.model.body('yuyuan').id
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        pos = env.data.xpos[body_id].copy()

        # 1. Potential-based distance shaping toward the goal.
        # The released reward used delta-S along the A* path gated by
        # exp(-(CTE/tol)^2). That signal vanishes as soon as the policy drifts
        # more than a few metres off the route: measured on a 50k checkpoint the
        # agent sat 8-13 m off-path, tilted ~57 deg, earning 149 points of
        # progress against 9805 of CTE and 9069 of attitude cost. Nothing in the
        # reward then points back. The distance potential is non-zero
        # everywhere, so there is always a gradient, and the CTE term below
        # still biases the agent onto the planned (obstacle-free) route.
        dist_to_final = np.linalg.norm(pos - env.target_pos)
        current_potential = -(dist_to_final / self.goals.max_dist) * self.phi_dist
        reward_progress = (current_potential - self.last_potential) * 10.0
        self.last_potential = current_potential

        # 2. Cross-track error penalty (Huber-like: quadratic near path, linear far)
        # Capped: an unbounded CTE cost lets a lost policy accumulate -8e4 over
        # one timeout episode, which destroys the critic's value scale.
        s_current, cross_track_error = self._get_projection_status(pos)
        if cross_track_error < 1.0:
            cost_cte = self.w_cte * (cross_track_error ** 2)
        else:
            cost_cte = self.w_cte * (2.0 * cross_track_error - 1.0)
        cost_cte = min(cost_cte, self.max_cte_cost)

        # 3. Attitude cost: penalise non-horizontal orientation
        z_x, z_y = rot_mat[0, 2], rot_mat[1, 2]
        cost_pitch_roll = (z_x ** 2 + z_y ** 2) * getattr(self.cfg, 'w_pitch_roll', 20.0)

        # 4. Heading alignment with the smoothed look-ahead point
        body_x = rot_mat[:, 0]
        vec_to_lookahead = self.smoothed_lookahead_pt - pos
        dist_lookahead = np.linalg.norm(vec_to_lookahead)
        desired_x = vec_to_lookahead / (dist_lookahead + 1e-6)
        align_cos = np.dot(body_x, desired_x)
        reward_align = align_cos * getattr(self.cfg, 'w_align', 2.0)

        # 5. Dynamics costs
        local_vel = raw.get('dvl', np.zeros(3))
        v_sway, v_heave = local_vel[1], local_vel[2]
        cost_sway_heave = getattr(self.cfg, 'w_sway_vel', 2.0) * (v_sway ** 2) + \
                          getattr(self.cfg, 'w_heave_vel', 2.0) * (v_heave ** 2)

        v_surge = local_vel[0]
        v_surge_excess = max(0.0, v_surge - 1.2)
        cost_overspeed = 10.0 * (v_surge_excess ** 2)

        cost_action = getattr(self.cfg, 'w_accel', 0.15) * np.sum(np.square(action))
        cost_smooth = getattr(self.cfg, 'w_delta_accel', 0.5) * \
                      np.sum(np.square(action - self.last_action))

        # 6. Obstacle penalty
        sonar_dists = raw.get('sonar', np.ones(15) * 12.0)
        min_sonar_dist = np.min(sonar_dists)
        reward_obstacle_penalty = 0.0
        self.current_is_collision = False

        if min_sonar_dist < self.critical_distance:
            self.current_is_collision = True
            penalty_factor = (self.critical_distance - min_sonar_dist) / self.critical_distance
            reward_obstacle_penalty = penalty_factor * self.w_collision_step
        elif min_sonar_dist < self.warning_distance:
            scale = (self.warning_distance - min_sonar_dist) / (self.warning_distance - self.critical_distance)
            reward_obstacle_penalty = 5.0 * (scale ** 2)

        # 7. Success / terminal reward
        reward_success = 0.0
        is_success = False

        if dist_to_final < self.goals.success_dist:
            is_success = True
            reward_success = self.cfg.success + self.w_final_bonus
            time_penalty_applied = 0.0
        else:
            time_penalty_applied = self.cfg.time_penalty

        # 8. Water-column bounds. Leaving the water column terminates the
        # episode in is_done(); charge a flat terminal penalty here so the
        # critic sees the cost on the transition that leaves the bounds.
        is_out_of_bounds = bool(pos[2] < self.z_min or pos[2] > self.z_max)
        cost_out_of_bounds = self.w_out_of_bounds if is_out_of_bounds else 0.0

        total_reward = (
            reward_progress
            + reward_align
            + reward_success
            - cost_cte
            - cost_pitch_roll
            - reward_obstacle_penalty
            - cost_sway_heave
            - cost_overspeed
            - cost_action
            - cost_smooth
            - time_penalty_applied
            - cost_out_of_bounds
        )

        self.last_action = action.copy()

        info = {
            "rew/progress": reward_progress,
            "rew/align": reward_align,
            "rew/cte_penalty": -cost_cte,
            "rew/pitch_roll_cost": -cost_pitch_roll,
            "rew/obstacle_penalty": -reward_obstacle_penalty,
            "rew/overspeed": -cost_overspeed,
            "rew/sway_heave": -cost_sway_heave,
            "rew/action": -cost_action,
            "rew/smooth": -cost_smooth,
            "rew/time": -time_penalty_applied,
            "rew/out_of_bounds": -cost_out_of_bounds,
            "state/s_current": s_current,
            "state/cross_track_err": cross_track_error,
            "state/min_sonar_dist": min_sonar_dist,
            "is_success": float(is_success),
            "is_collision": float(self.current_is_collision),
            "is_out_of_bounds": float(is_out_of_bounds),
        }

        return total_reward, is_success, info

    def is_done(self, env, current_step, max_steps):
        body_id = env.model.body('yuyuan').id
        pos = env.data.xpos[body_id].copy()
        if np.linalg.norm(pos - env.target_pos) < self.goals.success_dist:
            return True, "success"
        if self.current_is_collision:
            return True, "collision"
        if pos[2] < self.z_min or pos[2] > self.z_max:
            return True, "out_of_bounds"
        if current_step >= max_steps:
            return True, "timeout"
        return False, None

    def get_obs(self, env):
        raw = env.sensors.get_raw_data()
        pos_world = env.data.xpos[env.robot.body_id]
        rot_mat = env.data.xmat[env.robot.body_id].reshape(3, 3)

        # Update 1-D progress and smoothed look-ahead point
        s_curr, _ = self._get_projection_status(pos_world)
        target_s = min(self.max_s_reached + self.lookahead_distance, self.total_path_length)
        raw_lookahead_pt = self._map_1d_s_to_3d_pos(target_s)

        alpha = 0.5
        if self.smoothed_lookahead_pt is None:
            self.smoothed_lookahead_pt = raw_lookahead_pt.copy()
        self.smoothed_lookahead_pt = (1.0 - alpha) * self.smoothed_lookahead_pt + alpha * raw_lookahead_pt

        # Observation components
        target_vec_body = rot_mat.T @ (self.smoothed_lookahead_pt - pos_world)
        gravity_body = rot_mat.T @ np.array([0.0, 0.0, -1.0])

        # The lookahead point is always ~0.7 m ahead, so normalising it by
        # goals.max_dist (45 m) squeezes the only goal-direction cue down to
        # |obs| ~ 0.017 — 20-50x smaller than quat/gravity/accel, and the policy
        # cannot see it (measured: stage 0 success stuck at 1.7%). A fixed 3 m
        # scale keeps the cue at O(0.3) and still saturates when far off-path.
        obs_pos = np.clip(target_vec_body / 3.0, -1.0, 1.0)
        obs_vel = np.clip(raw['dvl'] / 2.0, -1.0, 1.0)
        obs_gyro = np.clip(raw['gyro'] / 6.0, -1.0, 1.0)
        obs_quat = raw['quat']
        depth = env.WATER_SURFACE_Z - pos_world[2]
        obs_depth = np.array([np.clip(depth / 50.0, 0.0, 1.0)])
        obs_sonar = np.clip(raw.get('sonar', np.zeros(15)) / 12.0, 0.0, 1.0)
        obs_alt = np.array([np.clip(raw.get('altitude', 0) / 50.0, 0.0, 1.0)])
        obs_accel = np.clip(raw['accel'] / 9.81, -3.0, 3.0)

        # CTE vector in body frame (gives agent directional correction hint)
        proj_pt_world = self._map_1d_s_to_3d_pos(s_curr)
        cte_vec_body = rot_mat.T @ (proj_pt_world - pos_world)
        obs_cte = np.clip(cte_vec_body / 5.0, -1.0, 1.0)

        return np.concatenate([
            obs_pos, obs_vel, obs_gyro, obs_quat, gravity_body,
            obs_depth, obs_sonar, obs_alt, obs_accel, obs_cte
        ]).astype(np.float32)

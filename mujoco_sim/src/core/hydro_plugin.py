import numpy as np
import torch
import joblib

from src.utils.kalman_filter import KalmanFilter6D
from src.core.models import build_model


class HydroInference:
    """
    Wraps DeepHydroMLP for online inference inside the MuJoCo step loop.

    Coordinate convention
    ---------------------
    The MLP was trained on Stonefish (NED) data; MuJoCo uses ENU.
    ``coord_mask`` flips the axes that differ between the two frames
    before feeding the network and after reading its output.
    """

    # ENU <-> NED sign flip for [Surge, Sway, Heave, Roll, Pitch, Yaw]
    COORD_MASK = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])

    # Input clipping bounds (physical saturation limits)
    MAX_LIN_VEL = 1.3   # m/s
    MAX_ANG_VEL = 1.0   # rad/s

    def __init__(self, model_path: str, scaler_x_path: str, scaler_y_path: str,
                 device: str = 'cpu', model_type: str = 'baseline',
                 force_clip: float = 180.0, max_lin_vel: float = 1.3,
                 extrapolate: bool = False):
        self.device = torch.device(device)
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.model = build_model(model_type, input_dim=12, output_dim=6).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        # Output clamp. 180 N/Nm was tuned for the legacy model; the
        # thrust-corrected model legitimately predicts up to ~400 N of surge
        # drag, so the clamp must be raised or the plant becomes too slippery.
        self.force_clip = float(force_clip)
        self.max_lin_vel = float(max_lin_vel)
        self.extrapolate = bool(extrapolate)

    def predict(self, velocity: np.ndarray, acceleration: np.ndarray, gain: float = -1.0,
                accel_clip_sigma: float = 3.0) -> np.ndarray:
        """
        Predict 6-DOF hydrodynamic force in the MuJoCo (ENU) body frame.

        Parameters
        ----------
        velocity     : 6-DOF body-frame velocity [u, v, w, p, q, r] in ENU
        acceleration : 6-DOF body-frame acceleration from the Kalman filter
        gain         : scalar multiplier applied to the raw network output
                       (use -1.0 to convert from ``force on fluid'' to
                       ``force on body'' convention)
        accel_clip_sigma : clip the standardised acceleration inputs to
                       +/- this many training standard deviations. The network
                       was trained on a bounded acceleration distribution;
                       extrapolating beyond it produces large, unphysical
                       torques that the closed loop amplifies.

        Returns
        -------
        np.ndarray of shape (6,), clipped to [-180, 180] N / Nm
        """
        vel_ned = velocity * self.COORD_MASK
        acc_ned = acceleration * self.COORD_MASK
        vel_ned[:3] = np.clip(vel_ned[:3], -self.max_lin_vel, self.max_lin_vel)
        vel_ned[3:] = np.clip(vel_ned[3:], -self.MAX_ANG_VEL, self.MAX_ANG_VEL)

        input_vector = np.concatenate([vel_ned, acc_ned]).reshape(1, -1)
        input_scaled = self.scaler_x.transform(input_vector)
        input_scaled[:, 6:] = np.clip(input_scaled[:, 6:], -accel_clip_sigma, accel_clip_sigma)
        with torch.no_grad():
            tensor_out = self.model(torch.tensor(input_scaled, dtype=torch.float32).to(self.device))
        output_force = self.scaler_y.inverse_transform(tensor_out.cpu().numpy())

        force_enu = (output_force * self.COORD_MASK * gain).flatten()
        # Quadratic extrapolation beyond the training envelope.  Inside the
        # envelope the network is accurate; outside it the velocity input is
        # clamped, which would otherwise freeze the drag at a constant value
        # and let the vehicle accelerate without bound.  Drag grows as v^2, so
        # rescale each linear axis by (|v| / v_max)^2 past the clamp.
        if self.extrapolate:
            for i in range(3):
                v = abs(velocity[i])
                if v > self.max_lin_vel:
                    force_enu[i] *= (v / self.max_lin_vel) ** 2
        return np.clip(force_enu, -self.force_clip, self.force_clip).flatten()


class HydroDynamicsPlugin:
    """
    Applies hydrodynamic forces (drag, added mass, buoyancy) to the AUV
    body at every simulation step.

    Two operating modes are available for comparative experiments:

    simplified_mode=True  (default)
        Only added-mass forces are applied; the MLP drag prediction is
        zeroed out.  Use this as a physics-only baseline.

    simplified_mode=False
        Full MLP-predicted drag + added-mass forces.  Surge drag is
        overridden with a tuned quadratic model to account for the
        known sway-decoupled layout of the horizontal thrusters.
    """

    # Added-mass coefficients [Surge, Sway, Heave, Roll, Pitch, Yaw]
    MA = np.array([0.0, 102.31, 121.68, 0.0, 0.0, 0.0])

    BUOYANCY_MAGNITUDE = 1790.0  # N
    VEL_DEADBAND = 0.05          # m/s — suppress drag noise at near-zero velocity

    def __init__(
        self,
        model_path: str,
        scaler_x: str,
        scaler_y: str,
        dt: float,
        simplified_mode: bool = True,
        mlp_accel_input: bool = False,
        accel_smoothing_tau: float = 0.15,
        accel_clip_sigma: float = 3.0,
        model_type: str = 'baseline',
        surge_override: bool = True,
        mlp_drag_scale: float = 0.7,
        force_clip: float = 180.0,
        max_lin_vel: float = 1.3,
        extrapolate: bool = False,
        buoyancy_magnitude: float = 1790.0,
    ):
        # dt is the interval between plugin calls (the control period), not the
        # MuJoCo physics timestep — the Kalman filter's constant-acceleration
        # model must match how often it is actually stepped.
        self.dt = dt
        self.simplified_mode = simplified_mode
        # Feeding the Kalman-filter acceleration into the network closes a
        # feedback loop a -> F -> a. Three things keep that loop stable:
        #   1. the filter is stepped with the real control period (see above);
        #   2. the estimate is EMA-smoothed with a time constant comparable to
        #      the Savitzky-Golay window (51 samples @ 200 Hz = 0.26 s) used to
        #      build the training targets;
        #   3. the standardised acceleration input is clipped to the training
        #      distribution, so the network never extrapolates.
        # With the acceleration channel disabled the network is evaluated at
        # a = 0 (the mean of a zero-mean signal), i.e. as a steady-state drag
        # residual, and the structured added-mass term below carries the
        # acceleration dependence.
        self.mlp_accel_input = mlp_accel_input
        self.accel_smoothing_tau = accel_smoothing_tau
        self.accel_clip_sigma = accel_clip_sigma
        # Surge override: the released pipeline overrode the MLP's Fx with a
        # hand-tuned quadratic because the training label used thruster
        # setpoints (rad/s) instead of thrust (N).  The thrust-corrected model
        # predicts Fx from real propulsion forces, so the override must be
        # switched off for it.  ``mlp_drag_scale`` is the matching global gain
        # (0.7 was tuned against the wrong label; 1.0 is the physical value).
        self.surge_override = surge_override
        self.mlp_drag_scale = mlp_drag_scale
        # The MuJoCo body mass (184.19 kg) is the rigid mass plus surge added
        # mass, so MuJoCo's own gravity is 1807 N while the released buoyancy
        # is 1790 N — a 17 N downward bias.  Stonefish is neutrally buoyant
        # (measured dw/dt = 0.002 m/s^2 with the thrusters idle), so the
        # thrust-corrected config sets buoyancy = mass * g.
        self.buoyancy_magnitude = float(buoyancy_magnitude)
        self.accel_ema = None
        self.predictor = HydroInference(model_path, scaler_x, scaler_y, model_type=model_type,
                                        force_clip=force_clip, max_lin_vel=max_lin_vel,
                                        extrapolate=extrapolate)
        self.kf = None
        self.reset()

    def reset(self):
        """Reset the Kalman filter and acceleration smoother between episodes."""
        # Process/measurement noise kept at the filter defaults: very stiff
        # tuning (process_noise=10, measure_noise=1e-3) makes the estimate
        # track velocity noise instead of the physical acceleration.
        self.kf = KalmanFilter6D(dt=self.dt, process_noise=0.1, measure_noise=0.01)
        self.accel_ema = None
        self.surge_noise_state = 0.0
        self._noise_smoothing = 0.1

    def apply_hydrodynamics(self, robot):
        """
        Compute and apply all hydrodynamic forces to the robot body.

        Returns
        -------
        current_vel_body : np.ndarray (6,)  — body-frame velocity
        hydro_force_body : np.ndarray (6,)  — MLP drag term (body frame)
        total_hydro_body : np.ndarray (6,)  — drag + added mass (body frame)
        """
        data = robot.data
        current_vel_body, rot_mat = robot.get_body_state()
        accel_body = self.kf.update(current_vel_body)

        hydro_force_body = np.zeros(6)

        if self.simplified_mode:
            # Baseline: added-mass only; MLP drag disabled
            total_hydro_body = -self.MA * accel_body
        else:
            # Full mode: MLP drag + added mass
            if self.mlp_accel_input:
                alpha = 1.0 - np.exp(-self.dt / max(self.accel_smoothing_tau, 1e-6))
                if self.accel_ema is None:
                    self.accel_ema = accel_body.copy()
                else:
                    self.accel_ema = (1.0 - alpha) * self.accel_ema + alpha * accel_body
                accel_for_mlp = self.accel_ema
            else:
                accel_for_mlp = np.zeros(6)
            hydro_force_body = self.predictor.predict(
                current_vel_body, accel_for_mlp, gain=-1.0,
                accel_clip_sigma=self.accel_clip_sigma,
            )

            # Override Surge (index 0) with a tuned quadratic drag model to
            # account for the decoupled horizontal-thruster geometry.
            # Only for the legacy (setpoint-as-force) model; the thrust-
            # corrected model predicts Fx directly.
            if self.surge_override:
                u = current_vel_body[0]
                base_drag = 30.0 * abs(u) + 12.0 * u ** 2
                raw_noise = np.random.uniform(-35.0, 35.0)
                self.surge_noise_state = (
                    (1 - self._noise_smoothing) * self.surge_noise_state
                    + self._noise_smoothing * raw_noise
                )
                total_surge = np.clip(base_drag + self.surge_noise_state, 0, 200.0)
                hydro_force_body[0] = -np.sign(u) * total_surge if abs(u) > 1e-3 else 0.0

            # Velocity dead-band: suppress near-zero drag noise
            for i in range(6):
                vel_abs = abs(current_vel_body[i])
                if vel_abs < self.VEL_DEADBAND:
                    hydro_force_body[i] *= vel_abs / self.VEL_DEADBAND

            added_mass = -self.MA * accel_body
            total_hydro_body = self.mlp_drag_scale * hydro_force_body + added_mass

        # Transform to world frame
        f_hydro_world = rot_mat @ total_hydro_body[:3]
        t_hydro_world = rot_mat @ total_hydro_body[3:]

        # Buoyancy (always applied)
        pos_com, pos_cob = robot.get_world_pose()
        f_buoyancy = np.array([0.0, 0.0, self.buoyancy_magnitude])
        t_buoyancy = np.cross(pos_cob - pos_com, f_buoyancy)

        data.xfrc_applied[robot.body_id] = np.concatenate([
            f_buoyancy + f_hydro_world,
            t_buoyancy + t_hydro_world,
        ])

        return current_vel_body, hydro_force_body, total_hydro_body

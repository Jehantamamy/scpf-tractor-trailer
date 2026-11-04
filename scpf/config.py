"""Simulation and controller parameters (Table 1 of the manuscript)."""
from dataclasses import dataclass, field, asdict
import math


@dataclass
class Params:
    # Integration and kinematics
    dt: float = 0.1                  # control / integration period [s]
    v: float = 1.0                   # constant forward speed [m/s]
    L1: float = 0.8                  # tractor wheelbase [m]
    L2: float = 1.2                  # hitch-to-trailer-axle distance [m]
    body_width: float = 0.6          # body width of tractor and trailer [m]
    tractor_front: float = 1.0       # rear axle to tractor front bumper [m]
    tractor_rear: float = 0.2        # rear axle to tractor rear bumper [m]
    trailer_rear: float = 0.2        # trailer axle to trailer rear bumper [m]

    # Actuator limits (Assumptions 2-3)
    delta_max_deg: float = 45.0      # steering travel limit [deg]
    delta_rate_max: float = 1.0      # steering slew-rate limit [rad/s]
    alpha_steer: float = 0.5         # first-order filter coefficient

    # Articulation thresholds
    gamma_safe_deg: float = 60.0     # governor intervention threshold [deg]
    gamma_max_deg: float = 70.0      # mechanical jackknife limit [deg]

    # Common potential-field gains (shared by APF, CPF and SCPF)
    k_att: float = 5.0
    k_rep: float = 10.0
    rho_0: float = 3.0               # obstacle influence radius [m]
    d_floor: float = 0.05            # clearance floor in the repulsive law [m]
    F_sat: float = 50.0              # saturation of the resultant field
    k_head: float = 1.0              # heading-error-to-steering gain (all methods)
    k_t: float = 2.5                 # tangential (circulatory) gain relative to repulsion

    # SCPF-specific terms
    k_overdrive: float = 5.0         # attractive gain multiplier toward a subgoal
    eta_g: float = 0.8               # goal-proximity attenuation range factor
    w_single: float = 2.0            # repulsion weight of isolated obstacles
    w_cluster: float = 0.6           # repulsion weight of clustered obstacles / lids
    rho_trailer: float = 1.5         # trailer clearance base radius [m]
    k_gamma: float = 1.5             # trailer clearance expansion gain [1/rad]
    k_trailer: float = 1.0           # trailer correction gain
    tau_trailer_max_deg: float = 45.0  # bound on trailer correction [deg]
    k_jk: float = 10.0               # virtual counter-torque gain
    eps_latch: float = 1.0           # subgoal latch radius [m]
    stage_dist: float = 2.5          # portal staging distance along the portal normal [m]
    R_sense: float = 8.0             # sensing radius [m]
    d_margin: float = 0.4            # bilateral clearance margin in W_swept [m]

    # Episode termination
    goal_tol: float = 1.0            # goal acceptance radius (rear-axle point) [m]
    max_steps: int = 1500            # 150 s horizon
    stall_window: int = 100          # 10 s
    stall_dist: float = 1.0          # minimum displacement within the window [m]

    # Component toggles (ablation)
    use_hysteresis: bool = True
    use_projection: bool = True
    use_hull: bool = True
    use_gap_screening: bool = True
    use_governor: bool = True
    use_trailer_zone: bool = True
    use_goal_attenuation: bool = True

    @property
    def delta_max(self) -> float:
        return math.radians(self.delta_max_deg)

    @property
    def gamma_safe(self) -> float:
        return math.radians(self.gamma_safe_deg)

    @property
    def gamma_max(self) -> float:
        return math.radians(self.gamma_max_deg)

    @property
    def delta_R_max(self) -> float:
        """Transient trailer off-tracking evaluated at the jackknife limit."""
        return self.L2 * math.sin(self.gamma_max)

    @property
    def W_swept(self) -> float:
        """Minimum traversable aperture used by gap screening."""
        return self.body_width + self.delta_R_max + self.d_margin

    def to_dict(self):
        d = asdict(self)
        d["W_swept"] = self.W_swept
        d["delta_R_max"] = self.delta_R_max
        return d

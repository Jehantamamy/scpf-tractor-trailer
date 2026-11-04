"""Vehicle kinematics, RK4 numerical integration, and geometric footprint."""
import math
import numpy as np
from shapely.geometry import Polygon


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rk4_step(x: float, y: float, theta: float, trailer_theta: float,
             v: float, delta: float, L1: float, L2: float, dt: float):
    """One 4th-order Runge-Kutta numerical integration step for the 1-trailer system.

    Equations of motion:
        dx/dt     = v * cos(theta)
        dy/dt     = v * sin(theta)
        dtheta/dt = (v / L1) * tan(delta)
        dtheta_t/dt = (v / L2) * sin(theta - theta_t)
    """
    def f(th, tr_th):
        dx = v * math.cos(th)
        dy = v * math.sin(th)
        dth = (v / L1) * math.tan(delta)
        dtr_th = (v / L2) * math.sin(th - tr_th)
        return np.array([dx, dy, dth, dtr_th])

    k1 = f(theta, trailer_theta)
    k2 = f(theta + 0.5 * dt * k1[2], trailer_theta + 0.5 * dt * k1[3])
    k3 = f(theta + 0.5 * dt * k2[2], trailer_theta + 0.5 * dt * k2[3])
    k4 = f(theta + dt * k3[2], trailer_theta + dt * k3[3])

    delta_state = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    new_x = x + delta_state[0]
    new_y = y + delta_state[1]
    new_theta = wrap_to_pi(theta + delta_state[2])
    new_tr_theta = wrap_to_pi(trailer_theta + delta_state[3])

    return new_x, new_y, new_theta, new_tr_theta


class ActuatorPipeline:
    """Actuator dynamic pipeline: rate limiter, first-order filter, and travel clamp."""

    def __init__(self, max_steer_deg: float = 35.0, max_rate_deg_s: float = 34.38,
                 alpha: float = 0.5, dt: float = 0.1, delta0: float = 0.0):
        self.delta_max = math.radians(max_steer_deg)
        self.step_max = math.radians(max_rate_deg_s) * dt
        self.alpha = alpha
        self.rate_state = delta0
        self.filt_state = delta0

    def step(self, delta_cmd: float) -> float:
        rate_change = float(np.clip(delta_cmd - self.rate_state, -self.step_max, self.step_max))
        self.rate_state += rate_change
        self.filt_state = self.alpha * self.rate_state + (1.0 - self.alpha) * self.filt_state
        return float(np.clip(self.filt_state, -self.delta_max, self.delta_max))


def get_tractor_trailer_footprint(t_pos, tr_pos, t_ori, tr_ori, L1=0.8, L2=1.2, W1=0.45, W2=0.45):
    """Generate Shapely polygons for the tractor and trailer body boxes."""
    # Tractor box corners around rear axle
    trac_corners = np.array([[-0.2, -W1/2], [L1-0.2, -W1/2], [L1-0.2, W1/2], [-0.2, W1/2]])
    R_t = np.array([[math.cos(t_ori), -math.sin(t_ori)], [math.sin(t_ori), math.cos(t_ori)]])
    trac_rot = trac_corners @ R_t.T + t_pos

    # Trailer box corners behind hitch
    trail_corners = np.array([[-L2, -W2/2], [0, -W2/2], [0, W2/2], [-L2, W2/2]])
    R_tr = np.array([[math.cos(tr_ori), -math.sin(tr_ori)], [math.sin(tr_ori), math.cos(tr_ori)]])
    trail_rot = trail_corners @ R_tr.T + t_pos

    return Polygon(trac_rot), Polygon(trail_rot)

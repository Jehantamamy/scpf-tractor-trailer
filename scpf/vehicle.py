"""Tractor-trailer kinematics, actuator pipeline and body footprint."""
import math
import numpy as np
from shapely.geometry import Polygon


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rk4_step(state, v, delta, L1, L2, dt):
    """One RK4 step of x' = v cos th, y' = v sin th, th' = v/L1 tan(delta),
    th_t' = v/L2 sin(th - th_t). Steering is held constant over the step."""
    x, y, th, tht = state

    def f(th_, tht_):
        return np.array([v * math.cos(th_), v * math.sin(th_),
                         v / L1 * math.tan(delta), v / L2 * math.sin(th_ - tht_)])

    k1 = f(th, tht)
    k2 = f(th + 0.5 * dt * k1[2], tht + 0.5 * dt * k1[3])
    k3 = f(th + 0.5 * dt * k2[2], tht + 0.5 * dt * k2[3])
    k4 = f(th + dt * k3[2], tht + dt * k3[3])
    d = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array([x + d[0], y + d[1], wrap(th + d[2]), wrap(tht + d[3])])


class ActuatorPipeline:
    """Rate limiter -> first-order filter -> travel clamp (Eq. 17)."""

    def __init__(self, delta_max, rate_max, alpha, dt, delta0=0.0):
        self.delta_max = delta_max
        self.step_max = rate_max * dt
        self.alpha = alpha
        self.rate_state = delta0
        self.filt_state = delta0

    def step(self, delta_raw):
        change = float(np.clip(delta_raw - self.rate_state, -self.step_max, self.step_max))
        self.rate_state += change
        self.filt_state = self.alpha * self.rate_state + (1.0 - self.alpha) * self.filt_state
        return float(np.clip(self.filt_state, -self.delta_max, self.delta_max))


def articulation(state):
    return wrap(state[2] - state[3])


def trailer_axle(state, L2):
    return np.array([state[0] - L2 * math.cos(state[3]), state[1] - L2 * math.sin(state[3])])


def _rect(cx, cy, heading, back, front, half_w):
    c, s = math.cos(heading), math.sin(heading)
    pts = [(-back, -half_w), (front, -half_w), (front, half_w), (-back, half_w)]
    return Polygon([(cx + c * a - s * b, cy + s * a + c * b) for a, b in pts])


def footprint(state, p):
    """Tractor and trailer body rectangles (hitch located at the tractor rear axle)."""
    x, y, th, tht = state
    hw = 0.5 * p.body_width
    tractor = _rect(x, y, th, p.tractor_rear, p.tractor_front, hw)
    # trailer body spans from the hitch back to the trailer rear bumper
    trailer = _rect(x, y, tht, p.L2 + p.trailer_rear, 0.0, hw)
    return tractor, trailer

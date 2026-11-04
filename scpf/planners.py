"""Reactive planners: classical APF, memoryless circular PF (CPF) and the proposed SCPF.

All planners map the current state and perceived obstacles to a raw steering demand.
The common actuator pipeline (rate limiter, filter, clamp) is applied by the simulator.
"""
import math
import numpy as np
from shapely.ops import nearest_points
from .geometry import closest_point
from .vehicle import wrap, trailer_axle, footprint


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _sat_field(F, F_sat):
    n = float(np.linalg.norm(F))
    return F if n <= F_sat else F / n * F_sat


def _heading_to_steer(F, theta, p):
    err = wrap(math.atan2(F[1], F[0]) - theta)
    return float(np.clip(p.k_head * err, -p.delta_max, p.delta_max))


def _rep_mag(d, p):
    ds = max(d, p.d_floor)
    return (1.0 / ds - 1.0 / p.rho_0) / ds ** 2


def body_closest(body, o):
    """Clearance between the tractor body polygon and an obstacle, the outward unit
    normal (obstacle -> body) and the nearest obstacle point."""
    g = o["geom"]
    d = float(body.distance(g))
    if d > 1e-9:
        q_obs, q_body = nearest_points(g, body)
        c = np.array([q_obs.x, q_obs.y])
        n = np.array([q_body.x, q_body.y]) - c
        return d, n / np.linalg.norm(n), c
    bc = body.centroid
    P = np.array([bc.x, bc.y])
    _, c = closest_point(P, o)
    n = P - c
    return 0.0, n / (np.linalg.norm(n) + 1e-9), c


class APFPlanner:
    name = "APF"

    def __init__(self, p, goal):
        self.p, self.goal = p, np.asarray(goal, float)

    def reset(self):
        pass

    def control(self, state, obstacles, subgoals=None):
        p = self.p
        P = state[:2]
        g = self.goal - P
        F = p.k_att * g / (np.linalg.norm(g) + 1e-9)
        body = footprint(state, p)[0]
        for o in obstacles:
            d, n, c = body_closest(body, o)
            if d < p.rho_0:
                F = F + p.k_rep * _rep_mag(d, p) * n
        F = _sat_field(F, p.F_sat)
        return _heading_to_steer(F, state[2], p), {"tau_jk": 0.0, "tau_tr": 0.0, "subgoal": None}


class CPFPlanner(APFPlanner):
    """Circular potential field with a memoryless circulation sign (goal cross product)."""
    name = "CPF"

    def control(self, state, obstacles, subgoals=None):
        p = self.p
        P = state[:2]
        g = self.goal - P
        F = p.k_att * g / (np.linalg.norm(g) + 1e-9)
        body = footprint(state, p)[0]
        for o in obstacles:
            d, n, c = body_closest(body, o)
            if d < p.rho_0:
                s = 1.0 if _cross2(n, self.goal - c) > 0 else -1.0
                t = s * np.array([-n[1], n[0]])
                m = p.k_rep * _rep_mag(d, p)
                F = F + m * n + p.k_t * m * t
        F = _sat_field(F, p.F_sat)
        return _heading_to_steer(F, state[2], p), {"tau_jk": 0.0, "tau_tr": 0.0, "subgoal": None}


class SCPFPlanner:
    """Stateful circular potential field with subgoal latching, orthogonal projection,
    goal-proximity attenuation, adaptive trailer clearance and a counter-torque governor."""
    name = "SCPF"

    def __init__(self, p, goal):
        self.p, self.goal = p, np.asarray(goal, float)
        self.reset()

    def reset(self):
        self.signs = {}
        self.was_escaping = False
        self.stage = 0
        self.prev_target = None
        self.turn_sign = 0.0

    def _sign(self, key, n, t_cw, theta, target, c, cluster_sign):
        p = self.p
        if not p.use_hysteresis:
            return 1.0 if _cross2(n, target - c) > 0 else -1.0
        if key not in self.signs:
            m = math.cos(theta) * t_cw[0] + math.sin(theta) * t_cw[1]
            if abs(m) > 0.3:
                self.signs[key] = 1.0 if m > 0 else -1.0
            elif cluster_sign is not None:
                self.signs[key] = cluster_sign
            else:
                self.signs[key] = 1.0 if _cross2(n, target - c) > 0 else -1.0
        return self.signs[key]

    def control(self, state, obstacles, subgoals=None):
        p = self.p
        P = state[:2]
        theta, theta_t = state[2], state[3]
        gamma = wrap(theta - theta_t)

        # subgoal sequence (portal staging points) with spatial latching
        target, k_att, escaping = self.goal, p.k_att, False
        if subgoals:
            while self.stage < len(subgoals) and \
                    np.linalg.norm(P - np.asarray(subgoals[self.stage], float)) < p.eps_latch:
                self.stage += 1
            if self.stage < len(subgoals):
                target = np.asarray(subgoals[self.stage], float)
                k_att, escaping = p.k_att * p.k_overdrive, True
        else:
            self.stage = 0
        if escaping != self.was_escaping or (
                escaping and self.prev_target is not None
                and np.linalg.norm(target - self.prev_target) > 1.0):
            self.signs.clear()
        self.was_escaping = escaping
        self.prev_target = target if escaping else None

        g = target - P
        d_goal = float(np.linalg.norm(g)) + 1e-9
        F_att = k_att * g / d_goal
        d_final = float(np.linalg.norm(self.goal - P))
        beta = min(1.0, (d_final / (p.rho_0 * p.eta_g)) ** 2) if p.use_goal_attenuation else 1.0

        F_rep = np.zeros(2)
        F_circ = np.zeros(2)
        tau_tr = 0.0
        d_min, n_min = math.inf, None
        Ptr = trailer_axle(state, p.L2)
        rho_tr = p.rho_trailer * (1.0 + p.k_gamma * abs(gamma))
        active = set()
        body = footprint(state, p)[0]
        nearest, nearest_tr = {}, {}
        for idx, o in enumerate(obstacles):
            gkey = o.get("cluster") or o.get("key", idx)
            d, n, c = body_closest(body, o)
            if gkey not in nearest or d < nearest[gkey][0]:
                nearest[gkey] = (d, n, c, o)
            if p.use_trailer_zone:
                dt_, ct = closest_point(Ptr, o)
                if gkey not in nearest_tr or dt_ < nearest_tr[gkey][0]:
                    nearest_tr[gkey] = (dt_, ct)
        u = g / d_goal
        for key, (d, n, c, o) in nearest.items():
            if d < d_min:
                d_min, n_min = d, n
            if d < p.rho_0:
                active.add(key)
                t_cw = np.array([-n[1], n[0]])
                cs = o.get("cluster_sign")
                s = self._sign(key, n, t_cw, theta, target, c, cs)
                w = p.w_single if cs is None else p.w_cluster
                m = p.k_rep * _rep_mag(d, p) * beta * w
                dot = n @ u
                circ = 1.0 if dot <= 0 else max(0.0, 1.0 - dot / 0.15)
                F_circ += m * p.k_t * circ * s * t_cw
                F_rep += m * n
        th_dir = np.array([math.cos(theta_t), math.sin(theta_t)])
        for dt_, ct in nearest_tr.values():
            if dt_ < rho_tr:
                v = ct - Ptr
                side = th_dir[0] * v[1] - th_dir[1] * v[0]
                ds = max(dt_, 0.2)
                level = (1.0 / ds - 1.0 / rho_tr) ** 2
                tau_tr += (-1.0 if side > 0 else 1.0) * p.k_trailer * level * beta * (1.0 + p.k_gamma * abs(gamma))
        tau_tr = float(np.clip(tau_tr, -math.radians(p.tau_trailer_max_deg), math.radians(p.tau_trailer_max_deg)))

        if p.use_projection and n_min is not None and d_min < p.rho_0:
            a = F_att @ n_min
            if a < 0:
                F_att = F_att - beta * (1.0 - max(d_min, 0.0) / p.rho_0) * a * n_min

        for k in [k for k in self.signs if k not in active]:
            del self.signs[k]

        F = _sat_field(F_att + F_rep + F_circ, p.F_sat)
        err = wrap(math.atan2(F[1], F[0]) - theta)
        if p.use_hysteresis:
            if abs(err) > math.radians(150.0) and self.turn_sign != 0.0:
                err = self.turn_sign * abs(err)
            self.turn_sign = math.copysign(1.0, err) if abs(err) > math.radians(90.0) else 0.0
        delta_apf = float(np.clip(p.k_head * err, -p.delta_max, p.delta_max))
        delta_nom = float(np.clip(delta_apf + tau_tr, -p.delta_max, p.delta_max))
        tau_jk = 0.0
        if p.use_governor and abs(gamma) > p.gamma_safe:
            tau_jk = -p.k_jk * math.copysign(1.0, gamma) * (abs(gamma) - p.gamma_safe)
        delta_raw = float(np.clip(delta_nom + tau_jk, -p.delta_max, p.delta_max))
        return delta_raw, {"tau_jk": tau_jk, "tau_tr": tau_tr, "subgoal": target if escaping else None}

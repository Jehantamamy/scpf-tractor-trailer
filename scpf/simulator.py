"""Closed-loop episode runner shared by every experiment."""
import math
import time
import numpy as np

from .config import Params
from .geometry import make_obstacle, update_dynamic
from .perception import sense, perceive
from .vehicle import rk4_step, ActuatorPipeline, articulation, footprint


def prepare_obstacles(obstacles):
    out = []
    for k, o in enumerate(obstacles):
        o2 = make_obstacle(o)
        o2.setdefault("key", ("obs", k))
        out.append(o2)
    return out


def body_clearance(state, obstacles, p):
    """Minimum distance between the tractor/trailer footprint and real obstacles."""
    tractor, trailer = footprint(state, p)
    d = math.inf
    for o in obstacles:
        g = o["geom"]
        d = min(d, tractor.distance(g), trailer.distance(g))
    return d


def run_episode(planner_cls, obstacles, start, goal, p=None, theta0=None,
                gamma0=0.0, delta0=0.0, seed=0, steer_delay=0, noise_std=0.0,
                record=True, timing=False):
    """Simulate one episode.

    start   : (x, y) of the tractor rear axle
    theta0  : initial tractor heading (defaults to the start-goal bearing)
    gamma0  : initial articulation angle [rad]
    delta0  : initial actuator state [rad]
    steer_delay : number of control periods of pure transport delay on the command
    noise_std   : standard deviation of Gaussian noise on (x, y) [m] and angles [rad/10]
    """
    p = p or Params()
    rng = np.random.default_rng(seed)
    obs = prepare_obstacles(obstacles)
    goal = np.asarray(goal, float)
    start = np.asarray(start, float)
    if theta0 is None:
        theta0 = math.atan2(goal[1] - start[1], goal[0] - start[0])
    state = np.array([start[0], start[1], theta0, theta0 - gamma0])
    planner = planner_cls(p, goal)
    planner.reset()
    is_scpf = planner.name == "SCPF"
    act = ActuatorPipeline(p.delta_max, p.delta_rate_max, p.alpha_steer, p.dt, delta0)
    delay_buf = [delta0] * steer_delay

    log = {k: [] for k in ("x", "y", "theta", "theta_t", "gamma", "delta", "delta_raw",
                           "tau_jk", "tau_tr", "clearance")}
    path_len, outcome = 0.0, "timeout"
    gov_steps, latencies = 0, []
    peak_gamma, min_clear = abs(gamma0), math.inf
    prev_delta = delta0
    max_rate, rate_viol = 0.0, 0
    hist = [state[:2].copy()]

    for k in range(p.max_steps):
        meas = state.copy()
        if noise_std > 0:
            meas[:2] += rng.normal(0.0, noise_std, 2)
            meas[2:] += rng.normal(0.0, noise_std / 10.0, 2)

        t0 = time.perf_counter()
        visible = sense(obs, meas[:2], p.R_sense)
        subgoals = []
        if is_scpf:
            used, subgoals, _ = perceive(visible, goal, meas[:2], meas[2], p)
        else:
            used = visible
        delta_raw, info = planner.control(meas, used, subgoals)
        t1 = time.perf_counter()
        if timing:
            latencies.append(t1 - t0)

        if steer_delay > 0:
            delay_buf.append(delta_raw)
            delta_cmd = delay_buf.pop(0)
        else:
            delta_cmd = delta_raw
        delta = act.step(delta_cmd)

        rate = abs(delta - prev_delta) / p.dt
        max_rate = max(max_rate, rate)
        if rate > p.delta_rate_max + 1e-9:
            rate_viol += 1
        prev_delta = delta
        if info.get("tau_jk", 0.0) != 0.0:
            gov_steps += 1

        new_state = rk4_step(state, p.v, delta, p.L1, p.L2, p.dt)
        path_len += float(np.hypot(*(new_state[:2] - state[:2])))
        state = new_state
        update_dynamic(obs, p.dt)

        gamma = articulation(state)
        clear = body_clearance(state, obs, p)
        peak_gamma = max(peak_gamma, abs(gamma))
        min_clear = min(min_clear, clear)
        if record:
            for key, val in (("x", state[0]), ("y", state[1]), ("theta", state[2]),
                             ("theta_t", state[3]), ("gamma", gamma), ("delta", delta),
                             ("delta_raw", delta_raw), ("tau_jk", info.get("tau_jk", 0.0)),
                             ("tau_tr", info.get("tau_tr", 0.0)), ("clearance", clear)):
                log[key].append(float(val))
        hist.append(state[:2].copy())

        if clear <= 0.0:
            outcome = "collision"
            break
        if abs(gamma) >= p.gamma_max:
            outcome = "jackknife"
            break
        if np.linalg.norm(state[:2] - goal) <= p.goal_tol:
            outcome = "success"
            break
        if k >= p.stall_window and np.linalg.norm(hist[-1] - hist[-1 - p.stall_window]) < p.stall_dist:
            outcome = "trapped"
            break

    steps = k + 1
    d = np.asarray(log["delta"]) if record else np.array([])
    chatter = float(np.mean(np.abs(np.diff(d)))) if d.size > 1 else 0.0
    result = {
        "planner": planner.name, "outcome": outcome, "steps": steps, "time_s": steps * p.dt,
        "path_length": path_len, "peak_gamma_deg": math.degrees(peak_gamma),
        "min_clearance": min_clear, "chatter_deg": math.degrees(chatter),
        "governor_steps": gov_steps, "max_rate": max_rate, "rate_violations": rate_viol,
    }
    if timing and latencies:
        lat = np.asarray(latencies) * 1e3
        result.update({"latency_ms_mean": float(lat.mean()), "latency_ms_std": float(lat.std()),
                       "latency_n": int(lat.size), "latency_ms": lat.tolist()})
    if record:
        result["log"] = log
    return result

"""Articulation governor: analytic overshoot bound (Theorem 2) versus adversarial
closed-loop simulation, and robustness of the full SCPF pipeline under
reduced actuator authority, command delay and measurement noise."""
import math
import numpy as np
from collections import Counter
from common import run_all, save_csv, save_json, Params
from scpf.scenarios import escape_starts


def gamma_rate(g, d, p):
    return p.v / p.L1 * math.tan(d) - p.v / p.L2 * math.sin(g)


def analytic_bound(p, delay=0):
    """Upper bound on max|gamma| - gamma_safe for |delta_nom| <= delta_max,
    zero-order hold, rate limiter -> filter -> clamp and `delay` samples of
    pure command delay. Returns (bound_rad, filtered sequence)."""
    step = p.delta_rate_max * p.dt
    d_crit = math.atan(p.L1 / p.L2 * math.sin(p.gamma_safe))
    seq = [p.delta_max] * (1 + delay)
    d_rate, d_filt = p.delta_max, p.delta_max
    while True:
        d_rate -= step
        d_filt = p.alpha_steer * d_rate + (1 - p.alpha_steer) * d_filt
        if d_filt <= d_crit:
            break
        seq.append(d_filt)
    bound = step / p.k_jk + p.dt * sum(gamma_rate(p.gamma_safe, d, p) for d in seq)
    return bound, seq


def adversarial_peak(p, delay=0, n_phase=400, substeps=50):
    """Closed loop with the nominal command pinned at +delta_max (worst case),
    entering the intervention region at every sampling phase."""
    step = p.delta_rate_max * p.dt
    worst = 0.0
    g_dot_max = gamma_rate(p.gamma_safe, p.delta_max, p) * p.dt
    for g0 in np.linspace(p.gamma_safe - g_dot_max, p.gamma_safe, n_phase):
        g, dr, df, peak = g0, p.delta_max, p.delta_max, g0
        buf = [p.delta_max] * delay
        for _ in range(80):
            e = abs(g) - p.gamma_safe
            tau = -p.k_jk * math.copysign(1.0, g) * e if e > 0 else 0.0
            raw = float(np.clip(p.delta_max + tau, -p.delta_max, p.delta_max))
            if delay:
                buf.append(raw)
                raw = buf.pop(0)
            dr += float(np.clip(raw - dr, -step, step))
            df = p.alpha_steer * dr + (1 - p.alpha_steer) * df
            d = float(np.clip(df, -p.delta_max, p.delta_max))
            for _ in range(substeps):
                g += gamma_rate(g, d, p) * p.dt / substeps
                peak = max(peak, g)
        worst = max(worst, peak - p.gamma_safe)
    return worst


VARIANTS = [
    ("nominal", {}, 0),
    ("slew 0.5 rad/s", {"delta_rate_max": 0.5}, 0),
    ("slew 0.25 rad/s", {"delta_rate_max": 0.25}, 0),
    ("alpha 0.3", {"alpha_steer": 0.3}, 0),
    ("delay 1 step", {}, 1),
    ("delay 2 steps", {}, 2),
    ("k_jk 5", {"k_jk": 5.0}, 0),
    ("v 1.5 m/s", {"v": 1.5}, 0),
]

ROBUST = [
    ("nominal", {}, {}),
    ("slew 0.5 rad/s", {"delta_rate_max": 0.5}, {}),
    ("delay 2 steps", {}, {"steer_delay": 2}),
    ("noise 0.05 m", {}, {"noise_std": 0.05}),
    ("noise 0.10 m", {}, {"noise_std": 0.10}),
    ("v 1.5 m/s", {"v": 1.5}, {}),
]

if __name__ == "__main__":
    bound_rows = []
    for name, ov, delay in VARIANTS:
        p = Params(**ov)
        b, seq = analytic_bound(p, delay)
        sim = adversarial_peak(p, delay)
        bound_rows.append({"variant": name, "bound_deg": math.degrees(b), "simulated_deg": math.degrees(sim),
                           "peak_bound_deg": p.gamma_safe_deg + math.degrees(b),
                           "within_limit": p.gamma_safe_deg + math.degrees(b) < p.gamma_max_deg,
                           "filtered_sequence_deg": [round(math.degrees(s), 2) for s in seq]})
        print(f"{name:16s} bound {math.degrees(b):6.2f} deg  simulated {math.degrees(sim):6.2f} deg")
    save_json(bound_rows, "governor_bound.json")

    cases = []
    for name, ov, kw in ROBUST:
        for (x, y, th) in escape_starts():
            cases.append(dict(planner="SCPF", scene="cshape_escape", start=(x, y), theta0=th,
                              overrides=ov, variant=name, suite="escape_set", **kw))
        for sd in range(100):
            cases.append(dict(planner="SCPF", scene="mc", seed=sd, overrides=ov, variant=name,
                              suite="monte_carlo", **kw))
        for sc in ("zigzag", "clutter"):
            for g0 in (-55.0, 55.0):
                cases.append(dict(planner="SCPF", scene=sc, gamma0=math.radians(g0),
                                  delta0=math.copysign(math.radians(45.0), g0), overrides=ov,
                                  variant=name, suite=f"{sc}_g0", **kw))
    rows = run_all(cases)
    for r in rows:
        r.pop("theta0", None)
    save_csv(rows, "robustness.csv")
    summ = {}
    for name, _, _ in ROBUST:
        summ[name] = {}
        for suite in sorted({r["suite"] for r in rows}):
            sel = [r for r in rows if r["variant"] == name and r["suite"] == suite]
            summ[name][suite] = {"n": len(sel), "outcomes": dict(Counter(r["outcome"] for r in sel)),
                                 "peak_gamma_max": max(r["peak_gamma_deg"] for r in sel),
                                 "rate_violations": sum(r["rate_violations"] for r in sel)}
        print(name, {s: (v["outcomes"].get("success", 0), v["n"], round(v["peak_gamma_max"], 1))
                     for s, v in summ[name].items()})
    save_json(summ, "robustness_summary.json")

"""Shared helpers for the experiment scripts."""
import csv
import json
import math
import os
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from scpf import Params, PLANNERS, run_episode  # noqa: E402
from scpf.scenarios import scenario, monte_carlo_scene  # noqa: E402

SUMMARY_KEYS = ["outcome", "steps", "time_s", "path_length", "peak_gamma_deg", "min_clearance",
                "chatter_deg", "governor_steps", "max_rate", "rate_violations"]


def run_case(case):
    """case: dict(planner, scene, seed?, overrides?, start?, theta0?, gamma0?, delta0?,
    steer_delay?, noise_std?, record?, timing?, tag?)"""
    p = Params(**case.get("overrides", {}))
    if case["scene"] == "mc":
        sc = monte_carlo_scene(case["seed"])
    elif case["scene"] == "dynamic":
        from scpf.scenarios import dynamic_scenario
        sc = dynamic_scenario(**case["dyn"])
    else:
        sc = scenario(case["scene"])
    start = case.get("start", sc["start"])
    theta0 = case.get("theta0", sc["theta0"])
    r = run_episode(PLANNERS[case["planner"]], sc["obstacles"], start, sc["goal"], p,
                    theta0=theta0, gamma0=case.get("gamma0", 0.0), delta0=case.get("delta0", 0.0),
                    seed=case.get("seed", 0), steer_delay=case.get("steer_delay", 0),
                    noise_std=case.get("noise_std", 0.0), record=case.get("record", False),
                    timing=case.get("timing", False))
    row = {k: v for k, v in case.items() if k not in ("overrides", "dyn", "record", "timing")}
    for k, v in case.get("overrides", {}).items():
        row[k] = v
    for k, v in case.get("dyn", {}).items():
        row["dyn_" + k] = v
    for k in SUMMARY_KEYS:
        row[k] = r[k]
    for k in ("latency_ms_mean", "latency_ms_std", "latency_n"):
        if k in r:
            row[k] = r[k]
    if case.get("record"):
        row["log"] = r["log"]
    return row


def run_all(cases, processes=None):
    processes = processes or max(1, (os.cpu_count() or 2) - 1)
    with Pool(processes) as pool:
        return pool.map(run_case, cases, chunksize=1)


def save_csv(rows, name):
    path = os.path.join(RESULTS, name)
    keys = []
    for r in rows:
        for k in r:
            if k != "log" and k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})
    return path


def save_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    return path


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (tuple, list)):
        return json.dumps(v, default=_default)
    return v


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except ImportError:
        pass
    return str(o)


def wilson(k, n, z=1.959964):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    s = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * s)

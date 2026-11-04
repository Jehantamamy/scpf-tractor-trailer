"""Per-step computation time (sensing + perception + field + governor), measured
serially with time.perf_counter on a single core."""
import platform
import numpy as np
from common import save_json, PLANNERS, Params, run_episode
from scpf.scenarios import scenario, monte_carlo_scene
from run_canonical import SCENES

if __name__ == "__main__":
    out = {"python": platform.python_version(), "machine": platform.processor() or platform.machine(),
           "system": platform.platform(), "planners": {}}
    scenes = [scenario(s) for s in SCENES] + [monte_carlo_scene(sd) for sd in range(20)]
    for pn in PLANNERS:
        lat = []
        for sc in scenes:
            r = run_episode(PLANNERS[pn], sc["obstacles"], sc["start"], sc["goal"], Params(),
                            theta0=sc["theta0"], record=False, timing=True)
            lat.extend(r["latency_ms"])
        a = np.asarray(lat)
        out["planners"][pn] = {"n_steps": int(a.size), "n_episodes": len(scenes),
                               "mean_ms": float(a.mean()), "std_ms": float(a.std()),
                               "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
                               "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}
        print(pn, {k: round(v, 3) if isinstance(v, float) else v for k, v in out["planners"][pn].items()})
    save_json(out, "timing.json")

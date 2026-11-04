"""Component ablation of SCPF (Table 3): canonical scenes, escape start set and
the 100 Monte Carlo layouts, one component (or one pair) removed at a time."""
import math
from collections import Counter, defaultdict
from common import run_all, save_csv, save_json, wilson
from run_canonical import SCENES
from scpf.scenarios import escape_starts

CONFIGS = {
    "Full SCPF": {},
    "w/o governor (NCT)": {"use_governor": False},
    "w/o hull/subgoals": {"use_hull": False},
    "w/o gap screening": {"use_gap_screening": False},
    "w/o projection": {"use_projection": False},
    "w/o hysteresis": {"use_hysteresis": False},
    "w/o trailer zone": {"use_trailer_zone": False},
    "w/o goal attenuation": {"use_goal_attenuation": False},
    "w/o governor + trailer zone": {"use_governor": False, "use_trailer_zone": False},
    "w/o hysteresis + projection": {"use_hysteresis": False, "use_projection": False},
}
MC_SEEDS = range(100)

if __name__ == "__main__":
    cases = []
    for name, ov in CONFIGS.items():
        for s in SCENES:
            cases.append(dict(planner="SCPF", scene=s, overrides=ov, config=name, suite="canonical"))
        for (x, y, th) in escape_starts():
            cases.append(dict(planner="SCPF", scene="cshape_escape", start=(x, y), theta0=th,
                              overrides=ov, config=name, suite="escape_set"))
        for sd in MC_SEEDS:
            cases.append(dict(planner="SCPF", scene="mc", seed=sd, overrides=ov, config=name,
                              suite="monte_carlo"))
    rows = run_all(cases)
    for r in rows:
        r.pop("theta0", None)
    save_csv(rows, "ablation.csv")

    summary = defaultdict(dict)
    for name in CONFIGS:
        for suite in ("escape_set", "monte_carlo"):
            sel = [r for r in rows if r["config"] == name and r["suite"] == suite]
            c = Counter(r["outcome"] for r in sel)
            k, n = c.get("success", 0), len(sel)
            lo, hi = wilson(k, n)
            summary[name][suite] = {"n": n, "success": k, "ci95": [lo, hi], "outcomes": dict(c),
                                    "peak_gamma_max": max(r["peak_gamma_deg"] for r in sel),
                                    "governor_episodes": sum(1 for r in sel if r["governor_steps"] > 0)}
        summary[name]["canonical"] = {r["scene"]: {kk: r[kk] for kk in ("outcome", "path_length",
                                                                       "peak_gamma_deg", "chatter_deg",
                                                                       "min_clearance", "governor_steps")}
                                      for r in rows if r["config"] == name and r["suite"] == "canonical"}
    save_json(summary, "ablation_summary.json")
    for name in CONFIGS:
        s = summary[name]
        canon = " ".join(f"{sc[:6]}:{v['outcome'][:4]}" for sc, v in s["canonical"].items())
        print(f"{name:30s} MC {s['monte_carlo']['success']:3d}/100  ESC {s['escape_set']['success']:2d}/"
              f"{s['escape_set']['n']}  {canon}")

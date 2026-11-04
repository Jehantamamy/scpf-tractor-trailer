"""100-trial Monte Carlo benchmark with paired comparison and two feasibility
denominators (Section 4.3, Fig. 11)."""
from collections import Counter
from multiprocessing import Pool
from common import run_all, save_csv, save_json, wilson, mcnemar_exact, PLANNERS, Params
from scpf.scenarios import monte_carlo_scene, feasible

SEEDS = range(100)


def feas(seed):
    sc = monte_carlo_scene(seed)
    p = Params()
    return {"seed": seed, "n_obstacles": sc["n_obstacles"],
            "feasible_body": feasible(sc["obstacles"], sc["start"], sc["goal"], p.body_width / 2),
            "feasible_swept": feasible(sc["obstacles"], sc["start"], sc["goal"], p.W_swept / 2)}


if __name__ == "__main__":
    with Pool() as pool:
        fz = {f["seed"]: f for f in pool.map(feas, SEEDS)}
    cases = [dict(planner=pn, scene="mc", seed=s) for s in SEEDS for pn in PLANNERS]
    rows = run_all(cases)
    for r in rows:
        r.update({k: fz[r["seed"]][k] for k in ("n_obstacles", "feasible_body", "feasible_swept")})
    save_csv(rows, "monte_carlo.csv")

    ok = {(r["planner"], r["seed"]): r["outcome"] == "success" for r in rows}
    summary = {"n_trials": len(SEEDS),
               "n_feasible_body": sum(f["feasible_body"] for f in fz.values()),
               "n_feasible_swept": sum(f["feasible_swept"] for f in fz.values()),
               "planners": {}, "paired": {}}
    for pn in PLANNERS:
        sel = [r for r in rows if r["planner"] == pn]
        out = {}
        for den in ("all", "feasible_body", "feasible_swept"):
            s2 = sel if den == "all" else [r for r in sel if r[den]]
            k = sum(r["outcome"] == "success" for r in s2)
            out[den] = {"success": k, "n": len(s2), "rate": k / max(1, len(s2)),
                        "ci95": wilson(k, len(s2))}
        succ = [r for r in sel if r["outcome"] == "success"]
        out["outcomes"] = dict(Counter(r["outcome"] for r in sel))
        out["peak_gamma_max"] = max(r["peak_gamma_deg"] for r in sel)
        out["mean_path_length_success"] = sum(r["path_length"] for r in succ) / max(1, len(succ))
        out["mean_chatter_success"] = sum(r["chatter_deg"] for r in succ) / max(1, len(succ))
        out["min_clearance_success"] = min((r["min_clearance"] for r in succ), default=None)
        summary["planners"][pn] = out
    for other in ("APF", "CPF"):
        b = sum(ok[("SCPF", s)] and not ok[(other, s)] for s in SEEDS)
        c = sum(ok[(other, s)] and not ok[("SCPF", s)] for s in SEEDS)
        summary["paired"][f"SCPF_vs_{other}"] = {"scpf_only": b, "other_only": c,
                                                  "mcnemar_p": mcnemar_exact(b, c)}
    save_json(summary, "monte_carlo_summary.json")
    for pn, o in summary["planners"].items():
        a = o["all"]
        print(f"{pn}: {a['success']}/{a['n']} CI=[{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}] "
              f"feas_swept {o['feasible_swept']['success']}/{o['feasible_swept']['n']} {o['outcomes']} "
              f"gmax={o['peak_gamma_max']:.1f}")
    print(summary["paired"], "feasible body/swept:", summary["n_feasible_body"], summary["n_feasible_swept"])

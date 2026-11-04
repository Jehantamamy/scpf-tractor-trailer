"""Dynamic crossing obstacle: speeds, approach angles and arrival phases (Section 4.4, Fig. 12)."""
from collections import defaultdict
from common import run_all, save_csv, save_json, PLANNERS

SPEEDS = [0.4, 0.8, 1.2, 1.6]
ANGLES = [45.0, 90.0, 135.0]
PHASES = [4.5, 5.5, 6.5]
FIG_CASE = (0.8, 90.0, 5.5)

if __name__ == "__main__":
    cases = []
    for sp in SPEEDS:
        for an in ANGLES:
            for ph in PHASES:
                for pn in PLANNERS:
                    cases.append(dict(planner=pn, scene="dynamic",
                                      dyn={"speed": sp, "angle_deg": an, "t_cross": ph},
                                      record=(sp, an, ph) == FIG_CASE))
    rows = run_all(cases)
    save_csv(rows, "dynamic.csv")
    save_json({r["planner"]: r["log"] for r in rows if "log" in r}, "dynamic_logs.json")
    summ = defaultdict(dict)
    for pn in PLANNERS:
        for sp in SPEEDS:
            sel = [r for r in rows if r["planner"] == pn and r["dyn_speed"] == sp]
            summ[pn][str(sp)] = {"success": sum(r["outcome"] == "success" for r in sel), "n": len(sel),
                                 "min_clearance": min(r["min_clearance"] for r in sel),
                                 "peak_gamma_max": max(r["peak_gamma_deg"] for r in sel)}
        tot = [r for r in rows if r["planner"] == pn]
        summ[pn]["all"] = {"success": sum(r["outcome"] == "success" for r in tot), "n": len(tot)}
        print(pn, {k: (v["success"], v["n"]) for k, v in summ[pn].items()})
    save_json(summ, "dynamic_summary.json")

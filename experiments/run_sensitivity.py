"""One-at-a-time sensitivity sweeps of SCPF gains (Fig. 10), evaluated on the
100 Monte Carlo layouts and the escape start set."""
from collections import defaultdict
from common import run_all, save_csv, save_json, Params
from scpf.scenarios import escape_starts

SWEEPS = {
    "k_jk": [2.0, 5.0, 10.0, 15.0, 20.0],
    "k_gamma": [0.0, 0.75, 1.5, 2.25, 3.0],
    "k_t": [1.0, 1.75, 2.5, 3.25, 4.0],
    "k_rep": [5.0, 7.5, 10.0, 12.5, 15.0],
}

if __name__ == "__main__":
    cases = []
    for par, vals in SWEEPS.items():
        for v in vals:
            ov = {par: v}
            for sd in range(100):
                cases.append(dict(planner="SCPF", scene="mc", seed=sd, overrides=ov,
                                  sweep=par, value=v, suite="monte_carlo"))
            for (x, y, th) in escape_starts():
                cases.append(dict(planner="SCPF", scene="cshape_escape", start=(x, y), theta0=th,
                                  overrides=ov, sweep=par, value=v, suite="escape_set"))
    rows = run_all(cases)
    for r in rows:
        r.pop("theta0", None)
    save_csv(rows, "sensitivity.csv")
    summ = defaultdict(list)
    for par, vals in SWEEPS.items():
        for v in vals:
            sel = [r for r in rows if r["sweep"] == par and r["value"] == v]
            ent = {"value": v}
            for suite in ("monte_carlo", "escape_set"):
                s2 = [r for r in sel if r["suite"] == suite]
                succ = [r for r in s2 if r["outcome"] == "success"]
                ent[suite] = {
                    "success_rate": len(succ) / len(s2),
                    "jackknife": sum(r["outcome"] == "jackknife" for r in s2),
                    "peak_gamma_max": max(r["peak_gamma_deg"] for r in s2),
                    "peak_gamma_mean": sum(r["peak_gamma_deg"] for r in s2) / len(s2),
                    "chatter_mean": sum(r["chatter_deg"] for r in succ) / max(1, len(succ)),
                    "clearance_mean": sum(r["min_clearance"] for r in succ) / max(1, len(succ)),
                    "path_mean": sum(r["path_length"] for r in succ) / max(1, len(succ)),
                }
            summ[par].append(ent)
            print(par, v, {s: round(ent[s]["success_rate"], 2) for s in ("monte_carlo", "escape_set")},
                  round(ent["escape_set"]["peak_gamma_max"], 1), round(ent["monte_carlo"]["peak_gamma_max"], 1))
    save_json(summ, "sensitivity_summary.json")

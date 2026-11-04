"""Canonical scenarios: APF, CPF and SCPF on six fixed scenes (Table 2, Figs. 6-9)."""
from common import run_all, save_csv, save_json, PLANNERS

SCENES = ["baseline", "cshape_trap", "cshape_entrance", "cshape_escape", "zigzag", "clutter"]

if __name__ == "__main__":
    cases = [dict(planner=pn, scene=s, record=True) for s in SCENES for pn in PLANNERS]
    rows = run_all(cases)
    save_csv(rows, "canonical.csv")
    save_json({f"{r['scene']}|{r['planner']}": r["log"] for r in rows}, "canonical_logs.json")
    for r in rows:
        print(f"{r['scene']:16s} {r['planner']:5s} {r['outcome']:9s} L={r['path_length']:6.1f} "
              f"gmax={r['peak_gamma_deg']:5.1f} clr={r['min_clearance']:5.2f} "
              f"chat={r['chatter_deg']:5.2f} gov={r['governor_steps']}")

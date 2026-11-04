"""Escape benchmark: all admissible start poses inside the C-shape cavity."""
import math
from collections import Counter
from common import run_all, save_csv, PLANNERS
from scpf.scenarios import escape_starts

if __name__ == "__main__":
    starts = escape_starts()
    cases = [dict(planner=pn, scene="cshape_escape", start=(x, y), theta0=th,
                  start_heading_deg=round(math.degrees(th)))
             for pn in PLANNERS for (x, y, th) in starts]
    rows = run_all(cases)
    for r in rows:
        r.pop("theta0", None)
    save_csv(rows, "escape_set.csv")
    for pn in PLANNERS:
        c = Counter(r["outcome"] for r in rows if r["planner"] == pn)
        print(pn, len(starts), dict(c))

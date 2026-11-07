"""Benchmark scenes. Coordinates in metres, headings in radians."""
import math
import numpy as np
from scipy import ndimage
from shapely.geometry import Point, Polygon

C_SHAPE = [[6.0, 3.0], [15.0, 3.0], [15.0, 13.0], [6.0, 13.0], [6.0, 9.75], [7.0, 9.75],
           [7.0, 12.0], [14.0, 12.0], [14.0, 4.0], [7.0, 4.0], [7.0, 6.25], [6.0, 6.25]]

ZIGZAG = [
    [[4.0, 0.0], [6.0, 0.0], [6.0, 10.0], [4.0, 10.0]],
    [[9.0, 5.0], [10.0, 5.0], [10.0, 25.0], [0.0, 25.0], [0.0, 14.0], [9.0, 14.0]],
]

DENSE = [
    ((2.2, 11.2), 0.60), ((4.6, 12.0), 0.60), ((2.5, 5.8), 0.60), ((3.7, 7.2), 0.60),
    ((2.3, 8.8), 0.60), ((5.5, 9.5), 0.65), ((7.0, 8.9), 0.65), ((8.65, 7.85), 0.60),
    ((7.2, 12.4), 0.60), ((9.5, 12.7), 0.60), ((11.5, 12.7), 0.55), ((13.3, 12.85), 0.55),
    ((5.6, 2.2), 0.60), ((7.3, 4.0), 0.60), ((9.2, 3.4), 0.60), ((10.6, 3.6), 0.60),
    ((11.9, 5.0), 0.60), ((13.0, 6.4), 0.55), ((13.6, 7.8), 0.55), ((13.8, 9.4), 0.55),
]


def _circle(c, r, vel=(0.0, 0.0)):
    return {"type": "circle", "pos": list(c), "rad": r, "vel": list(vel)}


def _poly(v):
    return {"type": "polygon", "vertices": v}


def scenario(name):
    """Return a dict with obstacles, start, goal, theta0 and plotting limits."""
    if name == "baseline":
        return dict(obstacles=[_circle((7.0, 7.0), 1.5)], start=(2.0, 2.0), goal=(13.0, 13.0),
                    theta0=0.785, xlim=(0, 15), ylim=(0, 15))
    if name == "cshape_trap":
        return dict(obstacles=[_poly(C_SHAPE)], start=(1.5, 8.0), goal=(19.0, 8.0),
                    theta0=0.0, xlim=(0, 20), ylim=(0, 16))
    if name == "cshape_entrance":
        return dict(obstacles=[_poly(C_SHAPE)], start=(1.5, 15.0), goal=(11.0, 8.0),
                    theta0=0.0, xlim=(0, 15), ylim=(0, 15))
    if name == "cshape_escape":
        return dict(obstacles=[_poly(C_SHAPE)], start=(10.0, 8.0), goal=(19.0, 8.0),
                    theta0=0.0, xlim=(0, 20), ylim=(0, 16))
    if name == "zigzag":
        return dict(obstacles=[_poly(v) for v in ZIGZAG], start=(3.0, 3.0), goal=(13.0, 13.0),
                    theta0=math.pi / 2, xlim=(0, 15), ylim=(0, 16))
    if name == "clutter":
        return dict(obstacles=[_circle(c, r) for c, r in DENSE], start=(1.8, 2.2),
                    goal=(13.0, 12.0), theta0=None, xlim=(0, 15), ylim=(0, 15))
    raise KeyError(name)


def dynamic_scenario(speed=1.2, angle_deg=90.0, t_cross=5.5):
    """A circular obstacle (r = 1 m) crossing the straight start-goal line.

    The obstacle is launched so that, at constant speed, it would reach the
    line midpoint (7.5, 7.5) after t_cross seconds; angle_deg is the angle
    between its velocity and the vehicle's nominal heading (+x).
    """
    a = math.radians(angle_deg)
    vel = (speed * math.cos(a), speed * math.sin(a))
    c = (7.5 - vel[0] * t_cross, 7.5 - vel[1] * t_cross)
    return dict(obstacles=[_circle(c, 1.0, vel)], start=(2.0, 7.5), goal=(13.0, 7.5),
                theta0=0.0, xlim=(0, 15), ylim=(0, 15))


MC_FIELD = 50.0


def escape_starts(min_clearance=0.3):
    """Start poses (x, y, heading) inside the C-shape cavity whose tractor-trailer
    footprint keeps at least `min_clearance` from the walls. The trailer is
    initially aligned with the tractor."""
    from .config import Params
    from .vehicle import footprint
    p = Params()
    wall = Polygon(C_SHAPE)
    out = []
    for x in (9.0, 10.5, 12.0):
        for y in (6.0, 8.0, 10.0):
            for th_deg in (0, 90, 180, 270):
                th = math.radians(th_deg)
                tr, tl = footprint((x, y, th, th), p)
                if min(tr.distance(wall), tl.distance(wall)) >= min_clearance:
                    out.append((x, y, th))
    return out
MC_GOAL = (25.0, 25.0)


def monte_carlo_scene(seed):
    """Dense Clustered Yard (Section 4.3): Central goal at (25, 25) with 35-50 circular
    obstacles arranged in 8-11 dense clusters with narrow winding corridors and concavities,
    plus corridor obstacles, with randomized perimeter start (R in [18.5, 21.5] m) oriented
    inward toward the central goal."""
    rng = np.random.default_rng(seed)
    goal = MC_GOAL
    phi = float(rng.uniform(0, 2 * math.pi))
    R = float(rng.uniform(18.5, 21.5))
    sx = float(np.clip(goal[0] + R * math.cos(phi), 4.0, 46.0))
    sy = float(np.clip(goal[1] + R * math.sin(phi), 4.0, 46.0))
    start = (sx, sy)
    th = math.atan2(goal[1] - sy, goal[0] - sx)

    n_clusters = int(rng.integers(8, 11))
    obs, cluster_centers = [], []
    for _ in range(n_clusters):
        for _ in range(500):
            ang = float(rng.uniform(0, 2 * math.pi))
            dist = float(rng.uniform(5.5, 17.0))
            ccx = goal[0] + dist * math.cos(ang)
            ccy = goal[1] + dist * math.sin(ang)
            if all(np.hypot(ccx - ox, ccy - oy) >= 5.0 for ox, oy in cluster_centers):
                cluster_centers.append((ccx, ccy))
                break

    for ccx, ccy in cluster_centers:
        n_in_c = int(rng.integers(3, 6))
        for _ in range(n_in_c):
            for _ in range(200):
                ox = ccx + float(rng.uniform(-2.2, 2.2))
                oy = ccy + float(rng.uniform(-2.2, 2.2))
                rad = float(rng.uniform(0.75, 1.25))
                if np.hypot(ox - start[0], oy - start[1]) < rad + 3.2:
                    continue
                if np.hypot(ox - goal[0], oy - goal[1]) < rad + 2.5:
                    continue
                if any(np.hypot(ox - o["pos"][0], oy - o["pos"][1]) < rad + o["rad"] + 0.3 for o in obs):
                    continue
                obs.append({"type": "circle", "pos": [ox, oy], "rad": rad})
                break

    for _ in range(8):
        for _ in range(200):
            ang = float(rng.uniform(0, 2 * math.pi))
            dist = float(rng.uniform(4.5, 18.5))
            ox = goal[0] + dist * math.cos(ang)
            oy = goal[1] + dist * math.sin(ang)
            rad = float(rng.uniform(0.7, 1.1))
            if np.hypot(ox - start[0], oy - start[1]) < rad + 3.2:
                continue
            if np.hypot(ox - goal[0], oy - goal[1]) < rad + 2.5:
                continue
            if any(np.hypot(ox - o["pos"][0], oy - o["pos"][1]) < rad + o["rad"] + 2.4 for o in obs):
                continue
            obs.append({"type": "circle", "pos": [ox, oy], "rad": rad})
            break

    return dict(obstacles=obs, start=start, goal=goal, theta0=th,
                xlim=(0, MC_FIELD), ylim=(0, MC_FIELD), n_obstacles=len(obs))



def feasible(obstacles, start, goal, inflation, res=0.25, extent=(0.0, 50.0)):
    """Grid connectivity test (equivalent to an A* existence query) on the free
    space inflated by `inflation` metres."""
    lo, hi = extent
    n = int(round((hi - lo) / res)) + 1
    xs = lo + res * np.arange(n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    occ = np.zeros_like(X, dtype=bool)
    for o in obstacles:
        if o.get("type", "circle") == "circle":
            occ |= np.hypot(X - o["pos"][0], Y - o["pos"][1]) <= o["rad"] + inflation
        else:
            g = Polygon(o["vertices"]).buffer(inflation)
            minx, miny, maxx, maxy = g.bounds
            idx = np.where((X >= minx) & (X <= maxx) & (Y >= miny) & (Y <= maxy))
            for i, j in zip(*idx):
                if g.contains(Point(X[i, j], Y[i, j])):
                    occ[i, j] = True
    lab, _ = ndimage.label(~occ)
    si = tuple(int(round((s - lo) / res)) for s in start)
    gi = tuple(int(round((g - lo) / res)) for g in goal)
    return bool(lab[si] != 0 and lab[si] == lab[gi])

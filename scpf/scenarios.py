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

DENSE = [((2.5, 11.5), 1.1), ((6.0, 12.8), 1.1), ((10.0, 13.5), 1.0), ((5.8, 2.2), 1.1),
         ((10.5, 2.5), 1.1), ((3.0, 6.2), 1.0), ((7.2, 5.5), 1.0), ((5.6, 9.2), 1.0),
         ((9.4, 8.8), 1.0), ((11.8, 10.4), 0.85)]


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
MC_START = (3.0, 3.0)
MC_GOAL = (47.0, 47.0)


def monte_carlo_scene(seed):
    """Random yard: 15-30 circular obstacles with radius U[0.8, 2.0] m, uniform
    centres in [5, 45]^2, rejecting obstacles that overlap each other or lie
    within 4 m of the start or goal."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(15, 31))
    obs, tries = [], 0
    while len(obs) < n and tries < 5000:
        tries += 1
        c = rng.uniform(5.0, 45.0, 2)
        r = float(rng.uniform(0.8, 2.0))
        if np.linalg.norm(c - MC_START) < r + 4.0 or np.linalg.norm(c - MC_GOAL) < r + 4.0:
            continue
        if any(np.linalg.norm(c - np.asarray(o["pos"])) < r + o["rad"] for o in obs):
            continue
        obs.append(_circle(c.tolist(), r))
    th = math.atan2(MC_GOAL[1] - MC_START[1], MC_GOAL[0] - MC_START[0])
    return dict(obstacles=obs, start=MC_START, goal=MC_GOAL, theta0=th,
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

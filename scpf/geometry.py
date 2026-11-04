"""Obstacle geometry helpers (circles, polygons and capsule-shaped virtual walls)."""
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points


def make_obstacle(obs):
    """Return a copy of an obstacle dict with a cached shapely geometry."""
    o = dict(obs)
    t = o.get("type", "circle")
    if t == "circle":
        o["pos"] = np.asarray(o["pos"], dtype=float)
        o["vel"] = np.asarray(o.get("vel", [0.0, 0.0]), dtype=float)
        o["geom"] = Point(o["pos"]).buffer(o["rad"], resolution=24)
    elif t == "polygon":
        o["vertices"] = [list(map(float, v)) for v in o["vertices"]]
        o["geom"] = Polygon(o["vertices"])
    elif t == "wall":
        o["geom"] = LineString([o["p1"], o["p2"]]).buffer(o.get("rad", 0.0), resolution=12)
    return o


def update_dynamic(obstacles, dt):
    for o in obstacles:
        if o.get("type", "circle") == "circle" and np.any(o["vel"] != 0.0):
            o["pos"] = o["pos"] + o["vel"] * dt
            o["geom"] = Point(o["pos"]).buffer(o["rad"], resolution=24)


def closest_point(p, obs):
    """Signed surface distance from point p to an obstacle and the nearest surface point."""
    p = np.asarray(p, dtype=float)
    t = obs.get("type", "circle")
    if t == "circle":
        v = p - obs["pos"]
        dc = float(np.hypot(v[0], v[1]))
        if dc < 1e-9:
            return -obs["rad"], obs["pos"].copy()
        return dc - obs["rad"], obs["pos"] + v / dc * obs["rad"]
    if t == "wall":
        p1 = np.asarray(obs["p1"], float)
        p2 = np.asarray(obs["p2"], float)
        seg = p2 - p1
        l2 = float(seg @ seg)
        s = 0.0 if l2 == 0 else float(np.clip((p - p1) @ seg / l2, 0.0, 1.0))
        c = p1 + s * seg
        v = p - c
        dc = float(np.hypot(v[0], v[1]))
        r = float(obs.get("rad", 0.0))
        if dc < 1e-9:
            return -r, c
        return dc - r, c + v / dc * r
    poly = obs["geom"]
    pt = Point(p[0], p[1])
    d = poly.distance(pt)
    q, _ = nearest_points(poly.exterior, pt)
    if d == 0.0:
        d = -poly.exterior.distance(pt)
    return float(d), np.array([q.x, q.y])


def obstacle_distance(geom_a, obs):
    return float(geom_a.distance(obs["geom"]))

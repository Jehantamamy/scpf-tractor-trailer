"""Local perception: range-limited sensing, gap screening, rolling convex hulls,
virtual lids and topological subgoal generation (Sections 3.1-3.2)."""
import math
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from .geometry import make_obstacle


def sense(obstacles, tractor_pos, R_sense):
    """Object-level sensing: an obstacle is perceived when any part lies within R_sense."""
    pt = Point(tractor_pos[0], tractor_pos[1])
    return [o for o in obstacles if o["geom"].distance(pt) <= R_sense]


def convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def _gap(a, b):
    if a.get("type", "circle") == "circle" and b.get("type", "circle") == "circle":
        return float(np.linalg.norm(a["pos"] - b["pos"]) - a["rad"] - b["rad"])
    return float(a["geom"].distance(b["geom"]))


def _components(obs, gap_thr):
    n = len(obs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if _gap(obs[i], obs[j]) < gap_thr:
                edges.append((i, j))
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), set(edges)


def _los_clear(a, b, obs, skip):
    line = LineString([a, b])
    for k, o in enumerate(obs):
        if k in skip:
            continue
        if line.intersects(o["geom"].buffer(-0.05)):
            return False
    return True


def perceive(perceived, goal, tractor_pos, tractor_ori, p):
    """Returns (augmented obstacle list, subgoal list, info).

    Obstacles separated by less than the screening threshold are merged into one
    cluster; apertures inside a cluster are sealed by capsule walls (gap screening).
    For clusters with a concave envelope, non-boundary hull edges become virtual
    lids unless the goal lies inside the hull (entrance) or the vehicle lies
    inside it (escape), in which case a subgoal is placed outside/inside the portal.
    """
    goal = np.asarray(goal, float)
    start = np.asarray(tractor_pos, float)
    trailer_pt = start - 1.5 * np.array([math.cos(tractor_ori), math.sin(tractor_ori)])
    info = {"n_clusters": 0, "n_lids": 0, "needs_escape": False}
    if not p.use_hull or len(perceived) == 0:
        return list(perceived), [], info

    gap_thr = p.W_swept if p.use_gap_screening else p.body_width
    groups, edges = _components(perceived, gap_thr)
    out = []
    signs = {}
    subgoals = []
    v_head = np.array([math.cos(tractor_ori), math.sin(tractor_ori)])

    for comp in groups:
        is_poly = any(perceived[i].get("type") == "polygon" for i in comp)
        if len(comp) < 2 and not is_poly:
            continue
        info["n_clusters"] += 1
        centroid = np.mean([_centroid(perceived[i]) for i in comp], axis=0)
        v_cr = start - centroid
        v_cr = v_cr / (np.linalg.norm(v_cr) + 1e-9)
        v_cg = goal - centroid
        v_cg = v_cg / (np.linalg.norm(v_cg) + 1e-9)
        t_cw = np.array([v_cr[1], -v_cr[0]])
        w_head = 2.5 if np.linalg.norm(start - centroid) < 6.0 else 0.2
        s_cw = t_cw @ v_cg + w_head * (t_cw @ v_head)
        s_ccw = -t_cw @ v_cg + w_head * (-t_cw @ v_head)
        sign = -1.0 if s_cw > s_ccw else 1.0
        cid = ("cluster",) + tuple(sorted(str(perceived[i].get("key", i)) for i in comp))[:1]
        for i in comp:
            signs[i] = (sign, cid)

        circles = [i for i in comp if perceived[i].get("type", "circle") == "circle"]
        # seal sub-threshold apertures between clustered circles
        for a in range(len(circles)):
            for b in range(a + 1, len(circles)):
                i, j = circles[a], circles[b]
                if (min(i, j), max(i, j)) in edges:
                    out.append(make_obstacle({"type": "wall", "p1": perceived[i]["pos"].tolist(),
                                              "p2": perceived[j]["pos"].tolist(),
                                              "rad": min(perceived[i]["rad"], perceived[j]["rad"]),
                                              "cluster_sign": sign, "virtual": True, "cluster": cid,
                                              "key": ("seal", perceived[i].get("key", i),
                                                      perceived[j].get("key", j))}))
                    info["n_lids"] += 1

        # hull of the cluster (circle centres or polygon vertices)
        pts, owner, own_edges, buf = [], {}, set(), 0.0
        for i in comp:
            o = perceived[i]
            if o.get("type", "circle") == "circle":
                key = (round(float(o["pos"][0]), 4), round(float(o["pos"][1]), 4))
                pts.append(key); owner[key] = i; buf = max(buf, o["rad"])
            else:
                vs = o["vertices"]
                for k, v in enumerate(vs):
                    key = (round(v[0], 4), round(v[1], 4))
                    nxt = vs[(k + 1) % len(vs)]
                    key2 = (round(nxt[0], 4), round(nxt[1], 4))
                    pts.append(key); owner[key] = i
                    own_edges.add((key, key2)); own_edges.add((key2, key))
        hull = convex_hull(pts)
        if len(hull) < 3:
            continue
        hull_poly = Polygon(hull).buffer(max(buf, 0.1))
        goal_in = hull_poly.contains(Point(goal))
        veh_in = hull_poly.contains(Point(start)) or hull_poly.contains(Point(trailer_pt))
        needs_escape = veh_in and not goal_in
        needs_entrance = goal_in and not veh_in
        if needs_escape:
            info["needs_escape"] = True
        candidates = []
        union = unary_union([perceived[i]["geom"] for i in comp]).buffer(0.05)
        portal_min = p.W_swept if p.use_gap_screening else p.body_width
        for k in range(len(hull)):
            a_k, b_k = hull[k], hull[(k + 1) % len(hull)]
            if (a_k, b_k) in own_edges:
                continue
            if owner.get(a_k) == owner.get(b_k) and perceived[owner[a_k]].get("type", "circle") == "circle":
                continue
            A, B = np.array(a_k), np.array(b_k)
            seg_len = float(np.linalg.norm(B - A))
            if seg_len < 0.1:
                continue
            line = LineString([A, B])
            blocked = any(line.distance(perceived[m]["geom"]) < p.W_swept
                          for m in range(len(perceived)) if m not in comp)
            if blocked:
                continue
            if (needs_escape or needs_entrance):
                free = line.difference(union)
                parts = list(getattr(free, "geoms", [free])) if not free.is_empty else []
                parts = [s for s in parts if s.length >= portal_min]
                if parts:
                    seg = max(parts, key=lambda s: s.length)
                    mid = np.array(seg.interpolate(0.5, normalized=True).coords[0])
                    n = np.array([-(B - A)[1], (B - A)[0]]) / seg_len
                    if n @ (mid - centroid) < 0:
                        n = -n
                    inner = None
                    for s_in in (p.stage_dist, 0.75 * p.stage_dist, 0.5 * p.stage_dist):
                        q = mid - s_in * n
                        if union.distance(Point(q)) >= 1.0:
                            inner = q
                            break
                    if inner is not None:
                        outer = mid + p.stage_dist * n
                        seq = [inner, outer] if needs_escape else [outer, inner]
                        if _los_clear(start, seq[0], perceived, set()):
                            candidates.append(seq)
                    continue
            if goal_in:
                continue
            ia, ib = owner.get(a_k), owner.get(b_k)
            radii = [perceived[m]["rad"] for m in (ia, ib)
                     if m is not None and perceived[m].get("type", "circle") == "circle"]
            if radii:
                rad, A2, B2 = min(radii), A, B
            else:
                rad = 0.4
                u = (B - A) / seg_len
                A2, B2 = (A + u * rad, B - u * rad) if seg_len > 2 * rad else (A, B)
            out.append(make_obstacle({"type": "wall", "p1": A2.tolist(), "p2": B2.tolist(),
                                      "rad": rad, "cluster_sign": sign, "virtual": True,
                                      "cluster": cid, "key": ("lid", a_k, b_k)}))
            info["n_lids"] += 1
        if candidates:
            best = min(candidates, key=lambda sq: np.linalg.norm(sq[-1] - goal))
            subgoals.extend(best)

    final = []
    for i, o in enumerate(perceived):
        if i in signs:
            o2 = dict(o); o2["cluster_sign"], o2["cluster"] = signs[i]
            final.append(o2)
        else:
            final.append(o)
    return final + out, subgoals, info


def _centroid(o):
    if o.get("type", "circle") == "circle":
        return o["pos"]
    c = o["geom"].centroid
    return np.array([c.x, c.y])

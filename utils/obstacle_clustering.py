# File: backend/utils/obstacle_clustering.py
import numpy as np
import math
from networkx import Graph, connected_components
from shapely.geometry import Polygon, Point, LineString

def get_convex_hull(points):
    pts = sorted(list(set(points)), key=lambda p: (p[0], p[1]))
    if len(pts) <= 2: return pts
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
        
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
        
    return lower[:-1] + upper[:-1]

# 🚀 FUNGSI BARU: Pengecekan Line-of-Sight (Mata Robot ke Celah)
def is_los_clear(start_pt, end_pt, geoms, comp_ids):
    """Mengecek apakah garis lurus dari start_pt ke end_pt terhalang obstacle lain di luar cluster"""
    path = LineString([start_pt, end_pt])
    for oid, geom in geoms.items():
        if oid in comp_ids: 
            continue # Abaikan obstacle dari cluster (ruangan) yang sama
        # Gunakan buffer negatif kecil agar tidak false positive jika hanya menyenggol pinggiran
        if path.intersects(geom.buffer(-0.05)):
            return False
    return True

def build_geometries(combine):
    geoms = {}
    for obs in combine:
        orig = obs.get("original", obs)
        t = orig.get("type", "circle")
        if t == "polygon":
            geoms[obs["id"]] = Polygon(orig["vertices"])
        elif t == "wall":
            geoms[obs["id"]] = LineString([orig["p1"], orig["p2"]])
        else:
            geoms[obs["id"]] = Point(orig.get("pos", [0,0])).buffer(orig.get("rad", 0.1) + 1e-3)
    return geoms

def find_clusters(combine, geoms, robot_radius):
    G_prox = Graph()   
    for i, oi in enumerate(combine):
        G_prox.add_node(oi["id"])
        geom_i = geoms[oi["id"]]
        for j, oj in enumerate(combine):
            if j <= i: continue
            geom_j = geoms[oj["id"]]
            gap = geom_i.distance(geom_j)
            
            if gap < (1.75 * robot_radius):
                G_prox.add_edge(oi["id"], oj["id"])
    return G_prox

def process_obstacle_clustering(raw_obstacles, goal, start_pos, robot_radius, start_ori=0.0, use_merge=True, use_cluster=True):
    escape_waypoints = [] 
    
    if not use_merge and not use_cluster:
        return raw_obstacles, [], escape_waypoints

    combine = []
    for obs in raw_obstacles:
        if obs.get("type") == "wall": continue 
        
        if obs.get("type") == "polygon":
            pts = np.array(obs["vertices"])
            pos = np.mean(pts, axis=0) 
            rad = np.max(np.linalg.norm(pts - pos, axis=1)) 
        else:
            pos = np.array(obs["pos"]) if isinstance(obs["pos"], list) else obs["pos"]
            rad = obs.get("rad", 0.0)
            
        combine.append({"pos": pos, "rad": rad, "id": obs["id"], "original": obs})

    geoms = build_geometries(combine)
    G_prox = find_clusters(combine, geoms, robot_radius)
    
    start_arr = np.array(start_pos)
    goal_arr = np.array(goal)
    v_head = np.array([math.cos(start_ori), math.sin(start_ori)])
    trailer_pos = start_arr - 1.5 * np.array([math.cos(start_ori), math.sin(start_ori)])

    cluster_signs = {} 
    valid_groups = [list(comp) for comp in connected_components(G_prox)]
    id_to_obs = {obs["id"]: obs for obs in combine}
    
    for comp in valid_groups:
        is_poly = any(id_to_obs[cid]["original"].get("type") == "polygon" for cid in comp)
        if len(comp) > 1 or is_poly:
            positions = [c["pos"] for c in combine if c["id"] in comp]
            centroid = np.mean(positions, axis=0)
            
            v_cr = start_arr - centroid
            dist_cr = np.linalg.norm(v_cr) + 1e-5
            v_cr_norm = v_cr / dist_cr 
            
            v_cg = goal_arr - centroid
            v_cg_norm = v_cg / (np.linalg.norm(v_cg) + 1e-5)
            
            T_cw = np.array([v_cr_norm[1], -v_cr_norm[0]])
            T_ccw = np.array([-v_cr_norm[1], v_cr_norm[0]])
            
            w_goal = 1.0
            w_head = 2.5 if dist_cr < (robot_radius * 6.0) else 0.2
            
            score_cw = w_goal * np.dot(T_cw, v_cg_norm) + w_head * np.dot(T_cw, v_head)
            score_ccw = w_goal * np.dot(T_ccw, v_cg_norm) + w_head * np.dot(T_ccw, v_head)
            
            sign = -1.0 if score_cw > score_ccw else +1.0
            for i in comp:
                 cluster_signs[i] = sign

    pseudo_obstacles = []
    pseudo_id = 5000
    created_bridges = []
    
    def is_duplicate(p1, p2):
        for b in created_bridges:
            if np.linalg.norm(p1 - b[0]) < 0.1 and np.linalg.norm(p2 - b[1]) < 0.1: return True
            if np.linalg.norm(p1 - b[1]) < 0.1 and np.linalg.norm(p2 - b[0]) < 0.1: return True
        return False

    safe_clearance = robot_radius * 2.0

    for comp in valid_groups:
        candidate_escapes = []
        positions = [c["pos"] for c in combine if c["id"] in comp]
        cluster_centroid = np.mean(positions, axis=0)

        # ==========================================================
        # 1. EVALUASI LINGKARAN (KAPSUL)
        # ==========================================================
        circle_ids = [cid for cid in comp if id_to_obs[cid]["original"].get("type", "circle") == "circle"]
        if len(circle_ids) >= 2:
            for i in range(len(circle_ids)):
                for j in range(i+1, len(circle_ids)):
                    id1, id2 = circle_ids[i], circle_ids[j]
                    if G_prox.has_edge(id1, id2):
                        asurf = np.array(id_to_obs[id1]["pos"])
                        bsurf = np.array(id_to_obs[id2]["pos"])
                        if np.linalg.norm(asurf - bsurf) > 0.1:
                            rad_min = min(id_to_obs[id1]["rad"], id_to_obs[id2]["rad"])
                            pseudo_obstacles.append({
                                "id": f"circ_int_{pseudo_id}", "type": "wall",
                                "p1": asurf.tolist(), "p2": bsurf.tolist(),
                                "rad": float(rad_min), 
                                "cluster_sign": cluster_signs.get(id1, 1.0), "is_internal": True
                            })
                            pseudo_id += 1
                            created_bridges.append((asurf, bsurf))
                            
            if len(circle_ids) >= 3:
                all_hull_points = []
                pt_to_obs_id = {}
                max_rad_circle = 0.0
                
                for cid in circle_ids:
                    pos = id_to_obs[cid]["pos"]
                    rad = id_to_obs[cid].get("rad", 0.0)
                    if rad > max_rad_circle: max_rad_circle = rad
                    
                    key = (round(float(pos[0]), 4), round(float(pos[1]), 4))
                    all_hull_points.append(key)
                    pt_to_obs_id[key] = cid
                
                hull_vertices = get_convex_hull(all_hull_points)
                cluster_hull_poly = Polygon(hull_vertices).buffer(max_rad_circle)
                strict_hull = Polygon(hull_vertices).buffer(0.1)
                
                is_goal_inside = cluster_hull_poly.contains(Point(goal))
                is_tractor_inside = strict_hull.contains(Point(start_pos))
                is_trailer_inside = strict_hull.contains(Point(trailer_pos))
                
                # 🚀 RULE 1 (SAMA-SAMA DI DALAM): Jangan buat waypoint kalau target dan robot di satu ruangan!
                if (is_tractor_inside or is_trailer_inside) and is_goal_inside:
                    needs_escape = False
                    needs_entrance = False
                else:
                    needs_escape = (is_tractor_inside or is_trailer_inside) and not is_goal_inside
                    needs_entrance = is_goal_inside and not (is_tractor_inside or is_trailer_inside)

                for i in range(len(hull_vertices)):
                    p1 = hull_vertices[i]
                    p2 = hull_vertices[(i + 1) % len(hull_vertices)]
                    id1 = pt_to_obs_id.get(p1)
                    id2 = pt_to_obs_id.get(p2)
                    
                    if id1 is not None and id2 is not None and id1 != id2:
                        asurf = np.array(p1)
                        bsurf = np.array(p2)
                        
                        if not is_duplicate(asurf, bsurf) and 0.1 < np.linalg.norm(bsurf - asurf) < 15.0:
                            bridge_line = LineString([asurf, bsurf])
                            
                            is_blocked = any(bridge_line.distance(geoms[oid["id"]]) < safe_clearance for oid in combine if oid["id"] not in comp)
                            
                            if not is_blocked:
                                if is_goal_inside or needs_escape:
                                    mid_pt = (asurf + bsurf) / 2.0
                                    target_pt = start_pos if needs_escape else goal
                                    
                                    vec = target_pt - mid_pt
                                    dist_vec = np.linalg.norm(vec)
                                    
                                    if dist_vec <= 0.1:
                                        target_visible = True
                                    else:
                                        offset_mid = mid_pt + (vec / dist_vec) * 0.1
                                        sight_line = LineString([offset_mid, target_pt])
                                        target_visible = True
                                        for cid in circle_ids:
                                            if cid in [id1, id2]: continue
                                            if sight_line.intersects(geoms[cid].buffer(-0.05)):
                                                target_visible = False
                                                break
                                            
                                    if target_visible: 
                                        # 🚀 RULE 2 (CEK LOS TERHALANG):
                                        if not is_los_clear(start_arr, mid_pt, geoms, comp):
                                            continue # Celah terhalang, cari celah lain!

                                        # 🚀 RULE 3 (VEKTOR TEGAK LURUS):
                                        gap_vec = bsurf - asurf
                                        perp_vec = np.array([-gap_vec[1], gap_vec[0]])
                                        perp_norm = perp_vec / (np.linalg.norm(perp_vec) + 1e-5)
                                        
                                        # Pastikan vektor selalu mengarah ke LUAR menjauhi centroid ruangan
                                        vec_from_centroid = mid_pt - cluster_centroid
                                        if np.dot(perp_norm, vec_from_centroid) < 0:
                                            perp_norm = -perp_norm
                                            
                                        if needs_escape:
                                            pushed_pt = mid_pt + perp_norm * 2.5
                                            candidate_escapes.append(pushed_pt)
                                        elif needs_entrance:
                                            dist_to_goal = np.linalg.norm(goal_arr - mid_pt)
                                            push_dist = min(2.5, dist_to_goal * 0.8)
                                            pushed_pt = mid_pt - perp_norm * push_dist # Masuk ke DALAM
                                            candidate_escapes.append(pushed_pt)
                                        continue 
                                
                                rad_min = min(id_to_obs[id1]["rad"], id_to_obs[id2]["rad"])
                                pseudo_obstacles.append({
                                    "id": f"circ_lid_{pseudo_id}", "type": "wall",
                                    "p1": asurf.tolist(), "p2": bsurf.tolist(),
                                    "rad": float(rad_min), 
                                    "cluster_sign": cluster_signs.get(id1, 1.0), "is_internal": False
                                })
                                pseudo_id += 1
                                created_bridges.append((asurf, bsurf))

        # ==========================================================
        # 2. EVALUASI POLIGON
        # ==========================================================
        poly_ids = [cid for cid in comp if id_to_obs[cid]["original"].get("type", "circle") == "polygon"]
        if len(poly_ids) > 0:
            all_verts = []
            original_edges = set()
            
            for pid in poly_ids:
                verts = id_to_obs[pid]["original"]["vertices"]
                for i, v in enumerate(verts):
                    key = (round(float(v[0]), 4), round(float(v[1]), 4))
                    all_verts.append({
                        "pt": np.array(v), "key": key,
                        "obs_id": pid, "index": i, "total": len(verts)
                    })
                    v_next = verts[(i+1)%len(verts)]
                    key_next = (round(float(v_next[0]), 4), round(float(v_next[1]), 4))
                    original_edges.add((key, key_next))
                    original_edges.add((key_next, key))

            hull_pts = get_convex_hull([v["key"] for v in all_verts]) if len(all_verts) >= 3 else []
            
            if len(hull_pts) >= 3:
                strict_hull = Polygon(hull_pts).buffer(0.1)
                
                is_goal_inside = strict_hull.contains(Point(goal))
                is_tractor_inside = strict_hull.contains(Point(start_pos))
                is_trailer_inside = strict_hull.contains(Point(trailer_pos))
                
                # 🚀 RULE 1 (SAMA-SAMA DI DALAM)
                if (is_tractor_inside or is_trailer_inside) and is_goal_inside:
                    needs_escape = False
                    needs_entrance = False
                else:
                    needs_escape = (is_tractor_inside or is_trailer_inside) and not is_goal_inside
                    needs_entrance = is_goal_inside and not (is_tractor_inside or is_trailer_inside)

                for i in range(len(hull_pts)):
                    p1_t = hull_pts[i]
                    p2_t = hull_pts[(i+1)%len(hull_pts)]
                    
                    if (p1_t, p2_t) not in original_edges:
                        asurf = np.array(p1_t)
                        bsurf = np.array(p2_t)
                        
                        if not is_duplicate(asurf, bsurf) and 0.1 < np.linalg.norm(bsurf - asurf) < 15.0:
                            bridge_line = LineString([asurf, bsurf])
                            
                            is_blocked = any(bridge_line.distance(geoms[oid["id"]]) < safe_clearance for oid in combine if oid["id"] not in comp)
                            
                            if not is_blocked:
                                if is_goal_inside or needs_escape:
                                    mid_pt = (asurf + bsurf) / 2.0
                                    target_pt = start_pos if needs_escape else goal
                                    
                                    vec = target_pt - mid_pt
                                    dist_vec = np.linalg.norm(vec)
                                    
                                    if dist_vec <= 0.1:
                                        target_visible = True
                                    else:
                                        offset_mid = mid_pt + (vec / dist_vec) * 0.1
                                        sight_line = LineString([offset_mid, target_pt])
                                        target_visible = True
                                        for pid in poly_ids:
                                            if sight_line.intersects(geoms[pid].buffer(-0.05)):
                                                target_visible = False
                                                break
                                            
                                    if target_visible: 
                                        # 🚀 RULE 2 (CEK LOS TERHALANG):
                                        if not is_los_clear(start_arr, mid_pt, geoms, comp):
                                            continue 

                                        # 🚀 RULE 3 (VEKTOR TEGAK LURUS):
                                        gap_vec = bsurf - asurf
                                        perp_vec = np.array([-gap_vec[1], gap_vec[0]])
                                        perp_norm = perp_vec / (np.linalg.norm(perp_vec) + 1e-5)
                                        
                                        vec_from_centroid = mid_pt - cluster_centroid
                                        if np.dot(perp_norm, vec_from_centroid) < 0:
                                            perp_norm = -perp_norm

                                        if needs_escape:
                                            pushed_pt = mid_pt + perp_norm * 2.5
                                            candidate_escapes.append(pushed_pt)
                                        elif needs_entrance:
                                            dist_to_goal = np.linalg.norm(goal_arr - mid_pt)
                                            push_dist = min(2.5, dist_to_goal * 0.8)
                                            pushed_pt = mid_pt - perp_norm * push_dist # Masuk ke DALAM
                                            candidate_escapes.append(pushed_pt)
                                        continue

                                obs_id = next((v["obs_id"] for v in all_verts if v["key"] == p1_t), list(poly_ids)[0])
                                
                                rad_wall = 0.4
                                vec_edge = bsurf - asurf
                                dist_edge = np.linalg.norm(vec_edge)
                                
                                if dist_edge > 2 * rad_wall:
                                    dir_norm = vec_edge / dist_edge
                                    p1_shrink = asurf + dir_norm * rad_wall
                                    p2_shrink = bsurf - dir_norm * rad_wall
                                else:
                                    mid_pt = (asurf + bsurf) / 2.0
                                    p1_shrink = mid_pt
                                    p2_shrink = mid_pt
                                
                                pseudo_obstacles.append({
                                    "id": f"poly_lid_{pseudo_id}", "type": "wall",
                                    "p1": p1_shrink.tolist(), "p2": p2_shrink.tolist(),
                                    "rad": rad_wall, 
                                    "cluster_sign": cluster_signs.get(obs_id, 1.0), "is_internal": False
                                })
                                pseudo_id += 1
                                created_bridges.append((asurf, bsurf))

        # Hanya pilih 1 escape waypoint terbaik (paling dekat dengan Goal)
        if candidate_escapes:
            best_escape = min(candidate_escapes, key=lambda pt: np.linalg.norm(pt - goal_arr))
            dist_to_escape = np.linalg.norm(best_escape - start_arr)
            
            if dist_to_escape > 1.5: 
                escape_waypoints.append(best_escape.tolist())

    final_obstacles = []
    for obs in combine:
        obs_copy = dict(obs["original"])
        if obs["id"] in cluster_signs:
            obs_copy["cluster_sign"] = cluster_signs[obs["id"]]
        final_obstacles.append(obs_copy)

    for po in pseudo_obstacles:
        po_copy = dict(po)
        po_copy["is_dynamic"] = False
        final_obstacles.append(po_copy)

    return final_obstacles, pseudo_obstacles, escape_waypoints
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

def get_closest_point_and_distance(robot_pos, obs):
    """
    Menghitung jarak permukaan terdekat dan koordinat titik ancaman.
    Mendukung 3 bentuk: 'polygon' (Shapely), 'wall' (Kapsul), dan 'circle' (Native).
    """
    if isinstance(robot_pos, list):
        robot_pos = np.array(robot_pos)
        
    obs_type = obs.get("type", "circle")
    
    # ==========================================
    # 1. HANDLING BENTUK POLIGON (SHAPELY)
    # ==========================================
    if obs_type == "polygon":
        robot_pt = Point(robot_pos[0], robot_pos[1])
        poly = Polygon(obs["vertices"])
        
        dist_surf = poly.distance(robot_pt)
        if dist_surf == 0.0:
            dist_surf = -poly.exterior.distance(robot_pt) 
            
        closest_poly_pt, _ = nearest_points(poly, robot_pt)
        nearest_coord = np.array([closest_poly_pt.x, closest_poly_pt.y])
        
        return dist_surf, nearest_coord
        
    # ==========================================
    # 2. HANDLING BENTUK TEMBOK KAPSUL (P1 - P2)
    # ==========================================
    elif obs_type == "wall":
        p1 = np.array(obs["p1"])
        p2 = np.array(obs["p2"])
        wall_rad = float(obs.get("rad", 0.0))
        
        line_vec = p2 - p1
        l2 = np.sum(line_vec**2)
        
        # Cari titik terdekat pada segmen garis
        if l2 == 0:
            c_pt = p1
        else:
            t = np.clip(np.dot(robot_pos - p1, line_vec) / (l2 + 1e-6), 0.0, 1.0)
            c_pt = p1 + t * line_vec
            
        vec_to_obs = robot_pos - c_pt
        dist_center = np.linalg.norm(vec_to_obs)
        dist_surf = dist_center - wall_rad
        
        # Cari koordinat di kulit luar kapsul
        if dist_center < 1e-6:
            nearest_coord = c_pt
        else:
            nearest_coord = c_pt + (vec_to_obs / dist_center) * wall_rad
            
        return dist_surf, nearest_coord
        
    # ==========================================
    # 3. HANDLING BENTUK LINGKARAN (DEFAULT)
    # ==========================================
    else:
        # Fallback aman jika 'pos' formatnya list atau array
        pos_val = obs.get("pos", [0.0, 0.0])
        obs_pos = np.array(pos_val) if isinstance(pos_val, list) else pos_val
        obs_rad = float(obs.get("rad", 0.0))
        
        vec_to_obs = robot_pos - obs_pos
        dist_center = np.linalg.norm(vec_to_obs)
        dist_surf = dist_center - obs_rad
        
        if dist_center < 1e-6:
            nearest_coord = obs_pos
        else:
            nearest_coord = obs_pos + (vec_to_obs / dist_center) * obs_rad
            
        return dist_surf, nearest_coord
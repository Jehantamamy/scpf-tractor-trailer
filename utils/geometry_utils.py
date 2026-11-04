"""Geometric distance and surface normal calculation utilities."""
from typing import Dict, Tuple, Any
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


def get_closest_point_and_distance(robot_pos: np.ndarray, obs: Dict[str, Any]) -> Tuple[float, np.ndarray]:
    """Calculate Euclidean clearance distance and nearest obstacle surface point.

    Supports polygonal obstacles, wall segments (capsule hulls), and circular obstacles.

    Args:
        robot_pos: Coordinate array [x, y].
        obs: Obstacle dictionary.

    Returns:
        Tuple of (clearance_distance, closest_surface_point).
    """
    if isinstance(robot_pos, list):
        robot_pos = np.array(robot_pos, dtype=float)

    obs_type = obs.get("type", "circle")

    # 1. Polygonal obstacle geometry
    if obs_type == "polygon":
        robot_pt = Point(robot_pos[0], robot_pos[1])
        poly = Polygon(obs["vertices"])

        dist_surf = float(poly.distance(robot_pt))
        if dist_surf == 0.0:
            dist_surf = float(-poly.exterior.distance(robot_pt))

        closest_poly_pt, _ = nearest_points(poly, robot_pt)
        nearest_coord = np.array([closest_poly_pt.x, closest_poly_pt.y])
        return dist_surf, nearest_coord

    # 2. Linear wall segment (capsule geometry)
    elif obs_type == "wall":
        p1 = np.array(obs["p1"], dtype=float)
        p2 = np.array(obs["p2"], dtype=float)
        wall_rad = float(obs.get("rad", 0.0))

        line_vec = p2 - p1
        l2 = float(np.sum(line_vec ** 2))

        if l2 == 0.0:
            c_pt = p1
        else:
            t = float(np.clip(np.dot(robot_pos - p1, line_vec) / (l2 + 1e-6), 0.0, 1.0))
            c_pt = p1 + t * line_vec

        vec_to_obs = robot_pos - c_pt
        dist_center = float(np.linalg.norm(vec_to_obs))
        dist_surf = dist_center - wall_rad

        if dist_center < 1e-6:
            nearest_coord = c_pt
        else:
            nearest_coord = c_pt + (vec_to_obs / dist_center) * wall_rad

        return dist_surf, nearest_coord

    # 3. Circular obstacle geometry
    else:
        pos_val = obs.get("pos", [0.0, 0.0])
        obs_pos = np.array(pos_val, dtype=float) if isinstance(pos_val, list) else pos_val
        obs_rad = float(obs.get("rad", 0.0))

        vec_to_obs = robot_pos - obs_pos
        dist_center = float(np.linalg.norm(vec_to_obs))
        dist_surf = dist_center - obs_rad

        if dist_center < 1e-6:
            nearest_coord = obs_pos
        else:
            nearest_coord = obs_pos + (vec_to_obs / dist_center) * obs_rad

        return dist_surf, nearest_coord
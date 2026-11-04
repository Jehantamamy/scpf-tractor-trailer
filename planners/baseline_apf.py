"""Classical Khatib Artificial Potential Field (APF) Baseline Planner."""
import math
from typing import Dict, List, Tuple
import numpy as np

from .base_planner import BasePlanner
from utils.geometry_utils import get_closest_point_and_distance


class BaselineAPFPlanner(BasePlanner):
    """Classical Khatib APF planner with repulsive potential and no circulation."""

    def calculate_control(self, t_pos: np.ndarray, t_ori: float, tr_ori: float,
                          obstacles: List[Dict], escape_wps: List[np.ndarray] = None) -> Tuple[float, List[float], float]:
        """Compute steering control command using standard APF.

        Args:
            t_pos: Tractor position [x, y].
            t_ori: Tractor heading angle (rad).
            tr_ori: Trailer heading angle (rad).
            obstacles: List of obstacle geometry dictionaries.
            escape_wps: Ignored in baseline APF.

        Returns:
            Tuple of (delta_cmd, total_force, tau_jk).
        """
        goal = np.array(self.config.goal)
        k_att = self.config.k_att
        k_rep = self.config.k_rep
        rho_0 = self.config.rho_0
        phi_max = np.radians(self.config.max_steer_deg)

        # Linear attractive force
        dir_to_goal = goal - t_pos
        dist_to_goal = float(np.linalg.norm(dir_to_goal)) + 1e-5
        F_att = k_att * (dir_to_goal / dist_to_goal)

        # Pure repulsive force
        F_rep = np.zeros(2)

        for obs in obstacles:
            dist_surf, c_pt = get_closest_point_and_distance(t_pos, obs)

            if dist_surf < rho_0:
                safe_dist = max(dist_surf, 0.05)
                dir_out = (t_pos - c_pt) / (np.linalg.norm(t_pos - c_pt) + 1e-5)

                term1 = (1.0 / safe_dist - 1.0 / rho_0)
                term2 = 1.0 / (safe_dist ** 2)
                F_rep += k_rep * term1 * term2 * dir_out

        F_total = F_att + F_rep
        total_norm = float(np.linalg.norm(F_total))
        max_circ = getattr(self.config, 'max_circular', 10.0)
        if total_norm > max_circ:
            F_total = (F_total / total_norm) * max_circ

        desired_angle = math.atan2(F_total[1], F_total[0])
        angle_err = (desired_angle - t_ori + math.pi) % (2.0 * math.pi) - math.pi

        delta_final = float(np.clip(3.0 * angle_err, -phi_max, phi_max))

        return delta_final, F_total.tolist(), 0.0

    def calculate_field_matrix(self, X: np.ndarray, Y: np.ndarray, obstacles: List[Dict], escape_wps: List[np.ndarray] = None):
        """Compute full 2D force field vector matrix over spatial grid."""
        goal = np.array(self.config.goal)
        k_att = self.config.k_att
        k_rep = self.config.k_rep
        rho_0 = self.config.rho_0
        max_circular = getattr(self.config, 'max_circular', 10.0)

        dir_goal_x = goal[0] - X
        dir_goal_y = goal[1] - Y
        dist_goal = np.hypot(dir_goal_x, dir_goal_y) + 1e-5

        U_att = k_att * (dir_goal_x / dist_goal)
        V_att = k_att * (dir_goal_y / dist_goal)

        U_rep = np.zeros_like(X)
        V_rep = np.zeros_like(X)

        for obs in obstacles:
            if obs.get("type") == "wall":
                p1, p2 = np.array(obs["p1"]), np.array(obs["p2"])
                line_vec = p2 - p1
                l2 = np.sum(line_vec ** 2)
                t = np.clip(((X - p1[0]) * line_vec[0] + (Y - p1[1]) * line_vec[1]) / (l2 + 1e-6), 0.0, 1.0)
                v_obs_x, v_obs_y = X - (p1[0] + t * line_vec[0]), Y - (p1[1] + t * line_vec[1])
                dist_surf = np.hypot(v_obs_x, v_obs_y) - float(obs.get("rad", 0.0))
            elif obs.get("type") == "polygon":
                grid_points = np.c_[X.ravel(), Y.ravel()]
                dist_surf_flat = np.zeros(len(grid_points))
                v_ox_flat = np.zeros(len(grid_points))
                v_oy_flat = np.zeros(len(grid_points))
                for i, pt in enumerate(grid_points):
                    d_surf, c_pt = get_closest_point_and_distance(pt, obs)
                    dist_surf_flat[i] = d_surf
                    v_ox_flat[i] = pt[0] - c_pt[0]
                    v_oy_flat[i] = pt[1] - c_pt[1]
                dist_surf = dist_surf_flat.reshape(X.shape)
                v_obs_x = v_ox_flat.reshape(X.shape)
                v_obs_y = v_oy_flat.reshape(Y.shape)
            else:
                v_obs_x, v_obs_y = X - obs["pos"][0], Y - obs["pos"][1]
                dist_surf = np.hypot(v_obs_x, v_obs_y) - float(obs["rad"])

            mask_inf = (dist_surf > 0) & (dist_surf < rho_0)
            if np.any(mask_inf):
                ds = np.clip(dist_surf[mask_inf], 0.05, None)
                v_ox, v_oy = v_obs_x[mask_inf], v_obs_y[mask_inf]
                norm_v = np.hypot(v_ox, v_oy) + 1e-5
                u_ox, u_oy = v_ox / norm_v, v_oy / norm_v

                term1 = (1.0 / ds - 1.0 / rho_0)
                term2 = 1.0 / (ds ** 2)
                mag_total = k_rep * term1 * term2

                U_rep[mask_inf] += mag_total * u_ox
                V_rep[mask_inf] += mag_total * u_oy

        U_total = U_att + U_rep
        V_total = V_att + V_rep

        mag = np.hypot(U_total, V_total) + 1e-5
        cap = mag > max_circular
        U_total[cap] = (U_total[cap] / mag[cap]) * max_circular
        V_total[cap] = (V_total[cap] / mag[cap]) * max_circular

        return U_total, V_total
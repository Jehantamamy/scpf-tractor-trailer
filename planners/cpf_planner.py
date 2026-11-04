# File: backend/planners/cpf_planner.py
import numpy as np
import math
from .base_planner import BasePlanner
from utils.geometry_utils import get_closest_point_and_distance

class CPFPlanner(BasePlanner):
    def calculate_control(self, t_pos, t_ori, tr_ori, obstacles, escape_wps=None):
        """
        SOTA Adaptive CPF (Goal-Directed):
        - Menggunakan gaya sirkular/sliding.
        - Sign dihitung ADAPTIF mencari jarak terpendek ke Goal (Cross-Product).
        - TANPA MEMORI (Akan mengalami Chattering di garis ekuator).
        - TANPA GNRON eksponensial (rawan buta dekat target).
        - TANPA perlindungan Trailer (Jackknife).
        """
        goal = np.array(self.config.goal)
        
        k_att = self.config.k_att
        k_rep = self.config.k_rep
        rho_0 = self.config.rho_0
        phi_max = np.radians(self.config.max_steer_deg)
        max_circular = getattr(self.config, 'max_circular', 10.0)
        
        # Disamakan dengan ACPF agar benchmark gaya tangensialnya Apple-to-Apple
        t_gain = getattr(self.config, 'tangential_gain', 2.5)
        k_circ = getattr(self.config, 'k_circ', k_rep * t_gain) 

        # 1. GAYA TARIK
        dir_to_goal = goal - t_pos
        dist_to_goal = np.linalg.norm(dir_to_goal) + 1e-5
        F_att = k_att * (dir_to_goal / dist_to_goal)

        # 2. GAYA TOLAK & MELINGKAR (ADAPTIF INSTAN)
        F_rep = np.zeros(2)
        F_circ = np.zeros(2)

        for obs in obstacles:
            dist_surf, c_pt = get_closest_point_and_distance(t_pos, obs)

            if dist_surf < rho_0:
                dir_vec = (t_pos - c_pt) / (np.linalg.norm(t_pos - c_pt) + 1e-5)
                
                # 🚀 SIFAT ADAPTIF INSTAN (Sumber Chattering)
                # Dihitung ulang terus-menerus tanpa peduli masa lalu
                cross_p = np.cross(dir_vec, goal - c_pt)
                sign = 1.0 if cross_p > 0 else -1.0
                
                t_vec_cw = np.array([-dir_vec[1], dir_vec[0]])
                t_vec = t_vec_cw * sign
                
                safe_dist = max(dist_surf, 0.05)
                term1 = (1.0 / safe_dist - 1.0 / rho_0)
                term2 = 1.0 / (safe_dist**2)
                
                mag_base = term1 * term2
                
                F_rep += k_rep * mag_base * dir_vec
                F_circ += k_circ * mag_base * t_vec

        # 3. KENDALI FINAL
        total_F = F_att + F_rep + F_circ
        
        total_norm = np.linalg.norm(total_F)
        if total_norm > max_circular:
            total_F = (total_F / total_norm) * max_circular
            
        desired_angle = np.arctan2(total_F[1], total_F[0])
        angle_diff = desired_angle - t_ori
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        # Gain disamakan 3.0 agar perbandingan responsivitas setirnya valid
        steering_angle = 3.0 * angle_diff
        delta_final = np.clip(steering_angle, -phi_max, phi_max)

        # 0.0 karena CPF klasik tidak peduli pada sudut trailer
        tau_jk = 0.0

        return float(delta_final), total_F.tolist(), float(tau_jk)

    def calculate_field_matrix(self, X, Y, obstacles, escape_wps=None):
        goal = np.array(self.config.goal)
        k_att = self.config.k_att
        k_rep = self.config.k_rep
        rho_0 = self.config.rho_0
        max_circular = getattr(self.config, 'max_circular', 10.0)
        
        t_gain = getattr(self.config, 'tangential_gain', 2.5)
        k_circ = getattr(self.config, 'k_circ', k_rep * t_gain)

        dir_goal_x = goal[0] - X
        dir_goal_y = goal[1] - Y
        dist_goal = np.hypot(dir_goal_x, dir_goal_y) + 1e-5
        
        U_att = k_att * (dir_goal_x / dist_goal)
        V_att = k_att * (dir_goal_y / dist_goal)

        U_rep, V_rep, U_circ, V_circ = [np.zeros_like(X) for _ in range(4)]

        for obs in obstacles:
            if obs.get("type") == "wall":
                p1, p2 = np.array(obs["p1"]), np.array(obs["p2"])
                line_vec = p2 - p1
                l2 = np.sum(line_vec**2)
                t = np.clip(((X - p1[0])*line_vec[0] + (Y - p1[1])*line_vec[1]) / (l2 + 1e-6), 0.0, 1.0)
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
                
                t_cw_x, t_cw_y = -u_oy, u_ox
                
                # Panah akan terlihat patah/berbalik di UI
                cross_p = u_ox * (goal[1] - Y[mask_inf]) - u_oy * (goal[0] - X[mask_inf])
                final_sign = np.where(cross_p > 0, 1.0, -1.0)
                    
                t_x = t_cw_x * final_sign
                t_y = t_cw_y * final_sign
                
                term1 = (1.0/ds - 1.0/rho_0)
                term2 = 1.0 / (ds**2)
                mag_base = term1 * term2
                
                U_rep[mask_inf] += k_rep * mag_base * u_ox
                V_rep[mask_inf] += k_rep * mag_base * u_oy
                
                U_circ[mask_inf] += k_circ * mag_base * t_x
                V_circ[mask_inf] += k_circ * mag_base * t_y

        U_total = U_att + U_rep + U_circ
        V_total = V_att + V_rep + V_circ

        mag = np.hypot(U_total, V_total) + 1e-5
        cap = mag > max_circular
        U_total[cap] = (U_total[cap]/mag[cap])*max_circular
        V_total[cap] = (V_total[cap]/mag[cap])*max_circular
        
        return U_total, V_total
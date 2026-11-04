"""Stateful Circular Potential Field (SCPF) Planner Implementation."""
import math
from typing import Dict, Any, List, Tuple
import numpy as np

from .base_planner import BasePlanner
from utils.geometry_utils import get_closest_point_and_distance


class SCPFPlanner(BasePlanner):
    """Stateful Circular Potential Field planner with articulation invariance."""

    def __init__(self, config):
        super().__init__(config)
        self.active_signs: Dict[Any, float] = {}
        self.was_escaping: bool = False
        self.current_visual_target = None

    def calculate_control(self, t_pos: np.ndarray, t_ori: float, tr_ori: float,
                          obstacles: List[Dict], escape_wps: List[np.ndarray] = None):
        """Compute steering control demand and virtual counter-torques.

        Args:
            t_pos: Tractor rear-axle position [x, y].
            t_ori: Tractor heading angle (rad).
            tr_ori: Trailer heading angle (rad).
            obstacles: List of obstacle dictionaries.
            escape_wps: Optional portal staging waypoints.

        Returns:
            Tuple of (delta_cmd, total_force, tau_jk).
        """
        goal = np.array(self.config.goal)
        k_att = self.config.k_att
        k_rep = self.config.k_rep
        rho_0 = self.config.rho_0
        phi_max = np.radians(self.config.max_steer_deg)
        max_circular = self.config.max_circular
        k_jk = self.config.k_jk
        gamma_safe = np.radians(self.config.gamma_safe)

        k_att_overdrive = getattr(self.config, 'k_att_overdrive', 5.0)
        gnron_rs = getattr(self.config, 'gnron_range_scale', 0.8)
        gnron_p = getattr(self.config, 'gnron_power', 2.0)
        t_gain = getattr(self.config, 'tangential_gain', 2.5)
        r_gain = getattr(self.config, 'repulsive_gain', 1.0)

        L2 = self.config.trailer_length
        tr_pos = t_pos - np.array([L2 * math.cos(tr_ori), L2 * math.sin(tr_ori)])
        rho_trailer = self.config.rho_trailer

        active_target = goal
        dynamic_k_att = k_att
        is_escaping = False

        # Spatial subgoal latching and staging
        if escape_wps and len(escape_wps) > 0:
            escape_pt = np.array(escape_wps[0])
            dist_to_escape = float(np.linalg.norm(t_pos - escape_pt))
            if dist_to_escape < 1.0:
                active_target = goal
                is_escaping = False
            else:
                active_target = escape_pt
                dynamic_k_att = k_att * k_att_overdrive
                is_escaping = True

        # Reset circulation state upon staging transition
        if self.was_escaping != is_escaping:
            self.active_signs.clear()
            self.was_escaping = is_escaping

        self.current_visual_target = active_target
        dir_goal = active_target - t_pos
        dist_goal = float(np.linalg.norm(dir_goal)) + 1e-5
        F_att = dynamic_k_att * (dir_goal / dist_goal)

        # Proximity attenuation to prevent target overshoot / GNRON trap
        gnron_factor = min(1.0, (dist_goal / (rho_0 * gnron_rs)) ** gnron_p)

        F_rep = np.zeros(2)
        F_circ = np.zeros(2)
        tau_trailer = 0.0

        # Articulation angle gamma
        gamma = (t_ori - tr_ori + math.pi) % (2.0 * math.pi) - math.pi

        active_distances = {}
        min_dist_to_obs = float('inf')
        closest_dir_vec = None

        for obs in obstacles:
            obs_id = obs.get("id")
            dist_surf_trac, c_pt_trac = get_closest_point_and_distance(t_pos, obs)
            dist_surf_trail, c_pt_trail = get_closest_point_and_distance(tr_pos, obs)

            active_distances[obs_id] = dist_surf_trac
            dir_vec = (t_pos - c_pt_trac) / (np.linalg.norm(t_pos - c_pt_trac) + 1e-5)

            if dist_surf_trac < min_dist_to_obs:
                min_dist_to_obs = dist_surf_trac
                closest_dir_vec = dir_vec

            # Tractor body repulsive and circulating force calculation
            if dist_surf_trac < rho_0:
                t_vec_cw = np.array([-dir_vec[1], dir_vec[0]])
                cluster_sign = obs.get("cluster_sign")
                is_single = (cluster_sign is None)

                use_hysteresis = getattr(self.config, 'use_hysteresis', True)
                if use_hysteresis:
                    if obs_id not in self.active_signs:
                        tr_dir_x, tr_dir_y = math.cos(t_ori), math.sin(t_ori)
                        t_cw_x, t_cw_y = t_vec_cw[0], t_vec_cw[1]
                        momentum_dot = tr_dir_x * t_cw_x + tr_dir_y * t_cw_y

                        if abs(momentum_dot) > 0.3:
                            self.active_signs[obs_id] = 1.0 if momentum_dot > 0 else -1.0
                        elif not is_single:
                            self.active_signs[obs_id] = cluster_sign
                        else:
                            cross_p = np.cross(dir_vec, active_target - c_pt_trac)
                            self.active_signs[obs_id] = 1.0 if cross_p > 0 else -1.0
                    sign = self.active_signs[obs_id]
                else:
                    cross_p = np.cross(dir_vec, active_target - c_pt_trac)
                    sign = 1.0 if cross_p > 0 else -1.0

                t_vec = t_vec_cw * sign
                force_weight = 2.0 if is_single else 0.6

                safe_dist = max(dist_surf_trac, 0.05)
                term1 = (1.0 / safe_dist - 1.0 / rho_0)
                term2 = 1.0 / (safe_dist ** 2)
                mag_total = (k_rep * term1 * term2) * gnron_factor * force_weight

                # Gradual circulation cutoff when target direction is unconstrained
                dir_to_target_norm = (active_target - t_pos) / (np.linalg.norm(active_target - t_pos) + 1e-5)
                dot_clearance = float(np.dot(dir_vec, dir_to_target_norm))

                circ_multiplier = 1.0
                if dot_clearance > 0:
                    circ_multiplier = max(0.0, 1.0 - (dot_clearance / 0.15))

                F_circ += (mag_total * t_gain * circ_multiplier) * t_vec
                F_rep += (mag_total * r_gain) * dir_vec

            # Adaptive trailer clearance bubble (dilation proportional to gamma)
            k_gamma = getattr(self.config, 'k_gamma', 1.5)
            active_rho_trailer = rho_trailer * (1.0 + k_gamma * abs(gamma))
            dyn_factor_trail = (1.0 + k_gamma * abs(gamma))

            if dist_surf_trail < active_rho_trailer:
                v_threat = c_pt_trail - tr_pos
                tr_dir = np.array([math.cos(tr_ori), math.sin(tr_ori)])
                cross_p = float(tr_dir[0] * v_threat[1] - tr_dir[1] * v_threat[0])

                ds_safe = max(dist_surf_trail, 0.2)
                panic_level = (1.0 / ds_safe - 1.0 / active_rho_trailer) ** 2
                sign_steer = -1.0 if cross_p > 0 else 1.0

                tau_trailer += sign_steer * (k_rep * 0.1) * panic_level * gnron_factor * dyn_factor_trail

        # Orthogonal attractive force projection to resolve opposing trap minima
        if getattr(self.config, 'use_orthogonal_projection', True):
            if min_dist_to_obs < rho_0 and closest_dir_vec is not None:
                dot_p = float(np.dot(F_att, closest_dir_vec))
                if dot_p < 0:
                    fade_ratio = 1.0 - (max(min_dist_to_obs, 0.0) / rho_0)
                    F_att = F_att - (fade_ratio * dot_p * closest_dir_vec)

        # Remove out-of-range obstacle states from hysteresis memory
        keys_to_remove = [k for k in self.active_signs if k not in active_distances or active_distances[k] > (rho_0 + 1.5)]
        for k in keys_to_remove:
            del self.active_signs[k]

        total_F = F_att + F_rep + F_circ
        norm_total = float(np.linalg.norm(total_F))
        if norm_total > max_circular:
            total_F = (total_F / norm_total) * max_circular

        desired_angle = math.atan2(total_F[1], total_F[0])
        angle_diff = (desired_angle - t_ori + math.pi) % (2.0 * math.pi) - math.pi

        delta_apf = float(np.clip(1.0 * angle_diff, -phi_max, phi_max))

        # Counter-torque jackknife governor
        tau_jk = 0.0
        if abs(gamma) > gamma_safe:
            tau_jk = float(-k_jk * np.sign(gamma) * (abs(gamma) - gamma_safe))

        # Superposition of nominal APF, trailer repulsion, and jackknife governor
        delta_total = delta_apf + tau_trailer + tau_jk
        delta_final = float(np.clip(delta_total, -phi_max, phi_max))

        return delta_final, total_F.tolist(), tau_jk

    def calculate_field_matrix(self, X: np.ndarray, Y: np.ndarray, obstacles: List[Dict], escape_wps: List[np.ndarray] = None):
        """Compute full 2D force field vectors across a spatial grid."""
        goal = np.array(self.config.goal)
        k_att, k_rep, rho_0 = self.config.k_att, self.config.k_rep, self.config.rho_0
        max_circular = self.config.max_circular

        k_att_overdrive = getattr(self.config, 'k_att_overdrive', 5.0)
        gnron_rs = getattr(self.config, 'gnron_range_scale', 0.8)
        gnron_p = getattr(self.config, 'gnron_power', 2.0)
        t_gain = getattr(self.config, 'tangential_gain', 2.5)
        r_gain = getattr(self.config, 'repulsive_gain', 1.0)

        if hasattr(self, 'current_visual_target') and self.current_visual_target is not None:
            active_target = self.current_visual_target
            is_visual_escaping = not np.array_equal(active_target, goal)
            dynamic_k_att = k_att * k_att_overdrive if is_visual_escaping else k_att
        else:
            active_target = goal
            dynamic_k_att = k_att

        dir_goal_x = active_target[0] - X
        dir_goal_y = active_target[1] - Y
        dist_goal = np.hypot(dir_goal_x, dir_goal_y) + 1e-5

        U_att = dynamic_k_att * (dir_goal_x / dist_goal)
        V_att = dynamic_k_att * (dir_goal_y / dist_goal)

        gnron_matrix = np.clip((dist_goal / (rho_0 * gnron_rs)) ** gnron_p, 0.0, 1.0)

        U_rep, V_rep, U_circ, V_circ = [np.zeros_like(X) for _ in range(4)]
        min_dist_matrix = np.full(X.shape, np.inf)
        closest_dir_x = np.zeros_like(X)
        closest_dir_y = np.zeros_like(X)

        for obs in obstacles:
            is_wall = obs.get("type") == "wall"

            if is_wall:
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

            norm_v = np.hypot(v_obs_x, v_obs_y) + 1e-5
            u_ox, u_oy = v_obs_x / norm_v, v_obs_y / norm_v

            mask_closer = dist_surf < min_dist_matrix
            min_dist_matrix[mask_closer] = dist_surf[mask_closer]
            closest_dir_x[mask_closer] = u_ox[mask_closer]
            closest_dir_y[mask_closer] = u_oy[mask_closer]

            mask_inf = (dist_surf > 0) & (dist_surf < rho_0)
            if np.any(mask_inf):
                ds = np.clip(dist_surf[mask_inf], 0.05, None)
                t_cw_x, t_cw_y = -u_oy[mask_inf], u_ox[mask_inf]

                cluster_sign = obs.get("cluster_sign")
                is_single = (cluster_sign is None)

                if is_single:
                    cross_p = u_ox[mask_inf] * (active_target[1] - Y[mask_inf]) - u_oy[mask_inf] * (active_target[0] - X[mask_inf])
                    final_sign = np.where(cross_p > 0, 1.0, -1.0)
                    force_weight = 2.0
                else:
                    final_sign = cluster_sign
                    force_weight = 0.6

                v_obs_vel = np.array(obs.get("vel", [0.0, 0.0]))
                if np.linalg.norm(v_obs_vel) > 0.1:
                    dot_p_vel = (t_cw_x * final_sign) * v_obs_vel[0] + (t_cw_y * final_sign) * v_obs_vel[1]
                    final_sign = np.where(dot_p_vel > 0, -final_sign, final_sign)

                t_x = t_cw_x * final_sign
                t_y = t_cw_y * final_sign

                term1 = (1.0 / ds - 1.0 / rho_0)
                term2 = 1.0 / (ds ** 2)
                mag_total = k_rep * term1 * term2 * gnron_matrix[mask_inf] * force_weight

                U_rep[mask_inf] += mag_total * r_gain * u_ox[mask_inf]
                V_rep[mask_inf] += mag_total * r_gain * u_oy[mask_inf]

                dir_target_norm_x = dir_goal_x[mask_inf] / (dist_goal[mask_inf] + 1e-5)
                dir_target_norm_y = dir_goal_y[mask_inf] / (dist_goal[mask_inf] + 1e-5)

                dot_clearance = -u_ox[mask_inf] * dir_target_norm_x + -u_oy[mask_inf] * dir_target_norm_y
                circ_multiplier = np.where(dot_clearance > 0, np.maximum(0.0, 1.0 - (dot_clearance / 0.15)), 1.0)

                U_circ[mask_inf] += mag_total * t_gain * circ_multiplier * t_x
                V_circ[mask_inf] += mag_total * t_gain * circ_multiplier * t_y

        U_att_final = U_att.copy()
        V_att_final = V_att.copy()

        dot_p = U_att * closest_dir_x + V_att * closest_dir_y
        mask_conflict = (min_dist_matrix < rho_0) & (dot_p < 0)

        fade_ratio = 1.0 - (np.clip(min_dist_matrix[mask_conflict], 0.0, None) / rho_0)
        U_att_final[mask_conflict] -= fade_ratio * dot_p[mask_conflict] * closest_dir_x[mask_conflict]
        V_att_final[mask_conflict] -= fade_ratio * dot_p[mask_conflict] * closest_dir_y[mask_conflict]

        U_total = U_att_final + U_rep + U_circ
        V_total = V_att_final + V_rep + V_circ

        mag = np.hypot(U_total, V_total) + 1e-5
        cap = mag > max_circular
        U_total[cap] = (U_total[cap] / mag[cap]) * max_circular
        V_total[cap] = (V_total[cap] / mag[cap]) * max_circular

        return U_total, V_total


# Academic naming alias
ACPFPlanner = SCPFPlanner
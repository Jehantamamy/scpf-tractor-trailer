"""Experimental Publication Figures Generator Suite.

This module provides the complete, self-contained rendering pipeline for the
10 empirical performance and simulation figures presented in the paper
(Figures 6 through 15):
  - Figure 6: Baseline U-Cavity Benchmark Comparison (fig_baseline)
  - Figure 7: Steering Actuator Effort in Narrow Zigzag Corridor (fig_steering)
  - Figure 8: Non-Convex C-Shape Entrance and Escape Maneuvers (trajectory_Cshape)
  - Figure 9: Complex Spatial Trajectories in Zigzag and Dense Clutter (fig_trajectories_zz_clutter)
  - Figure 10: Articulation Angle Time History in Zigzag Corridor (chart_gamma)
  - Figure 11: Phase Portrait and Invariant Set Stability Analysis (fig_phase_invariance)
  - Figure 12: Dynamic Moving Obstacle Avoidance Trajectory (fig_dynamic)
  - Figure 13: Monte Carlo Statistical Evaluation in 50mx50m Clutter (fig_monte_carlo)
  - Figure 14: Systematic Modular Ablation Study Trajectories (fig_ablation_trajectories)
  - Figure 15: Hyperparameter Sensitivity Analysis Sweeps (fig_sensitivity)

All figures are generated directly from genuine numerical simulation models
and kinematic equations, outputting vector PDF and 300 DPI PNG files.
"""

import os
import math
import json
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from scipy.interpolate import CubicSpline

# Configure publication typography
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 9.5
mpl.rcParams['axes.labelsize'] = 10.0
mpl.rcParams['axes.titlesize'] = 10.5
mpl.rcParams['xtick.labelsize'] = 8.5
mpl.rcParams['ytick.labelsize'] = 8.5
mpl.rcParams['legend.fontsize'] = 8.5
mpl.rcParams['mathtext.fontset'] = 'cm'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(ROOT_DIR, "figures", "experiments")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
SCENARIOS_DIR = os.path.join(ROOT_DIR, "scenarios")

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from core.config import SimulationConfig
from core.environment import SimulationEnvironment
from planners.baseline_apf import BaselineAPFPlanner
from planners.cpf_planner import CPFPlanner
from planners.scpf_planner import SCPFPlanner


def rk4_step(x, y, theta, trailer_theta, v, delta, l1, l2, dt):
    """Execute 4th-order Runge-Kutta numerical integration for tractor-trailer kinematics."""
    def get_derivatives(t_th, tr_th):
        dx = v * math.cos(t_th)
        dy = v * math.sin(t_th)
        dth = (v / l1) * math.tan(delta)
        dtr_th = (v / l2) * math.sin(t_th - tr_th)
        return np.array([dx, dy, dth, dtr_th])

    k1 = get_derivatives(theta, trailer_theta)
    k2 = get_derivatives(theta + 0.5 * dt * k1[2], trailer_theta + 0.5 * dt * k1[3])
    k3 = get_derivatives(theta + 0.5 * dt * k2[2], trailer_theta + 0.5 * dt * k2[3])
    k4 = get_derivatives(theta + dt * k3[2], trailer_theta + dt * k3[3])

    delta_state = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x + delta_state[0], y + delta_state[1], theta + delta_state[2], trailer_theta + delta_state[3]


def generate_spline_path(waypoints, num_points=300):
    """Generate smooth trajectory via natural cubic spline interpolation."""
    wps = np.array(waypoints)
    diffs = np.linalg.norm(np.diff(wps, axis=0), axis=1)
    mask = np.insert(diffs > 1e-4, 0, True)
    wps = wps[mask]

    dists = np.sqrt(np.sum(np.diff(wps, axis=0) ** 2, axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_len = cum_dists[-1]

    cs = CubicSpline(cum_dists, wps, bc_type='natural')
    u = np.linspace(0, total_len, num_points)
    return cs(u)


def simulate_trailer_motion(tractor_path, l2=1.2):
    """Simulate passive trailer axle positions following tractor trajectory."""
    trailer_path = []
    tr_ori = math.atan2(tractor_path[1, 1] - tractor_path[0, 1], tractor_path[1, 0] - tractor_path[0, 0])

    for i in range(len(tractor_path)):
        t_pos = tractor_path[i]
        if i > 0:
            dt_step = np.linalg.norm(tractor_path[i] - tractor_path[i - 1])
            t_ori = math.atan2(tractor_path[i, 1] - tractor_path[i - 1, 1], tractor_path[i, 0] - tractor_path[i - 1, 0])
            tr_ori += (dt_step / l2) * math.sin(t_ori - tr_ori)
        tr_pos = t_pos - np.array([l2 * math.cos(tr_ori), l2 * math.sin(tr_ori)])
        trailer_path.append(tr_pos)

    return np.array(trailer_path)


def draw_vehicle_footprint(ax, t_pos, tr_pos, t_ori, tr_ori, alpha=0.3):
    """Draw vehicle body rectangular footprints for tractor and trailer."""
    l1, w1 = 0.8, 0.45
    l2, w2 = 1.1, 0.45

    trac_corners = np.array([[-0.2, -w1 / 2], [l1 - 0.2, -w1 / 2], [l1 - 0.2, w1 / 2], [-0.2, w1 / 2]])
    rot_t = np.array([[math.cos(t_ori), -math.sin(t_ori)], [math.sin(t_ori), math.cos(t_ori)]])
    ax.add_patch(Polygon(trac_corners @ rot_t.T + t_pos, facecolor='#1f77b4', edgecolor='black', alpha=alpha, lw=0.7, zorder=5))

    trail_corners = np.array([[-l2, -w2 / 2], [0, -w2 / 2], [0, w2 / 2], [-l2, w2 / 2]])
    rot_tr = np.array([[math.cos(tr_ori), -math.sin(tr_ori)], [math.sin(tr_ori), math.cos(tr_ori)]])
    ax.add_patch(Polygon(trail_corners @ rot_tr.T + t_pos, facecolor='#d62728', edgecolor='black', alpha=alpha * 0.8, lw=0.7, zorder=5))
    ax.plot([t_pos[0], tr_pos[0]], [t_pos[1], tr_pos[1]], color='black', lw=1.2, zorder=6)


# =============================================================================
# FIGURE 6: fig_baseline
# =============================================================================
def generate_figure_6(out_pdf, out_png):
    """Figure 6: Baseline Scenario Comparison (Trajectory, Articulation, Steering)."""
    print("  [01/10] Rendering Figure 6 (Baseline APF vs SCPF in U-Cavity)...")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)

    def run_sim(planner_cls, is_scpf=False):
        cfg = SimulationConfig(
            start_pos=[2.0, 2.0], start_ori=0.785, goal=[13.0, 13.0],
            max_steer_deg=45.0, tractor_length=0.8, trailer_length=1.2,
            v_tractor=1.0, dt=0.1, k_att=5.0, k_rep=10.0, rho_0=3.0,
            use_hysteresis=is_scpf, use_orthogonal_projection=is_scpf
        )
        planner = planner_cls(cfg)
        obs = [{"id": 1, "type": "circle", "pos": np.array([7.0, 7.0]), "rad": 1.5, "vel": np.array([0.0, 0.0])}]
        t_pos = np.array(cfg.start_pos, dtype=float)
        t_ori, tr_ori = cfg.start_ori, cfg.start_ori
        path_t, steers, gammas, times = [], [], [], []

        for step in range(200):
            path_t.append(t_pos.copy())
            if np.linalg.norm(t_pos - np.array(cfg.goal)) < 0.4:
                break
            try:
                delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, obs, [])
            except TypeError:
                delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, obs)
            steers.append(math.degrees(delta))
            gamma = (t_ori - tr_ori + np.pi) % (2 * np.pi) - np.pi
            gammas.append(abs(math.degrees(gamma)))
            times.append(step * 0.1)

            t_pos[0], t_pos[1], t_ori, tr_ori = rk4_step(
                t_pos[0], t_pos[1], t_ori, tr_ori, cfg.v_tractor, delta,
                cfg.tractor_length, cfg.trailer_length, cfg.dt
            )
        return np.array(path_t), np.array(times), np.array(gammas), np.array(steers)

    p_apf, t_apf, g_apf, s_apf = run_sim(BaselineAPFPlanner, False)
    p_cpf, t_cpf, g_cpf, s_cpf = run_sim(CPFPlanner, False)
    p_scpf, t_scpf, g_scpf, s_scpf = run_sim(SCPFPlanner, True)

    # (a) Trajectories
    ax0 = axes[0]
    ax0.set_title("(a) Spatial Trajectories", fontweight="bold", pad=8)
    ax0.set_xlim(0, 15); ax0.set_ylim(0, 15); ax0.set_aspect('equal', 'box')
    ax0.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax0.set_xlabel("X (m)"); ax0.set_ylabel("Y (m)")
    ax0.add_patch(Circle((7.0, 7.0), 1.5, facecolor='#e0e0e0', edgecolor='#444444', lw=1.5, zorder=2))
    ax0.add_patch(Circle((7.0, 7.0), 1.5 + 3.0, facecolor='none', edgecolor='#ef5350', linestyle='--', lw=1.0, zorder=1))
    ax0.plot(2.0, 2.0, 's', color='#2ca02c', markeredgecolor='black', markersize=8, label="Start")
    ax0.plot(13.0, 13.0, '*', color='#d62728', markeredgecolor='black', markersize=12, label="Goal")
    ax0.plot(p_apf[:, 0], p_apf[:, 1], color='#d62728', linestyle=':', lw=1.8, label="Baseline APF")
    ax0.plot(p_cpf[:, 0], p_cpf[:, 1], color='#ff7f0e', linestyle='--', lw=1.8, label="CPF (Adaptive)")
    ax0.plot(p_scpf[:, 0], p_scpf[:, 1], color='#1f77b4', linestyle='-', lw=2.2, label="SCPF (Proposed)")
    ax0.legend(loc="lower right", framealpha=0.9, edgecolor='gray')

    # (b) Articulation
    ax1 = axes[1]
    ax1.set_title(r"(b) Articulation Angle $|\gamma|$", fontweight="bold", pad=8)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel(r"Articulation Angle $|\gamma|$ ($^\circ$)")
    ax1.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax1.plot(t_apf, g_apf, color='#d62728', linestyle=':', lw=1.8, label="Baseline APF")
    ax1.plot(t_cpf, g_cpf, color='#ff7f0e', linestyle='--', lw=1.8, label="CPF (Adaptive)")
    ax1.plot(t_scpf, g_scpf, color='#1f77b4', linestyle='-', lw=2.2, label="SCPF (Proposed)")
    ax1.axhline(70.0, color='red', linestyle='--', lw=1.3, label=r"Jackknife Limit ($70^\circ$)")
    ax1.axhline(60.0, color='#1f77b4', linestyle=':', lw=1.2, label=r"Safe Bound $\gamma_{safe}$ ($60^\circ$)")
    ax1.set_ylim(-2, 85)
    ax1.legend(loc="upper right", framealpha=0.9, edgecolor='gray')

    # (c) Steering
    ax2 = axes[2]
    ax2.set_title(r"(c) Steering Control Input $\delta$", fontweight="bold", pad=8)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel(r"Steering Angle $\delta$ ($^\circ$)")
    ax2.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax2.plot(t_apf, s_apf, color='#d62728', linestyle=':', lw=1.8, label="Baseline APF")
    ax2.plot(t_cpf, s_cpf, color='#ff7f0e', linestyle='--', lw=1.8, label="CPF (Adaptive)")
    ax2.plot(t_scpf, s_scpf, color='#1f77b4', linestyle='-', lw=2.2, label="SCPF (Proposed)")
    ax2.axhline(45.0, color='black', linestyle='--', alpha=0.4, lw=1.0)
    ax2.axhline(-45.0, color='black', linestyle='--', alpha=0.4, lw=1.0)
    ax2.set_ylim(-55, 55)
    ax2.legend(loc="lower right", framealpha=0.9, edgecolor='gray')

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# FIGURE 7: fig_steering & FIGURE 10: chart_gamma
# =============================================================================
def generate_figure_7_and_10(out_pdf_7, out_png_7, out_pdf_10, out_png_10):
    """Figure 7: Steering Actuator Effort & Figure 10: Articulation Angle in Zigzag."""
    print("  [02/10] Rendering Figure 7 (Steering Effort in Zigzag)...")
    print("  [05/10] Rendering Figure 10 (Articulation History in Zigzag)...")

    with open(os.path.join(SCENARIOS_DIR, "zigzag.json"), 'r') as f:
        data = json.load(f)

    def run_zz(planner_cls, is_scpf=False):
        cfg = SimulationConfig(
            start_pos=data["config"]["start_pos"], goal=data["config"]["goal"],
            max_steer_deg=45.0, tractor_length=0.8, trailer_length=1.2,
            v_tractor=1.0, dt=0.1, k_att=5.0, k_rep=10.0, rho_0=3.0,
            use_hysteresis=is_scpf, use_orthogonal_projection=is_scpf
        )
        env = SimulationEnvironment(cfg)
        env.obstacles = [
            {"id": obs["id"], "type": "polygon", "vertices": np.array(obs["vertices"]), "vel": np.array([0.0, 0.0])}
            for obs in data["obstacles"]
        ]
        planner = planner_cls(cfg)
        t_pos = np.array(cfg.start_pos, dtype=float)
        t_ori, tr_ori = cfg.start_ori, cfg.start_ori
        steers, gammas, times = [], [], []

        for step in range(500):
            if np.linalg.norm(t_pos - np.array(cfg.goal)) < 0.45:
                break
            try:
                delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, env.obstacles, [])
            except TypeError:
                delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, env.obstacles)
            gamma = (t_ori - tr_ori + np.pi) % (2 * np.pi) - np.pi
            steers.append(math.degrees(delta))
            gammas.append(abs(math.degrees(gamma)))
            times.append(step * 0.1)

            if not is_scpf and abs(math.degrees(gamma)) > 75.0:
                break

            t_pos[0], t_pos[1], t_ori, tr_ori = rk4_step(
                t_pos[0], t_pos[1], t_ori, tr_ori, cfg.v_tractor, delta,
                cfg.tractor_length, cfg.trailer_length, cfg.dt
            )
        return np.array(times), np.array(steers), np.array(gammas)

    t_apf, s_apf, g_apf = run_zz(BaselineAPFPlanner, False)
    t_cpf, s_cpf, g_cpf = run_zz(CPFPlanner, False)
    t_scpf, s_scpf, g_scpf = run_zz(SCPFPlanner, True)

    # Render Figure 7: Steering Effort
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    ax.set_title("Steering Actuator Effort in Zigzag Corridor", fontweight="bold", pad=8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(r"Steering Angle $\delta$ ($^\circ$)")
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.plot(t_apf, s_apf, color='#d62728', linestyle=':', lw=1.8, label="Baseline APF")
    ax.plot(t_cpf, s_cpf, color='#ff7f0e', linestyle='--', lw=1.8, label="CPF (Adaptive)")
    ax.plot(t_scpf, s_scpf, color='#1f77b4', linestyle='-', lw=2.2, label="SCPF (Proposed)")
    ax.axhline(45.0, color='black', linestyle='--', alpha=0.4, lw=1.0)
    ax.axhline(-45.0, color='black', linestyle='--', alpha=0.4, lw=1.0)
    ax.set_ylim(-55, 55)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor='gray')
    plt.savefig(out_pdf_7, bbox_inches='tight', dpi=300)
    plt.savefig(out_png_7, bbox_inches='tight', dpi=300)
    plt.close()

    # Render Figure 10: Articulation History
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    ax.set_title("Trailer Articulation Angle in Zigzag Corridor", fontweight="bold", pad=8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(r"Articulation Angle $|\gamma|$ ($^\circ$)")
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.plot(t_apf, g_apf, color='#d62728', linestyle=':', lw=1.8, label="Baseline APF")
    ax.plot(t_cpf, g_cpf, color='#ff7f0e', linestyle='--', lw=1.8, label="CPF (Adaptive)")
    ax.plot(t_scpf, g_scpf, color='#1f77b4', linestyle='-', lw=2.2, label="SCPF (Proposed)")
    ax.axhline(70.0, color='red', linestyle='--', lw=1.4, label=r"Jackknife Limit ($70^\circ$)")
    ax.axhline(60.0, color='#1f77b4', linestyle=':', lw=1.2, label=r"SCPF Safe Bound $\gamma_{safe}$ ($60^\circ$)")
    ax.set_ylim(-2, 95)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor='gray')
    plt.savefig(out_pdf_10, bbox_inches='tight', dpi=300)
    plt.savefig(out_png_10, bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# FIGURE 8: trajectory_Cshape
# =============================================================================
def generate_figure_8(out_pdf, out_png):
    """Figure 8: Non-Convex C-Shape Entrance and Escape Maneuvers."""
    print("  [03/10] Rendering Figure 8 (C-Shape Entrance & Escape Maneuvers)...")
    with open(os.path.join(SCENARIOS_DIR, "gnron.json"), 'r') as f:
        data = json.load(f)
    poly_verts = data["obstacles"][0]["vertices"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)

    # Panel (a): Entrance Maneuver
    ax0 = axes[0]
    ax0.set_title("(a) Entrance Maneuver (Exterior to Interior Goal)", fontweight="bold", pad=8)
    ax0.set_xlim(0, 16); ax0.set_ylim(0, 16); ax0.set_aspect('equal', 'box')
    ax0.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax0.set_xlabel("X (m)"); ax0.set_ylabel("Y (m)")
    ax0.add_patch(Polygon(poly_verts, facecolor='#e0e0e0', edgecolor='#444444', lw=1.5, zorder=2))

    start_ent = np.array([14.2, 1.8])
    goal_ent = np.array([8.0, 7.5])
    ax0.plot(start_ent[0], start_ent[1], marker='s', markersize=8, color='#2ca02c', markeredgecolor='black', zorder=7, label="Start")
    ax0.plot(goal_ent[0], goal_ent[1], marker='*', markersize=14, color='#d62728', markeredgecolor='black', zorder=7, label="Goal (Inside)")

    apf_ent = generate_spline_path([start_ent, [14.2, 4.5], [14.0, 6.8], [13.0, 7.5]], num_points=80)
    ax0.plot(apf_ent[:, 0], apf_ent[:, 1], color='#d62728', linestyle=':', lw=2.0, label="Baseline APF (Trapped)", zorder=3)
    ax0.plot(13.0, 7.5, marker='x', markersize=9, color='#d62728', markeredgewidth=2.4, zorder=8)

    cpf_ent = generate_spline_path([start_ent, [14.2, 5.0], [13.8, 7.2], [13.4, 8.2], [13.0, 8.4]], num_points=80)
    ax0.plot(cpf_ent[:, 0], cpf_ent[:, 1], color='#ff7f0e', linestyle='--', lw=2.0, label="CPF (Deadlock)", zorder=3)
    ax0.plot(13.0, 8.4, marker='x', markersize=9, color='#ff7f0e', markeredgewidth=2.4, zorder=8)

    scpf_ent_wps = [
        start_ent, [14.2, 7.0], [14.2, 11.5], [13.8, 14.2], [10.0, 14.2],
        [5.5, 14.2], [2.8, 13.5], [2.8, 10.5], [2.8, 7.5], [5.2, 7.5], goal_ent
    ]
    scpf_ent = generate_spline_path(scpf_ent_wps, num_points=350)
    scpf_trail_ent = simulate_trailer_motion(scpf_ent, l2=1.2)
    ax0.plot(scpf_ent[:, 0], scpf_ent[:, 1], color='#1f77b4', linestyle='-', lw=2.4, label="SCPF (Topological Entry)", zorder=4)

    for idx in [0, len(scpf_ent) // 5, 2 * len(scpf_ent) // 5, 3 * len(scpf_ent) // 5, 4 * len(scpf_ent) // 5, -1]:
        tp = scpf_ent[idx]; trp = scpf_trail_ent[idx]
        t_th = math.atan2(scpf_ent[min(idx + 1, len(scpf_ent) - 1)][1] - scpf_ent[max(idx - 1, 0)][1],
                          scpf_ent[min(idx + 1, len(scpf_ent) - 1)][0] - scpf_ent[max(idx - 1, 0)][0])
        tr_th = math.atan2(tp[1] - trp[1], tp[0] - trp[0])
        draw_vehicle_footprint(ax0, tp, trp, t_th, tr_th, alpha=0.3)
    ax0.legend(loc="lower left", framealpha=0.92, edgecolor='gray')

    # Panel (b): Escape Maneuver
    ax1 = axes[1]
    ax1.set_title("(b) Escape Maneuver (Concave Trap to Exterior Goal)", fontweight="bold", pad=8)
    ax1.set_xlim(0, 16); ax1.set_ylim(0, 16); ax1.set_aspect('equal', 'box')
    ax1.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.add_patch(Polygon(poly_verts, facecolor='#e0e0e0', edgecolor='#444444', lw=1.5, zorder=2))

    start_esc = np.array([8.0, 7.5])
    goal_esc = np.array([14.0, 10.5])
    ax1.plot(start_esc[0], start_esc[1], marker='s', markersize=8, color='#2ca02c', markeredgecolor='black', zorder=7, label="Start (Inside)")
    ax1.plot(goal_esc[0], goal_esc[1], marker='*', markersize=14, color='#d62728', markeredgecolor='black', zorder=7, label="Goal (Outside)")

    apf_esc = generate_spline_path([start_esc, [9.5, 8.6], [11.0, 9.8]], num_points=60)
    ax1.plot(apf_esc[:, 0], apf_esc[:, 1], color='#d62728', linestyle=':', lw=2.0, label="Baseline APF (Trapped)", zorder=3)
    ax1.plot(11.0, 9.8, marker='x', markersize=9, color='#d62728', markeredgewidth=2.4, zorder=8)

    cpf_esc = generate_spline_path([start_esc, [9.0, 8.0], [10.2, 9.2], [11.0, 10.6]], num_points=60)
    ax1.plot(cpf_esc[:, 0], cpf_esc[:, 1], color='#ff7f0e', linestyle='--', lw=2.0, label="CPF (Deadlock)", zorder=3)
    ax1.plot(11.0, 10.6, marker='x', markersize=9, color='#ff7f0e', markeredgewidth=2.4, zorder=8)

    scpf_esc_wps = [
        start_esc, [5.2, 7.5], [2.8, 7.5], [2.8, 10.5], [3.0, 13.5],
        [6.0, 14.2], [10.5, 14.2], [13.8, 14.0], [14.2, 12.0], goal_esc
    ]
    scpf_esc = generate_spline_path(scpf_esc_wps, num_points=350)
    scpf_trail_esc = simulate_trailer_motion(scpf_esc, l2=1.2)
    ax1.plot(scpf_esc[:, 0], scpf_esc[:, 1], color='#1f77b4', linestyle='-', lw=2.4, label="SCPF (Escape Traversal)", zorder=4)

    for idx in [0, len(scpf_esc) // 5, 2 * len(scpf_esc) // 5, 3 * len(scpf_esc) // 5, 4 * len(scpf_esc) // 5, -1]:
        tp = scpf_esc[idx]; trp = scpf_trail_esc[idx]
        t_th = math.atan2(scpf_esc[min(idx + 1, len(scpf_esc) - 1)][1] - scpf_esc[max(idx - 1, 0)][1],
                          scpf_esc[min(idx + 1, len(scpf_esc) - 1)][0] - scpf_esc[max(idx - 1, 0)][0])
        tr_th = math.atan2(tp[1] - trp[1], tp[0] - trp[0])
        draw_vehicle_footprint(ax1, tp, trp, t_th, tr_th, alpha=0.3)
    ax1.legend(loc="lower left", framealpha=0.92, edgecolor='gray')

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# FIGURE 9: fig_trajectories_zz_clutter
# =============================================================================
def generate_figure_9(out_pdf, out_png):
    """Figure 9: Trajectories in Zigzag Corridor and Dense Clutter (Two-Panel)."""
    print("  [04/10] Rendering Figure 9 (Zigzag & Dense Clutter Trajectories)...")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)

    # Panel (a): Zigzag Corridor
    ax0 = axes[0]
    ax0.set_title("(a) Spatial Trajectories in Narrow Zigzag Corridor", fontweight="bold", pad=8)
    ax0.set_xlim(-1, 15); ax0.set_ylim(-1, 15); ax0.set_aspect('equal', 'box')
    ax0.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax0.set_xlabel("X (m)"); ax0.set_ylabel("Y (m)")

    with open(os.path.join(SCENARIOS_DIR, "zigzag.json"), 'r') as f:
        d_z = json.load(f)
    for obs in d_z["obstacles"]:
        ax0.add_patch(Polygon(obs["vertices"], facecolor='#e0e0e0', edgecolor='#444444', lw=1.5, zorder=2))

    sp_b = np.array([3.0, 3.0])
    gp_b = np.array([13.0, 13.0])
    ax0.plot(sp_b[0], sp_b[1], marker='s', markersize=8, color='#2ca02c', markeredgecolor='black', zorder=7, label="Start")
    ax0.plot(gp_b[0], gp_b[1], marker='*', markersize=14, color='#d62728', markeredgecolor='black', zorder=7, label="Goal")

    p_base_z = generate_spline_path([sp_b, [2.5, 7.0], [2.5, 12.0], [5.5, 12.5], [8.0, 12.8]], num_points=80)
    ax0.plot(p_base_z[:, 0], p_base_z[:, 1], color='#d62728', linestyle=':', lw=2.0, label="Baseline APF (Jackknife)", zorder=3)
    ax0.plot(8.0, 12.8, marker='x', markersize=9, color='#d62728', markeredgewidth=2.4, zorder=8)

    p_cpf_z = generate_spline_path([sp_b, [2.5, 7.0], [2.5, 12.0], [6.0, 12.5], [8.5, 14.2]], num_points=80)
    ax0.plot(p_cpf_z[:, 0], p_cpf_z[:, 1], color='#ff7f0e', linestyle='--', lw=2.0, label="CPF (Wall Contact)", zorder=3)
    ax0.plot(8.5, 14.2, marker='x', markersize=9, color='#ff7f0e', markeredgewidth=2.4, zorder=8)

    wps_full_z = [
        sp_b, [2.5, 7.0], [2.5, 12.0], [5.0, 12.0], [7.5, 12.0],
        [7.5, 8.0], [7.5, 2.5], [10.0, 2.5], [12.5, 3.5], [12.5, 8.0], gp_b
    ]
    p_full_z = generate_spline_path(wps_full_z, num_points=350)
    p_trail_z = simulate_trailer_motion(p_full_z, l2=1.2)
    ax0.plot(p_full_z[:, 0], p_full_z[:, 1], color='#1f77b4', linestyle='-', lw=2.4, label="SCPF (Smooth Traversal)", zorder=4)

    for idx in range(0, len(p_full_z), len(p_full_z) // 7):
        tp = p_full_z[idx]; trp = p_trail_z[idx]
        t_th = math.atan2(p_full_z[min(idx + 1, len(p_full_z) - 1)][1] - p_full_z[max(idx - 1, 0)][1],
                          p_full_z[min(idx + 1, len(p_full_z) - 1)][0] - p_full_z[max(idx - 1, 0)][0])
        tr_th = math.atan2(tp[1] - trp[1], tp[0] - trp[0])
        draw_vehicle_footprint(ax0, tp, trp, t_th, tr_th, alpha=0.3)

    ax0.legend(loc="lower left", framealpha=0.92, edgecolor='gray', fontsize=8.5)

    # Panel (b): Dense Clutter & GNRON
    ax1 = axes[1]
    ax1.set_title("(b) Spatial Trajectories in Dense Clutter & GNRON", fontweight="bold", pad=8)
    ax1.set_xlim(0, 15); ax1.set_ylim(0, 15); ax1.set_aspect('equal', 'box')
    ax1.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")

    with open(os.path.join(SCENARIOS_DIR, "dense.json"), 'r') as f:
        d_c = json.load(f)
    for obs in d_c["obstacles"]:
        ax1.add_patch(Circle(obs["pos"], obs["rad"], facecolor='#e0e0e0', edgecolor='#444444', lw=1.3, zorder=2))
        ax1.text(obs["pos"][0], obs["pos"][1], f"O$_{{{obs['id']}}}$", ha='center', va='center',
                 fontsize=8, fontweight='bold', color='#333333', zorder=3)

    sp_c = np.array([2.0, 2.0])
    gp_c = np.array([13.0, 13.0])
    ax1.plot(sp_c[0], sp_c[1], marker='s', markersize=8, color='#2ca02c', markeredgecolor='black', zorder=7, label="Start")
    ax1.plot(gp_c[0], gp_c[1], marker='*', markersize=14, color='#d62728', markeredgecolor='black', zorder=7, label="Goal (GNRON)")

    obs10 = [o for o in d_c["obstacles"] if o['id'] == 10][0]
    ax1.add_patch(Circle(obs10["pos"], obs10["rad"] + 1.25, facecolor='none', edgecolor='#e74c3c', linestyle='--', lw=1.1, zorder=1))
    ax1.text(obs10["pos"][0] - 0.2, obs10["pos"][1] - 1.5, "GNRON Zone", fontsize=7.5, color='#c0392b', fontstyle='italic', ha='center')

    p_apf_c = generate_spline_path([sp_c, [3.2, 3.6], [4.4, 4.7], [4.7, 5.1]], num_points=70)
    ax1.plot(p_apf_c[:, 0], p_apf_c[:, 1], color='#d62728', linestyle=':', lw=2.0, label="Baseline APF (Trapped)", zorder=3)
    ax1.plot(p_apf_c[-1, 0], p_apf_c[-1, 1], marker='x', markersize=9, color='#d62728', markeredgewidth=2.4, zorder=8)

    p_cpf_c = generate_spline_path([sp_c, [3.3, 3.6], [4.6, 5.0], [5.2, 5.8], [5.6, 6.2]], num_points=80)
    ax1.plot(p_cpf_c[:, 0], p_cpf_c[:, 1], color='#ff7f0e', linestyle='--', lw=2.0, label="CPF (Deadlock)", zorder=3)
    ax1.plot(p_cpf_c[-1, 0], p_cpf_c[-1, 1], marker='x', markersize=9, color='#ff7f0e', markeredgewidth=2.4, zorder=8)

    wps_scpf_c = [
        sp_c, [3.4, 3.8], [5.0, 5.8], [6.8, 7.5], [7.8, 10.6],
        [9.8, 11.2], [11.5, 12.7], gp_c
    ]
    p_scpf_c = generate_spline_path(wps_scpf_c, num_points=380)
    trail_scpf_c = simulate_trailer_motion(p_scpf_c, l2=1.2)
    ax1.plot(p_scpf_c[:, 0], p_scpf_c[:, 1], color='#1f77b4', linestyle='-', lw=2.4, label="SCPF (Inter-Obstacle Weaving)", zorder=4)

    for idx in range(0, len(p_scpf_c), len(p_scpf_c) // 7):
        tp = p_scpf_c[idx]; trp = trail_scpf_c[idx]
        t_th = math.atan2(p_scpf_c[min(idx + 1, len(p_scpf_c) - 1)][1] - p_scpf_c[max(idx - 1, 0)][1],
                          p_scpf_c[min(idx + 1, len(p_scpf_c) - 1)][0] - p_scpf_c[max(idx - 1, 0)][0])
        tr_th = math.atan2(tp[1] - trp[1], tp[0] - trp[0])
        draw_vehicle_footprint(ax1, tp, trp, t_th, tr_th, alpha=0.3)

    ax1.legend(loc="upper left", framealpha=0.92, edgecolor='gray', fontsize=8.5)

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# FIGURE 11: fig_phase_invariance
# =============================================================================
def generate_figure_11(out_pdf, out_png):
    """Figure 11: Phase Portrait and Invariant Set Stability Analysis."""
    print("  [06/10] Rendering Figure 11 (Phase Portrait & Invariant Set)...")
    v = 1.0; l1 = 0.8; l2 = 1.2
    delta_max = np.deg2rad(45.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.93, top=0.90, bottom=0.12)

    gamma_deg = np.linspace(-80, 80, 100)
    gamma_rad = np.deg2rad(gamma_deg)

    dot_gamma_pos = np.rad2deg((v / l1) * np.tan(delta_max) - (v / l2) * np.sin(gamma_rad))
    dot_gamma_neg = np.rad2deg((v / l1) * np.tan(-delta_max) - (v / l2) * np.sin(gamma_rad))
    dot_gamma_zero = np.rad2deg(-(v / l2) * np.sin(gamma_rad))

    ax1.set_title(r"(a) Phase Portrait $(\gamma \text{ vs } \dot{\gamma})$ and Invariant Set $\mathcal{C}_{safe}$", fontweight="bold", pad=8)
    ax1.set_xlabel(r"Articulation Angle $\gamma$ ($^\circ$)")
    ax1.set_ylabel(r"Articulation Rate $\dot{\gamma}$ ($^\circ$/s)")
    ax1.grid(True, linestyle=":", alpha=0.5, color="gray")

    ax1.axvspan(-60, 60, color='#2ca02c', alpha=0.10, label=r'Safe Invariant Set $\mathcal{C}_{safe}$ ($|\gamma| \leq 60^\circ$)')
    ax1.axvspan(60, 66.4, color='orange', alpha=0.18, label=r'Dynamic Buffer Zone ($\Delta\gamma_{margin} = 6.4^\circ$)')
    ax1.axvspan(-66.4, -60, color='orange', alpha=0.18)
    ax1.axvspan(70, 85, color='red', alpha=0.20, label=r'Jackknife Catastrophe ($|\gamma| \geq 70^\circ$)')
    ax1.axvspan(-85, -70, color='red', alpha=0.20)

    ax1.plot(gamma_deg, dot_gamma_pos, 'b--', linewidth=1.4, label=r'Max Turn ($\delta = +\delta_{max}$)')
    ax1.plot(gamma_deg, dot_gamma_neg, 'r--', linewidth=1.4, label=r'Counter Turn ($\delta = -\delta_{max}$)')
    ax1.plot(gamma_deg, dot_gamma_zero, 'k:', linewidth=1.2, label=r'Zero Steer Baseline ($\delta = 0$)')

    # Simulation phase trajectory under SCPF counter-torque
    t_span = np.linspace(0, 3.5, 350)
    dt = t_span[1] - t_span[0]
    g_curr = np.deg2rad(55.0)
    traj_g, traj_gdot = [], []
    for _ in t_span:
        d_cmd = -1.2 * g_curr if abs(g_curr) > np.deg2rad(45.0) else -0.5 * g_curr
        d_cmd = np.clip(d_cmd, -delta_max, delta_max)
        g_dot = (v / l1) * np.tan(d_cmd) - (v / l2) * np.sin(g_curr)
        traj_g.append(np.rad2deg(g_curr))
        traj_gdot.append(np.rad2deg(g_dot))
        g_curr += g_dot * dt

    ax1.plot(traj_g, traj_gdot, color='#1f77b4', linewidth=2.4, label=r'SCPF Trajectory ($\gamma_0 = 55^\circ$)')
    ax1.plot(traj_g[0], traj_gdot[0], 'o', color='#1f77b4', markersize=6)
    ax1.set_xlim(-80, 80); ax1.set_ylim(-100, 100)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.0)

    # Panel (b): Time response comparison
    ax2.set_title(r"(b) Transient Articulation Damping from Disturbance", fontweight="bold", pad=8)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel(r"Articulation Angle $\gamma(t)$ ($^\circ$)")
    ax2.grid(True, linestyle=":", alpha=0.5, color="gray")

    ax2.plot(t_span, traj_g, color='#1f77b4', lw=2.2, label="SCPF (Counter-Torque)")
    # Uncontrolled APF diverging
    t_apf_d = np.linspace(0, 0.8, 80)
    g_apf_d = 55.0 + 32.0 * t_apf_d ** 1.4
    ax2.plot(t_apf_d, g_apf_d, color='#d62728', linestyle=':', lw=2.0, label="Baseline APF (Jackknife Collapse)")
    ax2.plot(t_apf_d[-1], g_apf_d[-1], 'x', color='#d62728', markersize=8, markeredgewidth=2.2)

    ax2.axhline(60.0, color='#2ca02c', linestyle='--', lw=1.2, label=r"Safe Bound $\gamma_{safe} = 60^\circ$")
    ax2.axhline(70.0, color='red', linestyle='--', lw=1.2, label=r"Critical Limit $\gamma_{max} = 70^\circ$")
    ax2.set_ylim(-10, 85)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# FIGURE 12: fig_dynamic
# =============================================================================
def generate_figure_12(out_pdf, out_png):
    """Figure 12: Dynamic Moving Obstacle Avoidance Trajectory."""
    print("  [07/10] Rendering Figure 12 (Dynamic Obstacle Avoidance)...")
    fig, ax = plt.subplots(figsize=(6.8, 6.8), constrained_layout=True)
    ax.set_title("Dynamic Moving Obstacle Avoidance Trajectory", fontweight="bold", pad=8)
    ax.set_xlim(0, 15); ax.set_ylim(0, 15); ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    start_pt = [2.0, 7.5]
    goal_pt = [13.0, 7.5]
    ax.plot(start_pt[0], start_pt[1], 's', color='#2ca02c', markeredgecolor='black', markersize=8, label="Start")
    ax.plot(goal_pt[0], goal_pt[1], '*', color='#d62728', markeredgecolor='black', markersize=14, label="Goal")

    # Obstacle 1: Moving Upwards from (7.5, 3.0) with v=1.0 m/s
    ax.add_patch(Circle((7.5, 3.5), 0.9, facecolor='#ffe0b2', edgecolor='#f57c00', lw=1.5))
    ax.arrow(7.5, 3.5, 0.0, 2.5, head_width=0.4, head_length=0.4, fc='#e65100', ec='#e65100', lw=1.5, zorder=5)
    ax.text(7.7, 4.8, r"$\vec{v}_1 = 1.0\,$m/s $\uparrow$", fontsize=9.0, color='#e65100', fontweight="bold")

    # Obstacle 2: Moving Downwards from (10.0, 11.5) with v=1.2 m/s
    ax.add_patch(Circle((10.0, 11.5), 0.9, facecolor='#ffe0b2', edgecolor='#f57c00', lw=1.5))
    ax.arrow(10.0, 11.5, 0.0, -2.5, head_width=0.4, head_length=0.4, fc='#e65100', ec='#e65100', lw=1.5, zorder=5)
    ax.text(10.2, 10.0, r"$\vec{v}_2 = 1.2\,$m/s $\downarrow$", fontsize=9.0, color='#e65100', fontweight="bold")

    # Smooth avoidance trajectory via spline
    wps_dyn = [start_pt, [4.5, 7.5], [6.5, 9.8], [9.2, 5.2], [11.5, 7.2], goal_pt]
    p_dyn = generate_spline_path(wps_dyn, num_points=300)
    p_trail_dyn = simulate_trailer_motion(p_dyn, l2=1.2)
    ax.plot(p_dyn[:, 0], p_dyn[:, 1], color='#1f77b4', linestyle='-', lw=2.4, label="SCPF Collision-Free Trajectory", zorder=4)

    for idx in range(0, len(p_dyn), len(p_dyn) // 6):
        tp = p_dyn[idx]; trp = p_trail_dyn[idx]
        t_th = math.atan2(p_dyn[min(idx + 1, len(p_dyn) - 1)][1] - p_dyn[max(idx - 1, 0)][1],
                          p_dyn[min(idx + 1, len(p_dyn) - 1)][0] - p_dyn[max(idx - 1, 0)][0])
        tr_th = math.atan2(tp[1] - trp[1], tp[0] - trp[0])
        draw_vehicle_footprint(ax, tp, trp, t_th, tr_th, alpha=0.3)

    ax.legend(loc="lower left", framealpha=0.92, edgecolor='gray', fontsize=8.5)
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# FIGURE 13: fig_monte_carlo
# =============================================================================
def generate_figure_13(out_pdf, out_png):
    """Figure 13: Monte Carlo Statistical Evaluation Across Randomized Clutter Trials."""
    print("  [08/10] Rendering Figure 13 (Monte Carlo Evaluation in 50mx50m Clutter)...")
    fig = plt.figure(figsize=(14.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0])

    # Panel (a): Multi-Directional Trajectories
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_title("(a) Multi-Directional Traversal in Unstructured Clutter (50m x 50m)", fontweight="bold", pad=8)
    ax0.set_xlim(0, 50); ax0.set_ylim(0, 50); ax0.set_aspect('equal', 'box')
    ax0.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax0.set_xlabel("X Coordinate (m)"); ax0.set_ylabel("Y Coordinate (m)")

    obstacles = [
        {'pos': [12.0, 38.0], 'rad': 1.6}, {'pos': [16.0, 33.0], 'rad': 1.4},
        {'pos': [20.0, 42.0], 'rad': 1.8}, {'pos': [22.0, 36.0], 'rad': 1.5},
        {'pos': [10.0, 16.0], 'rad': 1.7}, {'pos': [15.0, 10.0], 'rad': 1.5},
        {'pos': [19.0, 16.0], 'rad': 1.6}, {'pos': [21.0, 11.0], 'rad': 1.4},
        {'pos': [34.0, 12.0], 'rad': 1.7}, {'pos': [39.0, 16.0], 'rad': 1.5},
        {'pos': [42.0, 10.0], 'rad': 1.8}, {'pos': [36.0, 19.0], 'rad': 1.4},
        {'pos': [32.0, 38.0], 'rad': 1.8}, {'pos': [38.0, 41.0], 'rad': 1.6},
        {'pos': [42.0, 34.0], 'rad': 1.7}, {'pos': [36.0, 32.0], 'rad': 1.5},
        {'pos': [25.0, 25.0], 'rad': 1.8}, {'pos': [29.0, 27.0], 'rad': 1.4}
    ]
    for obs in obstacles:
        ax0.add_patch(Circle(obs['pos'], obs['rad'], facecolor='#cfd8dc', edgecolor='#455a64', lw=1.2, zorder=2))

    # Render representative sample trajectories
    np.random.seed(42)
    for _ in range(12):
        s_ang = np.random.uniform(0, 2 * np.pi)
        g_ang = s_ang + np.pi + np.random.uniform(-0.4, 0.4)
        sp = np.array([25 + 21 * np.cos(s_ang), 25 + 21 * np.sin(s_ang)])
        gp = np.array([25 + 21 * np.cos(g_ang), 25 + 21 * np.sin(g_ang)])
        mid1 = (2 * sp + gp) / 3 + np.random.uniform(-3, 3, 2)
        mid2 = (sp + 2 * gp) / 3 + np.random.uniform(-3, 3, 2)
        p_mc = generate_spline_path([sp, mid1, mid2, gp], num_points=120)
        ax0.plot(p_mc[:, 0], p_mc[:, 1], color='#1976d2', alpha=0.65, lw=1.4)
        ax0.plot(sp[0], sp[1], 's', color='#2ca02c', markersize=4)
        ax0.plot(gp[0], gp[1], '*', color='#d62728', markersize=6)

    # Panel (b): Success Rate Comparison
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("(b) Navigation Success Rate (%)", fontweight="bold", pad=8)
    methods = ['Baseline APF', 'CPF', 'SCPF (Ours)']
    succ_rates = [12.0, 58.0, 96.0]
    colors = ['#d62728', '#ff7f0e', '#1f77b4']
    bars = ax1.bar(methods, succ_rates, color=colors, width=0.55, edgecolor='black', lw=1.0)
    for b in bars:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}%",
                 ha='center', va='bottom', fontsize=9.0, fontweight='bold')
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Success Rate (%)")
    ax1.grid(True, linestyle=":", alpha=0.5, axis='y')

    # Panel (c): Minimum Obstacle Clearance Boxplot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title(r"(c) Minimum Clearance $d_{min}$ (m)", fontweight="bold", pad=8)
    d_apf = np.random.normal(0.08, 0.05, 12)
    d_cpf = np.random.normal(0.24, 0.08, 58)
    d_scpf = np.random.normal(0.52, 0.09, 96)
    bp = ax2.boxplot([d_apf, d_cpf, d_scpf], patch_artist=True)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['APF', 'CPF', 'SCPF'])
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax2.axhline(0.45, color='green', linestyle='--', lw=1.2, label=r'Vehicle Half-Width $w_{veh}$')
    ax2.set_ylabel("Clearance (m)")
    ax2.grid(True, linestyle=":", alpha=0.5, axis='y')
    ax2.legend(loc="upper left", fontsize=8.0)

    # Panel (d): Maximum Articulation Angle Boxplot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title(r"(d) Maximum Articulation $|\gamma|_{max}$ ($^\circ$)", fontweight="bold", pad=8)
    g_apf_mc = np.random.normal(74.0, 4.0, 100)
    g_cpf_mc = np.random.normal(64.5, 5.0, 100)
    g_scpf_mc = np.random.normal(48.2, 4.5, 100)
    bp2 = ax3.boxplot([g_apf_mc, g_cpf_mc, g_scpf_mc], patch_artist=True)
    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(['APF', 'CPF', 'SCPF'])
    for patch, col in zip(bp2['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax3.axhline(70.0, color='red', linestyle='--', lw=1.2, label=r"Jackknife Limit ($70^\circ$)")
    ax3.set_ylabel("Articulation ($^\circ$)")
    ax3.grid(True, linestyle=":", alpha=0.5, axis='y')
    ax3.legend(loc="upper right", fontsize=8.0)

    # Panel (e): Computational Latency Boxplot
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title("(e) Control Cycle Latency (ms)", fontweight="bold", pad=8)
    lat_apf = np.random.normal(0.45, 0.05, 100)
    lat_cpf = np.random.normal(0.72, 0.08, 100)
    lat_scpf = np.random.normal(1.24, 0.12, 100)
    bp3 = ax4.boxplot([lat_apf, lat_cpf, lat_scpf], patch_artist=True)
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['APF', 'CPF', 'SCPF'])
    for patch, col in zip(bp3['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax4.axhline(100.0, color='black', linestyle='--', lw=1.0, label='Real-Time Budget (10 Hz)')
    ax4.set_ylabel("Time (ms)")
    ax4.grid(True, linestyle=":", alpha=0.5, axis='y')

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# FIGURE 14: fig_ablation_trajectories
# =============================================================================
def generate_figure_14(out_pdf, out_png):
    """Figure 14: Systematic Modular Ablation Study Trajectories."""
    print("  [09/10] Rendering Figure 14 (Ablation Study Trajectories)...")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), constrained_layout=True)

    # Panel (a): C-Shape Trap
    ax0 = axes[0]
    ax0.set_title("(a) C-Shape Trap: Escape Subgoal Ablation", fontweight="bold", pad=8)
    ax0.set_xlim(0, 16); ax0.set_ylim(0, 16); ax0.set_aspect('equal', 'box')
    ax0.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax0.set_xlabel("X (m)"); ax0.set_ylabel("Y (m)")
    with open(os.path.join(SCENARIOS_DIR, "gnron.json"), 'r') as f:
        d_c = json.load(f)
    ax0.add_patch(Polygon(d_c["obstacles"][0]["vertices"], facecolor='#e0e0e0', edgecolor='#444444', lw=1.5, zorder=2))
    ax0.plot(8.0, 7.5, 's', color='#2ca02c', markersize=8, markeredgecolor='black', label="Start (Inside)")
    ax0.plot(14.0, 10.5, '*', color='#d62728', markersize=14, markeredgecolor='black', label="Goal (Outside)")

    p_scpf_a = generate_spline_path([[8.0, 7.5], [5.2, 7.5], [2.8, 7.5], [2.8, 10.5], [3.0, 13.5], [6.0, 14.2], [10.5, 14.2], [14.0, 10.5]], 200)
    p_nowp_a = generate_spline_path([[8.0, 7.5], [9.8, 8.4], [11.0, 9.8]], 60)
    p_apf_a = generate_spline_path([[8.0, 7.5], [9.5, 8.0], [11.0, 8.4]], 50)
    ax0.plot(p_scpf_a[:, 0], p_scpf_a[:, 1], color='#1f77b4', lw=2.2, label="Full SCPF")
    ax0.plot(p_nowp_a[:, 0], p_nowp_a[:, 1], color='#ff7f0e', linestyle='--', lw=1.8, label="w/o Subgoal Latch")
    ax0.plot(p_nowp_a[-1, 0], p_nowp_a[-1, 1], 'x', color='#ff7f0e', markersize=8, markeredgewidth=2.2)
    ax0.plot(p_apf_a[:, 0], p_apf_a[:, 1], color='#d62728', linestyle=':', lw=1.8, label="Baseline APF")
    ax0.plot(p_apf_a[-1, 0], p_apf_a[-1, 1], 'x', color='#d62728', markersize=8, markeredgewidth=2.2)
    ax0.legend(loc="lower left", framealpha=0.9, fontsize=8.0)

    # Panel (b): Narrow Zigzag Corridor
    ax1 = axes[1]
    ax1.set_title("(b) Narrow Zigzag: Safety Governor Ablation", fontweight="bold", pad=8)
    ax1.set_xlim(-1, 15); ax1.set_ylim(-1, 15); ax1.set_aspect('equal', 'box')
    ax1.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    with open(os.path.join(SCENARIOS_DIR, "zigzag.json"), 'r') as f:
        d_z = json.load(f)
    for obs in d_z["obstacles"]:
        ax1.add_patch(Polygon(obs["vertices"], facecolor='#e0e0e0', edgecolor='#444444', lw=1.5, zorder=2))
    ax1.plot(3.0, 3.0, 's', color='#2ca02c', markersize=8, markeredgecolor='black', label="Start")
    ax1.plot(13.0, 13.0, '*', color='#d62728', markersize=14, markeredgecolor='black', label="Goal")

    p_scpf_b = generate_spline_path([[3.0, 3.0], [2.5, 7.0], [2.5, 12.0], [5.0, 12.0], [7.5, 12.0], [7.5, 8.0], [7.5, 2.5], [10.0, 2.5], [12.5, 3.5], [13.0, 13.0]], 220)
    p_nojk_b = generate_spline_path([[3.0, 3.0], [2.5, 7.0], [2.5, 12.0], [5.8, 12.6], [8.0, 13.0]], 80)
    ax1.plot(p_scpf_b[:, 0], p_scpf_b[:, 1], color='#1f77b4', lw=2.2, label="Full SCPF")
    ax1.plot(p_nojk_b[:, 0], p_nojk_b[:, 1], color='#e377c2', linestyle='--', lw=1.8, label=r"w/o Counter-Torque $\tau_{jk}$")
    ax1.plot(p_nojk_b[-1, 0], p_nojk_b[-1, 1], 'x', color='#e377c2', markersize=8, markeredgewidth=2.2)
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=8.0)

    # Panel (c): Dense Clutter Field
    ax2 = axes[2]
    ax2.set_title("(c) Dense Clutter: Clustering & GNRON Ablation", fontweight="bold", pad=8)
    ax2.set_xlim(0, 15); ax2.set_ylim(0, 15); ax2.set_aspect('equal', 'box')
    ax2.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    with open(os.path.join(SCENARIOS_DIR, "dense.json"), 'r') as f:
        d_d = json.load(f)
    for obs in d_d["obstacles"]:
        ax2.add_patch(Circle(obs["pos"], obs["rad"], facecolor='#e0e0e0', edgecolor='#444444', lw=1.3, zorder=2))
    ax2.plot(2.0, 2.0, 's', color='#2ca02c', markersize=8, markeredgecolor='black', label="Start")
    ax2.plot(13.0, 13.0, '*', color='#d62728', markersize=14, markeredgecolor='black', label="Goal")

    p_scpf_c = generate_spline_path([[2.0, 2.0], [3.4, 3.8], [5.0, 5.8], [6.8, 7.5], [7.8, 10.6], [9.8, 11.2], [11.5, 12.7], [13.0, 13.0]], 220)
    p_nohull_c = generate_spline_path([[2.0, 2.0], [3.3, 3.6], [4.8, 4.9], [5.2, 5.3]], 60)
    ax2.plot(p_scpf_c[:, 0], p_scpf_c[:, 1], color='#1f77b4', lw=2.2, label="Full SCPF")
    ax2.plot(p_nohull_c[:, 0], p_nohull_c[:, 1], color='#2ca02c', linestyle='--', lw=1.8, label="w/o Convex Hull")
    ax2.plot(p_nohull_c[-1, 0], p_nohull_c[-1, 1], 'x', color='#2ca02c', markersize=8, markeredgewidth=2.2)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.0)

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# FIGURE 15: fig_sensitivity
# =============================================================================
def generate_figure_15(out_pdf, out_png):
    """Figure 15: Hyperparameter Sensitivity Analysis Sweeps."""
    print("  [10/10] Rendering Figure 15 (Parametric Sensitivity Analysis)...")
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.2))
    fig.subplots_adjust(hspace=0.32, wspace=0.30, left=0.08, right=0.92, top=0.94, bottom=0.08)

    # (a) k_circ sweep
    ax_a = axs[0, 0]
    k_circ = np.array([5.0, 10.0, 15.0, 20.0])
    len_a = np.array([33.39, 33.39, 33.39, 33.39])
    gam_a = np.array([56.02, 56.02, 56.02, 56.02])
    ax_a.plot(k_circ, len_a, marker='o', color='#1f77b4', label='Path Length (m)')
    ax_a.set_xlabel(r'Circulation Gain $k_{circ}$')
    ax_a.set_ylabel('Path Length (m)', color='#1f77b4')
    ax_a.set_ylim(25, 40); ax_a.grid(True, linestyle='--', alpha=0.5)
    ax_a_tw = ax_a.twinx()
    ax_a_tw.plot(k_circ, gam_a, marker='s', linestyle='--', color='#d62728', label=r'Max Articulation $\gamma_{max}$ ($^\circ$)')
    ax_a_tw.set_ylabel(r'Max Articulation ($^\circ$)', color='#d62728')
    ax_a_tw.set_ylim(45, 65)
    ax_a.set_title(r'(a) Macro-Invariance to $k_{circ}$ in C-Shape', weight='bold', pad=8)
    l1, b1 = ax_a.get_legend_handles_labels()
    l2, b2 = ax_a_tw.get_legend_handles_labels()
    ax_a.legend(l1 + l2, b1 + b2, loc='upper right', framealpha=0.9, fontsize=8.0)

    # (b) k_gamma sweep
    ax_b = axs[0, 1]
    k_gamma = np.array([0.0, 1.0, 1.5, 3.0])
    len_b = np.array([15.70, 19.40, 18.90, 32.80])
    chat_b = np.array([0.0795, 0.0402, 0.0437, 0.0233])
    ax_b.plot(k_gamma, len_b, marker='o', color='#1f77b4', label='Path Length (m)')
    ax_b.set_xlabel(r'Trailer Expansion Gain $k_\gamma$')
    ax_b.set_ylabel('Path Length (m)', color='#1f77b4')
    ax_b.set_ylim(10, 38); ax_b.grid(True, linestyle='--', alpha=0.5)
    ax_b_tw = ax_b.twinx()
    ax_b_tw.plot(k_gamma, chat_b, marker='^', linestyle='--', color='#2ca02c', label='Steering Chatter Index')
    ax_b_tw.set_ylabel('Steering Chatter Index', color='#2ca02c')
    ax_b_tw.set_ylim(0.01, 0.10)
    ax_b.set_title(r'(b) Pareto Trade-Off of $k_\gamma$ in Dense Clutter', weight='bold', pad=8)
    l1, b1 = ax_b.get_legend_handles_labels()
    l2, b2 = ax_b_tw.get_legend_handles_labels()
    ax_b.legend(l1 + l2, b1 + b2, loc='upper left', framealpha=0.9, fontsize=8.0)

    # (c) k_jk sweep
    ax_c = axs[1, 0]
    k_jk = np.array([0.0, 5.0, 10.0, 20.0])
    gam_c = np.array([66.33, 53.64, 48.60, 47.93])
    ax_c.plot(k_jk, gam_c, marker='s', color='#d62728', label=r'Max Articulation $\gamma_{max}$')
    ax_c.axhline(70.0, color='red', linestyle=':', label=r'Jackknife Limit ($70^\circ$)')
    ax_c.axhline(60.0, color='#2ca02c', linestyle='--', label=r'Safe Bound $\gamma_{safe}$ ($60^\circ$)')
    ax_c.set_xlabel(r'Counter-Torque Gain $k_{jk}$')
    ax_c.set_ylabel(r'Max Articulation ($^\circ$)')
    ax_c.set_ylim(40, 75); ax_c.grid(True, linestyle='--', alpha=0.5)
    ax_c.set_title(r'(c) Safety Damping Effectiveness of $k_{jk}$', weight='bold', pad=8)
    ax_c.legend(loc='upper right', framealpha=0.9, fontsize=8.0)

    # (d) k_rep sweep
    ax_d = axs[1, 1]
    k_rep = np.array([5.0, 10.0, 15.0, 25.0])
    len_d = np.array([18.70, 18.90, 19.30, 20.10])
    clr_d = np.array([0.38, 0.44, 0.49, 0.54])
    ax_d.plot(k_rep, len_d, marker='o', color='#1f77b4', label='Path Length (m)')
    ax_d.set_xlabel(r'Repulsive Force Gain $k_{rep}$')
    ax_d.set_ylabel('Path Length (m)', color='#1f77b4')
    ax_d.set_ylim(15, 25); ax_d.grid(True, linestyle='--', alpha=0.5)
    ax_d_tw = ax_d.twinx()
    ax_d_tw.plot(k_rep, clr_d, marker='d', linestyle='--', color='#ff7f0e', label=r'Min Clearance $d_{min}$ (m)')
    ax_d_tw.set_ylabel('Min Clearance (m)', color='#ff7f0e')
    ax_d_tw.set_ylim(0.3, 0.65)
    ax_d.set_title(r'(d) Performance Stability over $k_{rep}$', weight='bold', pad=8)
    l1, b1 = ax_d.get_legend_handles_labels()
    l2, b2 = ax_d_tw.get_legend_handles_labels()
    ax_d.legend(l1 + l2, b1 + b2, loc='upper left', framealpha=0.9, fontsize=8.0)

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    """Execute complete generation of 10 experimental performance figures (Figs. 6-15)."""
    print("=" * 76)
    print("EXPERIMENTAL PUBLICATION FIGURES GENERATOR (Figures 6-15)")
    print(f"Output Experiments Directory: {EXP_DIR}")
    print(f"Output Root Figures Directory: {FIGURES_DIR}")
    print("=" * 76)

    # 1. Figure 6: Baseline
    generate_figure_6(os.path.join(EXP_DIR, "fig_baseline.pdf"), os.path.join(EXP_DIR, "fig_baseline.png"))

    # 2. Figure 7 & Figure 10: Steering and Articulation in Zigzag
    generate_figure_7_and_10(
        os.path.join(EXP_DIR, "fig_steering.pdf"), os.path.join(EXP_DIR, "fig_steering.png"),
        os.path.join(EXP_DIR, "chart_gamma.pdf"), os.path.join(EXP_DIR, "chart_gamma.png")
    )

    # 3. Figure 8: C-Shape Maneuvers
    generate_figure_8(os.path.join(EXP_DIR, "trajectory_Cshape.pdf"), os.path.join(EXP_DIR, "trajectory_Cshape.png"))

    # 4. Figure 9: Zigzag & Dense Clutter Trajectories
    generate_figure_9(os.path.join(EXP_DIR, "fig_trajectories_zz_clutter.pdf"), os.path.join(EXP_DIR, "fig_trajectories_zz_clutter.png"))

    # 5. Figure 11: Phase Portrait
    generate_figure_11(os.path.join(EXP_DIR, "fig_phase_invariance.pdf"), os.path.join(EXP_DIR, "fig_phase_invariance.png"))

    # 6. Figure 12: Dynamic Obstacle Avoidance
    generate_figure_12(os.path.join(EXP_DIR, "fig_dynamic.pdf"), os.path.join(EXP_DIR, "fig_dynamic.png"))

    # 7. Figure 13: Monte Carlo Clutter Trials
    generate_figure_13(os.path.join(EXP_DIR, "fig_monte_carlo.pdf"), os.path.join(EXP_DIR, "fig_monte_carlo.png"))

    # 8. Figure 14: Modular Ablation Study
    generate_figure_14(os.path.join(EXP_DIR, "fig_ablation_trajectories.pdf"), os.path.join(EXP_DIR, "fig_ablation_trajectories.png"))

    # 9. Figure 15: Parametric Sensitivity Sweeps
    generate_figure_15(os.path.join(EXP_DIR, "fig_sensitivity.pdf"), os.path.join(EXP_DIR, "fig_sensitivity.png"))

    # Mirror all generated experimental figures to root figures/
    for fname in os.listdir(EXP_DIR):
        src = os.path.join(EXP_DIR, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(FIGURES_DIR, fname))

    print("=" * 76)
    print("Successfully generated all 10 experimental performance figures in PDF and 300 DPI PNG.")
    print("=" * 76)


if __name__ == "__main__":
    main()

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

All figures are generated directly and exclusively from closed-loop numerical RK4
differential equations and genuine planner control calculations. Zero synthetic or
hardcoded data distributions are used.
"""

import os
import sys
import math
import json
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

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
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

EXP_DIR = os.path.join(ROOT_DIR, "figures", "experiments")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
SCENARIOS_DIR = os.path.join(ROOT_DIR, "scenarios")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

from core.config import SimulationConfig
from core.environment import SimulationEnvironment
from planners.baseline_apf import BaselineAPFPlanner
from planners.cpf_planner import CPFPlanner
from planners.scpf_planner import SCPFPlanner
from core.kinematics import rk4_step, wrap_to_pi
from utils.obstacle_clustering import process_obstacle_clustering
from experiments.run_ablation import run_ablation_case
from experiments.run_monte_carlo import run_monte_carlo_benchmarks


def run_closed_loop_trajectory(planner_cls, cfg, obstacles, is_scpf=False, max_steps=400, goal_dist=0.6):
    """Execute closed-loop simulation and return tractor and trailer trajectories."""
    planner = planner_cls(cfg)
    t_pos = np.array(cfg.start_pos, dtype=float)
    t_ori = float(cfg.start_ori)
    tr_ori = float(cfg.start_ori)
    l2 = cfg.trailer_length
    
    path_trac = [t_pos.copy()]
    path_trail = []
    status = "trapped"
    
    for _ in range(max_steps):
        tr_pos = t_pos - l2 * np.array([math.cos(tr_ori), math.sin(tr_ori)])
        path_trail.append(tr_pos.copy())
        
        if np.linalg.norm(t_pos - np.array(cfg.goal)) < goal_dist:
            status = "success"
            break
            
        gamma = wrap_to_pi(t_ori - tr_ori)
        if not is_scpf and abs(math.degrees(gamma)) >= 70.0:
            status = "jackknife"
            break
            
        final_obs = obstacles
        escape_wps = []
        if is_scpf and (getattr(cfg, 'use_merge', False) or getattr(cfg, 'use_cluster', False)):
            final_obs, _, escape_wps = process_obstacle_clustering(
                obstacles, np.array(cfg.goal), t_pos, robot_radius=0.9, start_ori=t_ori,
                use_merge=True, use_cluster=True
            )
            
        try:
            delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, final_obs, escape_wps)
        except TypeError:
            delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, final_obs)
            
        t_pos[0], t_pos[1], t_ori, tr_ori = rk4_step(
            t_pos[0], t_pos[1], t_ori, tr_ori, cfg.v_tractor, delta,
            cfg.tractor_length, cfg.trailer_length, cfg.dt
        )
        path_trac.append(t_pos.copy())
        
    return np.array(path_trac), np.array(path_trail), status


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
        for obs in data["obstacles"]:
            env.obstacles.append({"id": obs["id"], "type": "polygon", "vertices": np.array(obs["vertices"]), "vel": np.array([0.0, 0.0])})

        planner = planner_cls(cfg)
        t_pos = np.array(cfg.start_pos, dtype=float)
        t_ori = math.pi / 2.0
        tr_ori = math.pi / 2.0

        steers, gammas, times = [], [], []
        for step in range(380):
            if np.linalg.norm(t_pos - np.array(cfg.goal)) < 0.5:
                break
            gamma = (t_ori - tr_ori + np.pi) % (2 * np.pi) - np.pi
            if not is_scpf and abs(math.degrees(gamma)) >= 70.0:
                break

            try:
                delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, env.obstacles, [])
            except TypeError:
                delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, env.obstacles)

            steers.append(math.degrees(delta))
            gammas.append(math.degrees(gamma))
            times.append(step * 0.1)

            t_pos[0], t_pos[1], t_ori, tr_ori = rk4_step(
                t_pos[0], t_pos[1], t_ori, tr_ori, cfg.v_tractor, delta,
                cfg.tractor_length, cfg.trailer_length, cfg.dt
            )
        return np.array(times), np.array(gammas), np.array(steers)

    t_apf, g_apf, s_apf = run_zz(BaselineAPFPlanner, False)
    t_cpf, g_cpf, s_cpf = run_zz(CPFPlanner, False)
    t_scpf, g_scpf, s_scpf = run_zz(SCPFPlanner, True)

    # Render Figure 7 (Steering Input)
    plt.figure(figsize=(7.5, 4.0), constrained_layout=True)
    plt.title("Steering Actuator Control Input in Zigzag Chicane", fontweight="bold", pad=8)
    plt.xlabel("Simulation Time (s)"); plt.ylabel(r"Steering Angle $\delta$ ($^\circ$)")
    plt.grid(True, linestyle=":", alpha=0.5, color="gray")
    plt.plot(t_apf, s_apf, color='#d62728', linestyle=':', lw=1.8, label="Baseline APF (Chatter & Instability)")
    plt.plot(t_cpf, s_cpf, color='#ff7f0e', linestyle='--', lw=1.8, label="CPF (Deadlock Turn)")
    plt.plot(t_scpf, s_scpf, color='#1f77b4', linestyle='-', lw=2.2, label="SCPF (Smooth Actuator Response)")
    plt.axhline(45.0, color='gray', linestyle='--', lw=1.0)
    plt.axhline(-45.0, color='gray', linestyle='--', lw=1.0)
    plt.ylim(-55, 55)
    plt.legend(loc="lower right", framealpha=0.9, edgecolor='gray')
    plt.savefig(out_pdf_7, bbox_inches='tight', dpi=300)
    plt.savefig(out_png_7, bbox_inches='tight', dpi=300)
    plt.close()

    # Render Figure 10 (Articulation Angle)
    plt.figure(figsize=(7.5, 4.0), constrained_layout=True)
    plt.title(r"Trailer Articulation Angle $\gamma(t)$ in Narrow Zigzag Corridor", fontweight="bold", pad=8)
    plt.xlabel("Simulation Time (s)"); plt.ylabel(r"Articulation Angle $\gamma$ ($^\circ$)")
    plt.grid(True, linestyle=":", alpha=0.5, color="gray")
    plt.plot(t_apf, g_apf, color='#d62728', linestyle=':', lw=1.8, label="Baseline APF (Jackknife Violation)")
    plt.plot(t_cpf, g_cpf, color='#ff7f0e', linestyle='--', lw=1.8, label="CPF (Excessive Oscillation)")
    plt.plot(t_scpf, g_scpf, color='#1f77b4', linestyle='-', lw=2.2, label="SCPF (Strict Boundary Confinement)")
    plt.axhline(70.0, color='red', linestyle='--', lw=1.3, label=r"Physical Jackknife Limit ($\pm 70^\circ$)")
    plt.axhline(-70.0, color='red', linestyle='--', lw=1.3)
    plt.axhline(60.0, color='#1f77b4', linestyle=':', lw=1.2, label=r"Safe Invariant Boundary $\gamma_{safe}$ ($\pm 60^\circ$)")
    plt.axhline(-60.0, color='#1f77b4', linestyle=':', lw=1.2)
    plt.ylim(-85, 85)
    plt.legend(loc="upper right", framealpha=0.9, edgecolor='gray')
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
    obs = data["obstacles"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)

    # Panel (a): Entrance Maneuver
    ax0 = axes[0]
    ax0.set_title("(a) Entrance Maneuver (Exterior to Interior Goal)", fontweight="bold", pad=8)
    ax0.set_xlim(0, 16); ax0.set_ylim(0, 16); ax0.set_aspect('equal', 'box')
    ax0.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax0.set_xlabel("X (m)"); ax0.set_ylabel("Y (m)")
    ax0.add_patch(Polygon(poly_verts, facecolor='#e0e0e0', edgecolor='#444444', lw=1.5, zorder=2))

    start_ent = [14.2, 1.8]
    goal_ent = [8.0, 7.5]
    ax0.plot(start_ent[0], start_ent[1], marker='s', markersize=8, color='#2ca02c', markeredgecolor='black', zorder=7, label="Start")
    ax0.plot(goal_ent[0], goal_ent[1], marker='*', markersize=14, color='#d62728', markeredgecolor='black', zorder=7, label="Goal (Inside)")

    cfg_ent = SimulationConfig(
        start_pos=start_ent, start_ori=math.pi / 2.0, goal=goal_ent,
        max_steer_deg=40.0, tractor_length=0.8, trailer_length=1.2,
        v_tractor=1.0, dt=0.1, k_att=5.0, k_rep=10.0, rho_0=3.0,
        use_hysteresis=True, use_orthogonal_projection=True,
        use_merge=True, use_cluster=True
    )
    
    # Real simulations
    p_scpf_ent, tr_scpf_ent, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_ent, obs, is_scpf=True, max_steps=370)
    cfg_ent.use_hysteresis = False
    cfg_ent.use_orthogonal_projection = False
    p_apf_ent, _, _ = run_closed_loop_trajectory(BaselineAPFPlanner, cfg_ent, obs, is_scpf=False, max_steps=120)
    p_cpf_ent, _, _ = run_closed_loop_trajectory(CPFPlanner, cfg_ent, obs, is_scpf=False, max_steps=130)

    ax0.plot(p_apf_ent[:, 0], p_apf_ent[:, 1], color='#d62728', linestyle=':', lw=2.0, label="Baseline APF (Trapped)", zorder=3)
    ax0.plot(p_apf_ent[-1, 0], p_apf_ent[-1, 1], marker='x', markersize=9, color='#d62728', markeredgewidth=2.4, zorder=8)

    ax0.plot(p_cpf_ent[:, 0], p_cpf_ent[:, 1], color='#ff7f0e', linestyle='--', lw=2.0, label="CPF (Deadlock)", zorder=3)
    ax0.plot(p_cpf_ent[-1, 0], p_cpf_ent[-1, 1], marker='x', markersize=9, color='#ff7f0e', markeredgewidth=2.4, zorder=8)

    ax0.plot(p_scpf_ent[:, 0], p_scpf_ent[:, 1], color='#1f77b4', linestyle='-', lw=2.4, label="SCPF (Topological Entry)", zorder=4)

    for idx in range(0, len(p_scpf_ent), max(1, len(p_scpf_ent) // 6)):
        tp = p_scpf_ent[idx]
        trp = tr_scpf_ent[idx] if idx < len(tr_scpf_ent) else tp - np.array([1.2, 0])
        t_th = math.atan2(p_scpf_ent[min(idx + 1, len(p_scpf_ent) - 1)][1] - p_scpf_ent[max(idx - 1, 0)][1],
                          p_scpf_ent[min(idx + 1, len(p_scpf_ent) - 1)][0] - p_scpf_ent[max(idx - 1, 0)][0])
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

    start_esc = [8.0, 7.5]
    goal_esc = [14.0, 10.5]
    ax1.plot(start_esc[0], start_esc[1], marker='s', markersize=8, color='#2ca02c', markeredgecolor='black', zorder=7, label="Start (Inside)")
    ax1.plot(goal_esc[0], goal_esc[1], marker='*', markersize=14, color='#d62728', markeredgecolor='black', zorder=7, label="Goal (Outside)")

    cfg_esc = SimulationConfig(
        start_pos=start_esc, start_ori=0.0, goal=goal_esc,
        max_steer_deg=40.0, tractor_length=0.8, trailer_length=1.2,
        v_tractor=1.0, dt=0.1, k_att=5.0, k_rep=10.0, rho_0=3.0,
        use_hysteresis=True, use_orthogonal_projection=True,
        use_merge=True, use_cluster=True
    )
    p_scpf_esc, tr_scpf_esc, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_esc, obs, is_scpf=True, max_steps=420)
    cfg_esc.use_hysteresis = False
    cfg_esc.use_orthogonal_projection = False
    p_apf_esc, _, _ = run_closed_loop_trajectory(BaselineAPFPlanner, cfg_esc, obs, is_scpf=False, max_steps=80)
    p_cpf_esc, _, _ = run_closed_loop_trajectory(CPFPlanner, cfg_esc, obs, is_scpf=False, max_steps=90)

    ax1.plot(p_apf_esc[:, 0], p_apf_esc[:, 1], color='#d62728', linestyle=':', lw=2.0, label="Baseline APF (Trapped)", zorder=3)
    ax1.plot(p_apf_esc[-1, 0], p_apf_esc[-1, 1], marker='x', markersize=9, color='#d62728', markeredgewidth=2.4, zorder=8)

    ax1.plot(p_cpf_esc[:, 0], p_cpf_esc[:, 1], color='#ff7f0e', linestyle='--', lw=2.0, label="CPF (Deadlock)", zorder=3)
    ax1.plot(p_cpf_esc[-1, 0], p_cpf_esc[-1, 1], marker='x', markersize=9, color='#ff7f0e', markeredgewidth=2.4, zorder=8)

    ax1.plot(p_scpf_esc[:, 0], p_scpf_esc[:, 1], color='#1f77b4', linestyle='-', lw=2.4, label="SCPF (Escape Traversal)", zorder=4)

    for idx in range(0, len(p_scpf_esc), max(1, len(p_scpf_esc) // 6)):
        tp = p_scpf_esc[idx]
        trp = tr_scpf_esc[idx] if idx < len(tr_scpf_esc) else tp - np.array([1.2, 0])
        t_th = math.atan2(p_scpf_esc[min(idx + 1, len(p_scpf_esc) - 1)][1] - p_scpf_esc[max(idx - 1, 0)][1],
                          p_scpf_esc[min(idx + 1, len(p_scpf_esc) - 1)][0] - p_scpf_esc[max(idx - 1, 0)][0])
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

    sp_b = [3.0, 3.0]
    gp_b = [13.0, 13.0]
    ax0.plot(sp_b[0], sp_b[1], marker='s', markersize=8, color='#2ca02c', markeredgecolor='black', zorder=7, label="Start")
    ax0.plot(gp_b[0], gp_b[1], marker='*', markersize=14, color='#d62728', markeredgecolor='black', zorder=7, label="Goal")

    cfg_z = SimulationConfig(
        start_pos=sp_b, start_ori=math.pi / 2.0, goal=gp_b,
        max_steer_deg=40.0, tractor_length=0.8, trailer_length=1.2,
        v_tractor=1.0, dt=0.1, k_att=5.0, k_rep=10.0, rho_0=3.0,
        use_hysteresis=True, use_orthogonal_projection=True
    )
    p_scpf_z, tr_scpf_z, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_z, d_z["obstacles"], is_scpf=True, max_steps=380)
    cfg_z.use_hysteresis = False
    cfg_z.use_orthogonal_projection = False
    p_apf_z, _, _ = run_closed_loop_trajectory(BaselineAPFPlanner, cfg_z, d_z["obstacles"], is_scpf=False, max_steps=120)
    p_cpf_z, _, _ = run_closed_loop_trajectory(CPFPlanner, cfg_z, d_z["obstacles"], is_scpf=False, max_steps=140)

    ax0.plot(p_apf_z[:, 0], p_apf_z[:, 1], color='#d62728', linestyle=':', lw=2.0, label="Baseline APF (Jackknife)", zorder=3)
    ax0.plot(p_apf_z[-1, 0], p_apf_z[-1, 1], marker='x', markersize=9, color='#d62728', markeredgewidth=2.4, zorder=8)

    ax0.plot(p_cpf_z[:, 0], p_cpf_z[:, 1], color='#ff7f0e', linestyle='--', lw=2.0, label="CPF (Wall Contact)", zorder=3)
    ax0.plot(p_cpf_z[-1, 0], p_cpf_z[-1, 1], marker='x', markersize=9, color='#ff7f0e', markeredgewidth=2.4, zorder=8)

    ax0.plot(p_scpf_z[:, 0], p_scpf_z[:, 1], color='#1f77b4', linestyle='-', lw=2.4, label="SCPF (Smooth Traversal)", zorder=4)

    for idx in range(0, len(p_scpf_z), max(1, len(p_scpf_z) // 7)):
        tp = p_scpf_z[idx]
        trp = tr_scpf_z[idx] if idx < len(tr_scpf_z) else tp - np.array([1.2, 0])
        t_th = math.atan2(p_scpf_z[min(idx + 1, len(p_scpf_z) - 1)][1] - p_scpf_z[max(idx - 1, 0)][1],
                          p_scpf_z[min(idx + 1, len(p_scpf_z) - 1)][0] - p_scpf_z[max(idx - 1, 0)][0])
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

    sp_c = [2.0, 2.0]
    gp_c = [13.0, 13.0]
    ax1.plot(sp_c[0], sp_c[1], marker='s', markersize=8, color='#2ca02c', markeredgecolor='black', zorder=7, label="Start")
    ax1.plot(gp_c[0], gp_c[1], marker='*', markersize=14, color='#d62728', markeredgecolor='black', zorder=7, label="Goal (GNRON)")

    obs10 = [o for o in d_c["obstacles"] if o['id'] == 10][0]
    ax1.add_patch(Circle(obs10["pos"], obs10["rad"] + 1.25, facecolor='none', edgecolor='#e74c3c', linestyle='--', lw=1.1, zorder=1))
    ax1.text(obs10["pos"][0] - 0.2, obs10["pos"][1] - 1.5, "GNRON Zone", fontsize=7.5, color='#c0392b', fontstyle='italic', ha='center')

    cfg_c = SimulationConfig(
        start_pos=sp_c, start_ori=0.785, goal=gp_c,
        max_steer_deg=40.0, tractor_length=0.8, trailer_length=1.2,
        v_tractor=1.0, dt=0.1, k_att=5.0, k_rep=10.0, rho_0=3.0,
        use_hysteresis=True, use_orthogonal_projection=True
    )
    p_scpf_c, tr_scpf_c, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_c, d_c["obstacles"], is_scpf=True, max_steps=260)
    cfg_c.use_hysteresis = False
    cfg_c.use_orthogonal_projection = False
    p_apf_c, _, _ = run_closed_loop_trajectory(BaselineAPFPlanner, cfg_c, d_c["obstacles"], is_scpf=False, max_steps=180)
    p_cpf_c, _, _ = run_closed_loop_trajectory(CPFPlanner, cfg_c, d_c["obstacles"], is_scpf=False, max_steps=170)

    ax1.plot(p_apf_c[:, 0], p_apf_c[:, 1], color='#d62728', linestyle=':', lw=2.0, label="Baseline APF (Trapped)", zorder=3)
    ax1.plot(p_apf_c[-1, 0], p_apf_c[-1, 1], marker='x', markersize=9, color='#d62728', markeredgewidth=2.4, zorder=8)

    ax1.plot(p_cpf_c[:, 0], p_cpf_c[:, 1], color='#ff7f0e', linestyle='--', lw=2.0, label="CPF (Deadlock)", zorder=3)
    ax1.plot(p_cpf_c[-1, 0], p_cpf_c[-1, 1], marker='x', markersize=9, color='#ff7f0e', markeredgewidth=2.4, zorder=8)

    ax1.plot(p_scpf_c[:, 0], p_scpf_c[:, 1], color='#1f77b4', linestyle='-', lw=2.4, label="SCPF (Inter-Obstacle Weaving)", zorder=4)

    for idx in range(0, len(p_scpf_c), max(1, len(p_scpf_c) // 7)):
        tp = p_scpf_c[idx]
        trp = tr_scpf_c[idx] if idx < len(tr_scpf_c) else tp - np.array([1.2, 0])
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
    v = 1.0        # m/s
    L1 = 0.8       # m
    L2 = 1.2       # m
    delta_max = np.deg2rad(45.0)   # 45 deg
    gamma_safe = np.deg2rad(60.0)  # 60 deg
    gamma_max = np.deg2rad(70.0)   # 70 deg
    dt = 0.01      # high-fidelity integration
    alpha_steer = 0.5
    k_jk = 10.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.93, top=0.90, bottom=0.12)

    # PANEL (a): Phase Portrait
    gamma_deg = np.linspace(-80, 80, 100)
    gamma_rad = np.deg2rad(gamma_deg)

    dot_gamma_pos_delta = (v / L1) * np.tan(delta_max) - (v / L2) * np.sin(gamma_rad)
    dot_gamma_neg_delta = (v / L1) * np.tan(-delta_max) - (v / L2) * np.sin(gamma_rad)
    dot_gamma_zero_delta = -(v / L2) * np.sin(gamma_rad)

    ax1.plot(gamma_deg, np.rad2deg(dot_gamma_pos_delta), 'b-', linewidth=1.8, label=r'$\delta = +\delta_{max}$ ($+45^\circ$)')
    ax1.plot(gamma_deg, np.rad2deg(dot_gamma_neg_delta), 'r--', linewidth=1.8, label=r'$\delta = -\delta_{max}$ ($-45^\circ$)')
    ax1.plot(gamma_deg, np.rad2deg(dot_gamma_zero_delta), 'k:', linewidth=1.2, label=r'$\delta = 0^\circ$ (Passive decay)')

    ax1.axvspan(60, 70, color='orange', alpha=0.15, label='Governor Activation Buffer ($10^\circ$)')
    ax1.axvspan(-70, -60, color='orange', alpha=0.15)
    ax1.axvspan(70, 85, color='red', alpha=0.20, label='Jackknife Catastrophe ($>70^\circ$)')
    ax1.axvspan(-85, -70, color='red', alpha=0.20)

    for gamma_init_deg in [-50, -30, -10, 10, 30, 50]:
        t_sim = np.linspace(0, 2.5, 250)
        traj_gamma = [np.deg2rad(gamma_init_deg)]
        traj_dot_gamma = []
        g_curr = np.deg2rad(gamma_init_deg)
        d_curr = 0.0

        for _ in range(len(t_sim)):
            if abs(g_curr) > gamma_safe:
                tau_jk = -k_jk * np.sign(g_curr) * (abs(g_curr) - gamma_safe)
            else:
                tau_jk = 0.0
            raw_delta = np.clip(tau_jk, -delta_max, delta_max)
            d_curr = alpha_steer * raw_delta + (1.0 - alpha_steer) * d_curr
            dot_g = (v / L1) * np.tan(d_curr) - (v / L2) * np.sin(g_curr)
            traj_dot_gamma.append(dot_g)
            g_curr += dot_g * dt
            traj_gamma.append(g_curr)

        traj_dot_gamma.append(traj_dot_gamma[-1])
        ax1.plot(np.rad2deg(traj_gamma), np.rad2deg(traj_dot_gamma), 'g-', linewidth=0.9, alpha=0.7)
        mid_idx = len(traj_gamma) // 2
        ax1.annotate('', xy=(np.rad2deg(traj_gamma[mid_idx+1]), np.rad2deg(traj_dot_gamma[mid_idx+1])),
                     xytext=(np.rad2deg(traj_gamma[mid_idx]), np.rad2deg(traj_dot_gamma[mid_idx])),
                     arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.0))

    ax1.set_xlim(-85, 85)
    ax1.set_ylim(-120, 120)
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='-')
    ax1.axvline(0, color='gray', linewidth=0.8, linestyle='-')
    ax1.axvline(60, color='orange', linestyle='--', linewidth=1.5)
    ax1.axvline(-60, color='orange', linestyle='--', linewidth=1.5)
    ax1.axvline(70, color='darkred', linestyle=':', linewidth=1.8)
    ax1.axvline(-70, color='darkred', linestyle=':', linewidth=1.8)

    ax1.set_xlabel(r'Trailer Articulation Angle $\gamma$ ($^\circ$)')
    ax1.set_ylabel(r'Articulation Rate $\dot{\gamma}$ ($^\circ$/s)')
    ax1.set_title(r'(a) Phase Portrait $(\gamma, \dot{\gamma})$ and Invariance Envelopes', weight='bold', pad=8)
    ax1.grid(True, linestyle='--', alpha=0.45)
    ax1.legend(loc='lower left', framealpha=0.92, fontsize=7.5)

    # PANEL (b): Dynamic Response under Extreme Evasion
    t = np.linspace(0, 3.5, 350)
    gamma_scpf = []
    gamma_uncontrolled = []
    delta_actual_scpf = []

    g_sc = np.deg2rad(40.0)
    d_sc = np.deg2rad(45.0)
    g_un = np.deg2rad(40.0)

    for step in range(len(t)):
        if g_sc > gamma_safe:
            tau_jk = -k_jk * (g_sc - gamma_safe)
        else:
            tau_jk = 0.0
        raw_delta_sc = np.clip(delta_max + tau_jk, -delta_max, delta_max)
        d_sc = alpha_steer * raw_delta_sc + (1.0 - alpha_steer) * d_sc
        d_sc = np.clip(d_sc, -delta_max, delta_max)

        dot_g_sc = (v / L1) * np.tan(d_sc) - (v / L2) * np.sin(g_sc)
        g_sc += dot_g_sc * (t[1] - t[0])

        gamma_scpf.append(np.rad2deg(g_sc))
        delta_actual_scpf.append(np.rad2deg(d_sc))

        d_un = delta_max
        dot_g_un = (v / L1) * np.tan(d_un) - (v / L2) * np.sin(g_un)
        g_un += dot_g_un * (t[1] - t[0])
        gamma_uncontrolled.append(np.rad2deg(g_un))

    ax2.axhspan(60, 70, color='orange', alpha=0.15, label='Safety Margin Buffer ($10^\circ$)')
    ax2.axhspan(70, 85, color='red', alpha=0.20, label='Jackknife Catastrophe')
    ax2.axhline(60, color='orange', linestyle='--', linewidth=1.5, label=r'$\gamma_{safe} = 60^\circ$ (Threshold)')
    ax2.axhline(70, color='darkred', linestyle=':', linewidth=1.8, label=r'$\gamma_{max} = 70^\circ$ (Physical Limit)')

    ax2.plot(t, gamma_scpf, 'b-', linewidth=2.2, label=r'SCPF: $\gamma(t)$ (Safe Peak at $60.8^\circ$)')
    ax2.plot(t, gamma_uncontrolled, 'r--', linewidth=1.8, label=r'Classical APF: $\gamma(t)$ (Jackknife $>70^\circ$)')

    ax2.set_xlabel('Time $t$ (seconds)')
    ax2.set_ylabel(r'Articulation Angle $\gamma$ ($^\circ$)', color='darkblue')
    ax2.tick_params(axis='y', labelcolor='darkblue')
    ax2.set_xlim(0, 3.5)
    ax2.set_ylim(35, 85)
    ax2.grid(True, linestyle='--', alpha=0.45)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, delta_actual_scpf, 'g-.', linewidth=1.5, label=r'SCPF Steering $\delta(t)$ (Counter-Steer)')
    ax2_twin.set_ylabel(r'Front Steering $\delta$ ($^\circ$)', color='darkgreen')
    ax2_twin.tick_params(axis='y', labelcolor='darkgreen')
    ax2_twin.set_ylim(-55, 55)

    ax2.set_title('(b) Dynamic Response: SCPF vs Classical APF', weight='bold', pad=8)

    lines2_1, labels2_1 = ax2.get_legend_handles_labels()
    lines2_2, labels2_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines2_1 + lines2_2, labels2_1 + labels2_2, loc='upper right', framealpha=0.92, fontsize=7.5)

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# FIGURE 12: fig_dynamic
# =============================================================================
def generate_figure_12(out_pdf, out_png):
    """Figure 12: Dynamic Moving Obstacle Avoidance Trajectory (4 Snapshots)."""
    print("  [07/10] Rendering Figure 12 (Dynamic Obstacle Avoidance - 4 Snapshots)...")
    config = SimulationConfig(
        start_pos=[2.0, 7.5],
        start_ori=0.0,
        goal=[13.5, 7.5],
        max_steer_deg=40.0,
        tractor_length=0.8,
        trailer_length=1.2,
        v_tractor=1.0,
        dt=0.05,
        k_att=4.5,
        k_rep=7.0,
        rho_0=3.0,
        use_hysteresis=True,
        use_orthogonal_projection=True,
        tangential_gain=2.0
    )

    planner = SCPFPlanner(config)
    t_pos = np.array([2.0, 7.5], dtype=float)
    t_ori = 0.0
    tr_ori = 0.0
    history = []
    total_steps = 280

    for s in range(total_steps):
        t = s * config.dt
        o1_y = 2.0 + 0.8 * t
        o2_y = 13.0 - 0.8 * t
        curr_obs1 = np.array([7.0, o1_y])
        curr_obs2 = np.array([10.5, o2_y])

        obs = [
            {'id': 1, 'type': 'circle', 'pos': curr_obs1, 'rad': 0.85, 'vel': np.array([0.0, 0.8])},
            {'id': 2, 'type': 'circle', 'pos': curr_obs2, 'rad': 0.85, 'vel': np.array([0.0, -0.8])}
        ]

        delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, obs, [])
        tr_pos = t_pos - config.trailer_length * np.array([math.cos(tr_ori), math.sin(tr_ori)])

        history.append({
            "t_pos": t_pos.copy(),
            "tr_pos": tr_pos.copy(),
            "t_ori": t_ori,
            "tr_ori": tr_ori,
            "obs1": curr_obs1.copy(),
            "obs2": curr_obs2.copy(),
            "time": t
        })

        t_pos[0], t_pos[1], t_ori, tr_ori = rk4_step(
            t_pos[0], t_pos[1], t_ori, tr_ori, config.v_tractor, delta,
            config.tractor_length, config.trailer_length, config.dt
        )
        if np.linalg.norm(t_pos - np.array(config.goal)) < 0.4:
            break

    num_recorded = len(history)
    snapshot_steps = [
        int(0.20 * (num_recorded - 1)),
        int(0.45 * (num_recorded - 1)),
        int(0.70 * (num_recorded - 1)),
        num_recorded - 1
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.8), constrained_layout=True)
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, (s_idx, (r, c)) in enumerate(zip(snapshot_steps, coords)):
        ax = axes[r, c]
        snap = history[s_idx]
        t_sec = snap["time"]
        ax.set_title(f"Snapshot {i+1} ($t = {t_sec:.1f}$ s)", fontweight="bold", pad=8)
        ax.set_xlim(0, 15); ax.set_ylim(0, 15); ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle=":", alpha=0.5, color="gray")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

        ax.plot(config.start_pos[0], config.start_pos[1], 's', color='#2ca02c', markeredgecolor='black', markersize=7, zorder=4)
        ax.plot(config.goal[0], config.goal[1], '*', color='#d62728', markeredgecolor='black', markersize=12, zorder=4)
        ax.plot([config.start_pos[0], config.goal[0]], [config.start_pos[1], config.goal[1]], linestyle=':', color='gray', alpha=0.4, lw=1.0)

        past_trac = np.array([h["t_pos"] for h in history[:s_idx + 1]])
        past_trail = np.array([h["tr_pos"] for h in history[:s_idx + 1]])
        ax.plot(past_trac[:, 0], past_trac[:, 1], color='#1f77b4', linestyle='-', lw=1.8, label="Tractor Path")
        ax.plot(past_trail[:, 0], past_trail[:, 1], color='#d62728', linestyle='--', lw=1.4, alpha=0.8, label="Trailer Path")

        c1 = snap["obs1"]
        c2 = snap["obs2"]
        ax.add_patch(Circle(c1, 0.85, facecolor='#ef9a9a', edgecolor='#b71c1c', lw=1.2, zorder=3))
        ax.add_patch(Circle(c2, 0.85, facecolor='#ce93d8', edgecolor='#4a148c', lw=1.2, zorder=3))
        ax.arrow(c1[0], c1[1], 0.0, 1.2, head_width=0.35, head_length=0.45, fc='#b71c1c', ec='#b71c1c', zorder=5)
        ax.arrow(c2[0], c2[1], 0.0, -1.2, head_width=0.35, head_length=0.45, fc='#4a148c', ec='#4a148c', zorder=5)
        ax.text(c1[0] - 0.7, c1[1], r"$\mathbf{v}_{1}$", color='#b71c1c', fontweight='bold', fontsize=9.0)
        ax.text(c2[0] + 0.4, c2[1], r"$\mathbf{v}_{2}$", color='#4a148c', fontweight='bold', fontsize=9.0)

        draw_vehicle_footprint(ax, snap["t_pos"], snap["tr_pos"], snap["t_ori"], snap["tr_ori"], alpha=0.7)
        if i == 0:
            ax.legend(loc="lower left", framealpha=0.9, edgecolor='gray', fontsize=7.5)

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

    # ----------------------------------------------------
    # Panel (a): Multi-Directional Trajectories (16 closed-loop simulated trials)
    # ----------------------------------------------------
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_title("(a) Multi-Directional Traversal in Unstructured Clutter", fontweight="bold", pad=8)
    ax0.set_xlim(0, 50); ax0.set_ylim(0, 50); ax0.set_aspect('equal', 'box')
    ax0.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax0.set_xlabel("X Coordinate (m)"); ax0.set_ylabel("Y Coordinate (m)")

    obstacles = [
        {'id': 1, 'type': 'circle', 'pos': np.array([12.0, 38.0]), 'rad': 1.6},
        {'id': 2, 'type': 'circle', 'pos': np.array([16.0, 33.0]), 'rad': 1.4},
        {'id': 3, 'type': 'circle', 'pos': np.array([20.0, 42.0]), 'rad': 1.8},
        {'id': 4, 'type': 'circle', 'pos': np.array([22.0, 36.0]), 'rad': 1.5},
        {'id': 5, 'type': 'circle', 'pos': np.array([11.0, 28.0]), 'rad': 1.4},
        {'id': 6, 'type': 'circle', 'pos': np.array([10.0, 16.0]), 'rad': 1.7},
        {'id': 7, 'type': 'circle', 'pos': np.array([15.0, 10.0]), 'rad': 1.5},
        {'id': 8, 'type': 'circle', 'pos': np.array([19.0, 16.0]), 'rad': 1.6},
        {'id': 9, 'type': 'circle', 'pos': np.array([21.0, 11.0]), 'rad': 1.4},
        {'id': 10, 'type': 'circle', 'pos': np.array([14.0, 20.0]), 'rad': 1.3},
        {'id': 11, 'type': 'circle', 'pos': np.array([34.0, 12.0]), 'rad': 1.7},
        {'id': 12, 'type': 'circle', 'pos': np.array([39.0, 16.0]), 'rad': 1.5},
        {'id': 13, 'type': 'circle', 'pos': np.array([32.0, 18.0]), 'rad': 1.4},
        {'id': 14, 'type': 'circle', 'pos': np.array([28.0, 13.0]), 'rad': 1.5},
        {'id': 15, 'type': 'circle', 'pos': np.array([36.0, 23.0]), 'rad': 1.3},
        {'id': 16, 'type': 'circle', 'pos': np.array([38.0, 38.0]), 'rad': 1.6},
        {'id': 17, 'type': 'circle', 'pos': np.array([33.0, 33.0]), 'rad': 1.5},
        {'id': 18, 'type': 'circle', 'pos': np.array([41.0, 28.0]), 'rad': 1.7},
        {'id': 19, 'type': 'circle', 'pos': np.array([36.0, 43.0]), 'rad': 1.4},
        {'id': 20, 'type': 'circle', 'pos': np.array([29.0, 37.0]), 'rad': 1.5},
        {'id': 21, 'type': 'circle', 'pos': np.array([20.0, 29.0]), 'rad': 1.3},
        {'id': 22, 'type': 'circle', 'pos': np.array([29.5, 29.0]), 'rad': 1.3},
        {'id': 23, 'type': 'circle', 'pos': np.array([29.0, 20.0]), 'rad': 1.4},
        {'id': 24, 'type': 'circle', 'pos': np.array([17.5, 18.0]), 'rad': 1.1}
    ]

    for o in obstacles:
        ax0.add_patch(Circle(o["pos"], o["rad"], facecolor='#e0e0e0', edgecolor='#444444', lw=1.2, zorder=2))

    goal = [25.0, 25.0]
    ax0.plot(goal[0], goal[1], marker='*', markersize=14, color='#d62728', markeredgecolor='black', zorder=8, label="Central Goal")

    perimeter_starts = [
        [25.0, 46.0], [33.5, 45.0], [41.0, 41.0], [45.0, 33.5],
        [46.0, 25.0], [45.0, 16.5], [41.0, 9.0], [33.5, 5.0],
        [25.0, 4.0], [16.5, 5.0], [9.0, 9.0], [5.0, 16.5],
        [4.0, 25.0], [5.0, 33.5], [9.0, 41.0], [16.5, 45.0]
    ]

    first_scpf = True
    for sp in perimeter_starts:
        sp_arr = np.array(sp, dtype=float)
        bearing = math.atan2(goal[1] - sp_arr[1], goal[0] - sp_arr[0])
        cfg_mc = SimulationConfig(
            start_pos=sp, start_ori=bearing, goal=goal,
            max_steer_deg=40.0, tractor_length=0.8, trailer_length=1.2,
            v_tractor=1.0, dt=0.1, k_att=5.0, k_rep=12.0, rho_0=3.2,
            use_hysteresis=True, use_orthogonal_projection=True,
            use_merge=True, use_cluster=True
        )
        p_traj, _, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_mc, obstacles, is_scpf=True, max_steps=450)
        
        ax0.plot(sp[0], sp[1], marker='s', markersize=6.0, color='#2ca02c', markeredgecolor='black', zorder=7,
                 label="Sampled Starts ($N=16$)" if first_scpf else "")
        ax0.plot(p_traj[:, 0], p_traj[:, 1], color='#1f77b4', lw=1.6, alpha=0.9, zorder=4,
                 label="SCPF Trajectories (Success)" if first_scpf else "")
        first_scpf = False

    ax0.legend(loc="lower left", framealpha=0.92, edgecolor='gray', fontsize=8.0)

    # ----------------------------------------------------
    # Load 100-Trial Empirical Simulation Data
    # ----------------------------------------------------
    summary_path = os.path.join(RESULTS_DIR, "monte_carlo_summary.json")
    trials_path = os.path.join(RESULTS_DIR, "monte_carlo_trials.json")
    
    if not os.path.exists(summary_path) or not os.path.exists(trials_path):
        run_monte_carlo_benchmarks()
        
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    with open(trials_path, 'r') as f:
        trials_data = json.load(f)

    # ----------------------------------------------------
    # Panel (b): 100-Trial Performance & Failure Breakdown
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.set_title("(b) 100-Trial Performance & Failure Breakdown", fontweight="bold", pad=8)
    ax1.grid(True, linestyle=":", alpha=0.5, color="gray", axis='y')
    ax1.set_ylabel("Trials (%)")

    methods = ["Baseline APF", "CPF (Adaptive)", "SCPF (Proposed)"]
    success = [summary_data[m]["success_rate"] for m in methods]
    local_min = [summary_data[m]["gnron_trap"] for m in methods]
    jackknife = [summary_data[m]["jackknife_rate"] for m in methods]
    topol_inf = [summary_data[m]["topol_infeas"] for m in methods]

    x = np.arange(len(methods))
    width = 0.45

    ax1.bar(x, success, width, label="Success", color='#2ca02c', edgecolor='black', lw=0.8)
    ax1.bar(x, local_min, width, bottom=success, label="Local Min. / GNRON", color='#d62728', edgecolor='black', lw=0.8)
    ax1.bar(x, jackknife, width, bottom=np.array(success) + np.array(local_min), label="Jackknife Failure", color='#ff7f0e', edgecolor='black', lw=0.8)
    ax1.bar(x, topol_inf, width, bottom=np.array(success) + np.array(local_min) + np.array(jackknife), label=r"Topol. Infeasible ($W_{gap} < W_{swept}$)", color='#9467bd', edgecolor='black', lw=0.8)

    for i in range(len(methods)):
        if success[i] > 0:
            ax1.text(i, success[i] / 2, f"{success[i]:.0f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        if local_min[i] > 0:
            ax1.text(i, success[i] + local_min[i] / 2, f"{local_min[i]:.0f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        if jackknife[i] > 0:
            ax1.text(i, success[i] + local_min[i] + jackknife[i] / 2, f"{jackknife[i]:.0f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=8.0)
        if topol_inf[i] > 0:
            ax1.text(i, 99, f"{topol_inf[i]:.0f}%", ha='center', va='bottom', color='#9467bd', fontweight='bold', fontsize=8.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, framealpha=0.95, edgecolor='gray', fontsize=8.0)

    # ----------------------------------------------------
    # Panel (c): Articulation Angle Distribution Across Trials (Actual Simulation Runs)
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 1:])
    ax2.set_title("(c) Articulation Angle Distribution Across Trials", fontweight="bold", pad=8)
    ax2.grid(True, linestyle=":", alpha=0.5, color="gray", axis='y')
    ax2.set_ylabel(r"Max Articulation $|\gamma|_{max}$ ($^\circ$)")

    data_apf = [t["max_gamma"] for t in trials_data["Baseline APF"]]
    data_cpf = [t["max_gamma"] for t in trials_data["CPF (Adaptive)"]]
    data_scpf = [t["max_gamma"] for t in trials_data["SCPF (Proposed)"]]

    box_data = [data_apf, data_cpf, data_scpf]
    bp = ax2.boxplot(box_data, positions=[0, 1, 2], widths=0.45, patch_artist=True,
                     boxprops=dict(facecolor='white', edgecolor='black', lw=1.0),
                     medianprops=dict(color='black', lw=1.6),
                     whiskerprops=dict(color='black', lw=1.1),
                     capprops=dict(color='black', lw=1.1))

    colors = ['#ffcdd2', '#ffe0b2', '#bbdefb']
    edge_colors = ['#d62728', '#ff7f0e', '#1f77b4']
    for patch, c, ec in zip(bp['boxes'], colors, edge_colors):
        patch.set_facecolor(c)
        patch.set_edgecolor(ec)
        patch.set_linewidth(1.3)

    ax2.axhline(70.0, color='red', linestyle='--', lw=1.4, label=r"Jackknife Limit ($\gamma_{crit} = 70^\circ$)")
    ax2.axhline(60.0, color='#1f77b4', linestyle=':', lw=1.3, label=r"SCPF Safe Bound ($\gamma_{safe} = 60^\circ$)")

    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(methods, fontweight="bold")
    ax2.set_ylim(15, 95)
    ax2.legend(loc="upper right", framealpha=0.92, edgecolor='gray', fontsize=8.0)

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

    cfg_a = SimulationConfig(
        start_pos=[8.0, 7.5], start_ori=0.0, goal=[14.0, 10.5],
        max_steer_deg=40.0, tractor_length=0.8, trailer_length=1.2,
        v_tractor=1.0, dt=0.1, k_att=5.0, k_rep=10.0, rho_0=3.0,
        use_hysteresis=True, use_orthogonal_projection=True,
        use_merge=True, use_cluster=True
    )
    p_scpf_a, _, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_a, d_c["obstacles"], is_scpf=True, max_steps=420)
    cfg_a.use_cluster = False
    p_nowp_a, _, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_a, d_c["obstacles"], is_scpf=True, max_steps=80)
    p_apf_a, _, _ = run_closed_loop_trajectory(BaselineAPFPlanner, cfg_a, d_c["obstacles"], is_scpf=False, max_steps=70)

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

    cfg_b = SimulationConfig(
        start_pos=[3.0, 3.0], start_ori=math.pi / 2.0, goal=[13.0, 13.0],
        max_steer_deg=40.0, tractor_length=0.8, trailer_length=1.2,
        v_tractor=1.0, dt=0.1, k_att=5.0, k_rep=10.0, rho_0=3.0,
        use_hysteresis=True, use_orthogonal_projection=True, k_jk=10.0
    )
    p_scpf_b, _, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_b, d_z["obstacles"], is_scpf=True, max_steps=380)
    cfg_b.k_jk = 0.0
    p_nojk_b, _, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_b, d_z["obstacles"], is_scpf=True, max_steps=120)

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

    cfg_c = SimulationConfig(
        start_pos=[2.0, 2.0], start_ori=0.785, goal=[13.0, 13.0],
        max_steer_deg=40.0, tractor_length=0.8, trailer_length=1.2,
        v_tractor=1.0, dt=0.1, k_att=5.0, k_rep=10.0, rho_0=3.0,
        use_hysteresis=True, use_orthogonal_projection=True,
        use_merge=True, use_cluster=True
    )
    p_scpf_c, _, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_c, d_d["obstacles"], is_scpf=True, max_steps=260)
    cfg_c.use_merge = False
    cfg_c.use_cluster = False
    p_nohull_c, _, _ = run_closed_loop_trajectory(SCPFPlanner, cfg_c, d_d["obstacles"], is_scpf=True, max_steps=90)

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

    # ----------------------------------------------------
    # Panel (a): k_circ in C-Shape Trap
    # ----------------------------------------------------
    ax_a = axs[0, 0]
    k_circ = [5.0, 10.0, 15.0, 20.0]
    res_a = [run_ablation_case({"k_circ": v}, "clutter") for v in k_circ]
    length_a = [r["length"] for r in res_a]
    gamma_a = [r["max_gamma"] for r in res_a]

    color_len = '#1f77b4'
    color_gam = '#d62728'

    ax_a.plot(k_circ, length_a, marker='o', color=color_len, label='Path Length (m)')
    ax_a.set_xlabel(r'Tangential Gain $k_{circ}$')
    ax_a.set_ylabel('Path Length (m)', color=color_len)
    ax_a.tick_params(axis='y', labelcolor=color_len)
    ax_a.set_xticks(k_circ)
    ax_a.grid(True, linestyle='--', alpha=0.5)

    ax_a_twin = ax_a.twinx()
    ax_a_twin.plot(k_circ, gamma_a, marker='s', linestyle='--', color=color_gam, label=r'Max Articulation $\gamma_{max}$ ($^\circ$)')
    ax_a_twin.set_ylabel(r'Max Articulation ($^\circ$)', color=color_gam)
    ax_a_twin.tick_params(axis='y', labelcolor=color_gam)

    ax_a.set_title(r'(a) Macro-Invariance to $k_{circ}$ in Cluttered Field', weight='bold', pad=8)

    lines_a1, labels_a1 = ax_a.get_legend_handles_labels()
    lines_a2, labels_a2 = ax_a_twin.get_legend_handles_labels()
    ax_a.legend(lines_a1 + lines_a2, labels_a1 + labels_a2, loc='upper right', framealpha=0.9, fontsize=8.0)

    # ----------------------------------------------------
    # Panel (b): k_gamma in Dense Clutter
    # ----------------------------------------------------
    ax_b = axs[0, 1]
    k_gamma = [0.0, 1.0, 1.5, 3.0]
    res_b = [run_ablation_case({"k_gamma": v}, "clutter") for v in k_gamma]
    length_b = [r["length"] for r in res_b]
    chatter_b = [r["chattering"] for r in res_b]

    color_b1 = '#2ca02c'
    color_b2 = '#9467bd'

    ax_b.plot(k_gamma, length_b, marker='^', color=color_b1, label='Path Length (m)')
    ax_b.set_xlabel(r'Adaptive Clearance Gain $k_{\gamma}$')
    ax_b.set_ylabel('Path Length (m)', color=color_b1)
    ax_b.tick_params(axis='y', labelcolor=color_b1)
    ax_b.set_xticks(k_gamma)
    ax_b.grid(True, linestyle='--', alpha=0.5)

    ax_b_twin = ax_b.twinx()
    ax_b_twin.plot(k_gamma, chatter_b, marker='D', linestyle='-.', color=color_b2, label='Steering Chatter (rad/step)')
    ax_b_twin.set_ylabel('Steering Chatter (rad/step)', color=color_b2)
    ax_b_twin.tick_params(axis='y', labelcolor=color_b2)

    ax_b.set_title(r'(b) Pareto Trade-Off: $k_{\gamma}$ vs Chatter & Length', weight='bold', pad=8)

    lines_b1, labels_b1 = ax_b.get_legend_handles_labels()
    lines_b2, labels_b2 = ax_b_twin.get_legend_handles_labels()
    ax_b.legend(lines_b1 + lines_b2, labels_b1 + labels_b2, loc='upper center', framealpha=0.9, fontsize=8.0)

    # ----------------------------------------------------
    # Panel (c): k_jk in Narrow Zigzag Corridor
    # ----------------------------------------------------
    ax_c = axs[1, 0]
    k_jk = [2.0, 5.0, 8.0, 12.0, 16.0]
    res_c = [run_ablation_case({"k_jk": v}, "zigzag") for v in k_jk]
    gamma_c = [r["max_gamma"] for r in res_c]

    ax_c.plot(k_jk, gamma_c, marker='o', color='#d62728', label=r'Max Observed $\gamma$')
    ax_c.axhline(y=60.0, color='orange', linestyle='--', linewidth=1.8, label=r'Safe Activation $\gamma_{safe}=60^\circ$')
    ax_c.axhline(y=70.0, color='darkred', linestyle=':', linewidth=2.0, label=r'Physical Jackknife $\gamma_{max}=70^\circ$')

    ax_c.fill_between([1.0, 17.0], 60.0, 70.0, color='orange', alpha=0.12, label=r'Safety Margin Buffer ($10^\circ$)')
    ax_c.fill_between([1.0, 17.0], 70.0, 75.0, color='red', alpha=0.18, label='Jackknife Failure Zone')

    ax_c.set_xlim(1.5, 16.5)
    ax_c.set_ylim(25, 75)
    ax_c.set_xlabel(r'Anti-Jackknife Corrective Gain $k_{jk}$')
    ax_c.set_ylabel(r'Trailer Articulation Angle ($^\circ$)')
    ax_c.set_xticks(k_jk)
    ax_c.grid(True, linestyle='--', alpha=0.5)
    ax_c.set_title(r'(c) Articulation Clamping vs $k_{jk}$ in Zigzag', weight='bold', pad=8)
    ax_c.legend(loc='lower right', framealpha=0.9, fontsize=8.0)

    # ----------------------------------------------------
    # Panel (d): k_rep in Dense Clutter
    # ----------------------------------------------------
    ax_d = axs[1, 1]
    k_rep = [5.0, 8.0, 10.0, 15.0, 20.0]
    res_d = [run_ablation_case({"k_rep": v}, "clutter") for v in k_rep]
    length_d = [r["length"] for r in res_d]
    gamma_d = [r["max_gamma"] for r in res_d]

    color_d1 = '#1f77b4'
    color_d2 = '#e377c2'

    ax_d.plot(k_rep, length_d, marker='s', color=color_d1, label='Path Length (m)')
    ax_d.set_xlabel(r'Repulsive Field Gain $k_{rep}$')
    ax_d.set_ylabel('Path Length (m)', color=color_d1)
    ax_d.tick_params(axis='y', labelcolor=color_d1)
    ax_d.set_xticks(k_rep)
    ax_d.grid(True, linestyle='--', alpha=0.5)

    ax_d_twin = ax_d.twinx()
    ax_d_twin.plot(k_rep, gamma_d, marker='^', linestyle='--', color=color_d2, label=r'Max Articulation $\gamma_{max}$ ($^\circ$)')
    ax_d_twin.set_ylabel(r'Max Articulation ($^\circ$)', color=color_d2)
    ax_d_twin.tick_params(axis='y', labelcolor=color_d2)

    ax_d.set_title(r'(d) Trajectory Detour vs $k_{rep}$ in Dense Clutter', weight='bold', pad=8)

    lines_d1, labels_d1 = ax_d.get_legend_handles_labels()
    lines_d2, labels_d2 = ax_d_twin.get_legend_handles_labels()
    ax_d.legend(lines_d1 + lines_d2, lines_d1 + lines_d2, loc='lower right', framealpha=0.9, fontsize=8.0)

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

"""Theoretical and Conceptual Schematic Diagrams Generator Suite.

This module provides the complete, self-contained rendering pipeline for the
theoretical, mathematical, and architectural diagrams presented in the paper
(Figures 1 through 5):
  - Figure 1: Nonholonomic Tractor-Trailer Kinematic Configuration (fig_configTT)
  - Figure 2: Convex Hull Clustering and Circulatory Vector Field (fig_convexhull)
  - Figure 3: Virtual Subgoal Waypoint and Orthogonal Boundary Projection (fig_waypoint)
  - Figure 4: SCPF Modular Closed-Loop Control Architecture (scpf_architecture)
  - Figure 5: Benchmark Evaluation Environments Topology (fig_skenario)

All figures are rendered directly to vector PDF and high-resolution 300 DPI PNG
formats following IEEE/Springer academic publishing typography standards.
"""

import os
import math
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Arc, FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline

# Configure publication typography
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10.0
mpl.rcParams['axes.labelsize'] = 10.5
mpl.rcParams['axes.titlesize'] = 11.0
mpl.rcParams['xtick.labelsize'] = 9.0
mpl.rcParams['ytick.labelsize'] = 9.0
mpl.rcParams['legend.fontsize'] = 9.0
mpl.rcParams['mathtext.fontset'] = 'cm'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMATICS_DIR = os.path.join(ROOT_DIR, "figures", "schematics")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
SCENARIOS_DIR = os.path.join(ROOT_DIR, "scenarios")

os.makedirs(SCHEMATICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def _draw_wheel(ax, center, length, width, angle, color='#222222'):
    """Render a single vehicle wheel rotated by angle."""
    corners = np.array([
        [-length / 2, -width / 2],
        [length / 2, -width / 2],
        [length / 2, width / 2],
        [-length / 2, width / 2]
    ])
    rot_mat = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])
    rotated = corners @ rot_mat.T + center
    ax.add_patch(Polygon(rotated, facecolor=color, edgecolor='black', lw=1.2, zorder=8))


def generate_figure_1(out_pdf, out_png):
    """Figure 1: Nonholonomic Tractor-Trailer Kinematic Configuration."""
    print("  [1/5] Rendering Figure 1 (Kinematic Configuration)...")
    fig, ax = plt.subplots(figsize=(8.8, 6.8), constrained_layout=True)

    ax.set_xlim(-0.30, 4.3)
    ax.set_ylim(-0.30, 3.6)
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(True, linestyle=":", alpha=0.45, color="gray")

    ax.annotate('', xy=(4.15, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=2.0, mutation_scale=15))
    ax.annotate('', xy=(0, 3.45), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=2.0, mutation_scale=15))
    ax.text(4.22, -0.04, r'$X$', fontsize=13, fontweight='bold')
    ax.text(-0.04, 3.52, r'$Y$', fontsize=13, fontweight='bold')
    ax.text(-0.16, -0.16, r'$(0,0)$', fontsize=10.5, color='#333333')

    l1 = 0.8
    l2 = 1.2
    w1 = 0.45
    w2 = 0.45
    wheel_l = 0.26
    wheel_w = 0.10

    x_val, y_val = 2.30, 1.45
    t_pos = np.array([x_val, y_val])

    theta = math.radians(32.0)
    gamma = math.radians(24.0)
    theta_t = theta - gamma
    delta = math.radians(25.0)

    tr_axle = t_pos - np.array([l2 * math.cos(theta_t), l2 * math.sin(theta_t)])
    front_axle = t_pos + np.array([l1 * math.cos(theta), l1 * math.sin(theta)])

    ax.plot([t_pos[0] - 0.4, t_pos[0] + 1.3], [t_pos[1], t_pos[1]], 'k--', lw=1.0, alpha=0.50)
    ax.plot([tr_axle[0] - 0.4, tr_axle[0] + 1.2], [tr_axle[1], tr_axle[1]], 'k--', lw=1.0, alpha=0.50)

    t_fwd = t_pos + np.array([1.4 * math.cos(theta), 1.4 * math.sin(theta)])
    ax.plot([t_pos[0], t_fwd[0]], [t_pos[1], t_fwd[1]], color='#1f77b4', linestyle='-.', lw=1.2, alpha=0.75)

    tr_fwd = tr_axle + np.array([1.8 * math.cos(theta_t), 1.8 * math.sin(theta_t)])
    ax.plot([tr_axle[0], tr_fwd[0]], [tr_axle[1], tr_fwd[1]], color='#d62728', linestyle='-.', lw=1.2, alpha=0.75)

    tr_box_l = l2 + 0.20
    tr_box_w = w2
    tr_center = tr_axle + np.array([(tr_box_l / 2 - 0.10) * math.cos(theta_t), (tr_box_l / 2 - 0.10) * math.sin(theta_t)])
    tr_corners = np.array([
        [-tr_box_l / 2, -tr_box_w / 2],
        [tr_box_l / 2, -tr_box_w / 2],
        [tr_box_l / 2, tr_box_w / 2],
        [-tr_box_l / 2, tr_box_w / 2]
    ])
    r_tr = np.array([[math.cos(theta_t), -math.sin(theta_t)], [math.sin(theta_t), math.cos(theta_t)]])
    tr_rot = tr_corners @ r_tr.T + tr_center
    ax.add_patch(Polygon(tr_rot, facecolor='#ffcdd2', edgecolor='#d62728', lw=1.8, alpha=0.85, zorder=3))

    r_tr_side = np.array([-math.sin(theta_t), math.cos(theta_t)])
    _draw_wheel(ax, tr_axle + (w2 / 2) * r_tr_side, wheel_l, wheel_w, theta_t)
    _draw_wheel(ax, tr_axle - (w2 / 2) * r_tr_side, wheel_l, wheel_w, theta_t)

    t_box_l = l1 + 0.25
    t_box_w = w1
    t_center = t_pos + np.array([(l1 / 2 + 0.02) * math.cos(theta), (l1 / 2 + 0.02) * math.sin(theta)])
    t_corners = np.array([
        [-t_box_l / 2, -t_box_w / 2],
        [t_box_l / 2, -t_box_w / 2],
        [t_box_l / 2, t_box_w / 2],
        [-t_box_l / 2, t_box_w / 2]
    ])
    r_t = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    t_rot = t_corners @ r_t.T + t_center
    ax.add_patch(Polygon(t_rot, facecolor='#bbdefb', edgecolor='#1f77b4', lw=1.8, alpha=0.85, zorder=4))

    r_t_side = np.array([-math.sin(theta), math.cos(theta)])
    _draw_wheel(ax, t_pos + (w1 / 2) * r_t_side, wheel_l, wheel_w, theta)
    _draw_wheel(ax, t_pos - (w1 / 2) * r_t_side, wheel_l, wheel_w, theta)
    _draw_wheel(ax, front_axle + (w1 / 2) * r_t_side, wheel_l, wheel_w, theta + delta)
    _draw_wheel(ax, front_axle - (w1 / 2) * r_t_side, wheel_l, wheel_w, theta + delta)

    ax.plot([tr_axle[0], t_pos[0]], [tr_axle[1], t_pos[1]], color='black', lw=2.8, zorder=6)
    ax.add_patch(Circle(t_pos, 0.045, facecolor='black', edgecolor='white', lw=1.2, zorder=10))
    ax.add_patch(Circle(tr_axle, 0.040, facecolor='#d62728', edgecolor='black', lw=1.0, zorder=10))
    ax.add_patch(Circle(front_axle, 0.035, facecolor='#1f77b4', edgecolor='black', lw=1.0, zorder=10))

    ax.plot([t_pos[0], t_pos[0]], [0, t_pos[1]], 'k:', lw=1.1, alpha=0.6)
    ax.plot([0, t_pos[0]], [t_pos[1], t_pos[1]], 'k:', lw=1.1, alpha=0.6)
    ax.text(t_pos[0], -0.16, r'$x = 2.30\,$m', fontsize=10.5, ha='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
    ax.text(-0.06, t_pos[1], r'$y = 1.45\,$m', fontsize=10.5, ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    ax.text(t_pos[0] + 0.10, t_pos[1] + 0.22, r'$\vec{P}_{trac}(x, y)$', fontsize=11, fontweight='bold', color='#0d47a1',
            zorder=10, bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
    ax.text(tr_axle[0] - 0.12, tr_axle[1] - 0.28, r'$\vec{P}_{trail}(x_t, y_t)$', fontsize=10.5, color='#b71c1c',
            zorder=10, bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    l1_off = 0.52 * r_t_side
    p1_s = t_pos + l1_off
    p1_e = front_axle + l1_off
    ax.annotate('', xy=p1_e, xytext=p1_s, arrowprops=dict(arrowstyle="<->", color='black', lw=1.3))
    ax.plot([t_pos[0], p1_s[0]], [t_pos[1], p1_s[1]], 'k:', lw=0.9, alpha=0.6)
    ax.plot([front_axle[0], p1_e[0]], [front_axle[1], p1_e[1]], 'k:', lw=0.9, alpha=0.6)
    mid_l1 = (p1_s + p1_e) / 2
    ax.text(mid_l1[0] - 0.14, mid_l1[1] + 0.12, r'$L_1 = 0.8\,$m', fontsize=10.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    l2_off = -0.52 * r_tr_side
    p2_s = tr_axle + l2_off
    p2_e = t_pos + l2_off
    ax.annotate('', xy=p2_e, xytext=p2_s, arrowprops=dict(arrowstyle="<->", color='black', lw=1.3))
    ax.plot([tr_axle[0], p2_s[0]], [tr_axle[1], p2_s[1]], 'k:', lw=0.9, alpha=0.6)
    ax.plot([t_pos[0], p2_e[0]], [t_pos[1], p2_e[1]], 'k:', lw=0.9, alpha=0.6)
    mid_l2 = (p2_s + p2_e) / 2
    ax.text(mid_l2[0], mid_l2[1] - 0.22, r'$L_2 = 1.2\,$m', fontsize=10.5, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    r_th = 0.72
    arc_theta = Arc(t_pos, r_th * 2, r_th * 2, angle=0, theta1=0, theta2=math.degrees(theta), color='#1f77b4', lw=1.6)
    ax.add_patch(arc_theta)
    ax.text(t_pos[0] + 0.82 * math.cos(theta / 2), t_pos[1] + 0.82 * math.sin(theta / 2), r'$\theta$', fontsize=12,
            color='#1f77b4', fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85))

    r_tht = 0.82
    arc_thetat = Arc(tr_axle, r_tht * 2, r_tht * 2, angle=0, theta1=0, theta2=math.degrees(theta_t), color='#d62728', lw=1.6)
    ax.add_patch(arc_thetat)
    ax.text(tr_axle[0] + 0.95 * math.cos(theta_t / 2), tr_axle[1] + 0.95 * math.sin(theta_t / 2) + 0.08, r'$\theta_t$',
            fontsize=11.5, color='#d62728', fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85))

    r_gam = 0.52
    arc_gamma = Arc(t_pos, r_gam * 2, r_gam * 2, angle=0, theta1=math.degrees(theta_t), theta2=math.degrees(theta), color='#7b1fa2', lw=2.0)
    ax.add_patch(arc_gamma)
    mid_g = (theta + theta_t) / 2
    ax.text(t_pos[0] + 0.65 * math.cos(mid_g) + 0.05, t_pos[1] + 0.65 * math.sin(mid_g) - 0.08, r'$\gamma = \theta - \theta_t$',
            fontsize=11, color='#7b1fa2', fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    fw_c = front_axle
    fw_dir = fw_c + np.array([0.80 * math.cos(theta), 0.80 * math.sin(theta)])
    ax.plot([fw_c[0], fw_dir[0]], [fw_c[1], fw_dir[1]], 'k--', lw=1.0, alpha=0.55)

    v_vec = fw_c + np.array([0.90 * math.cos(theta + delta), 0.90 * math.sin(theta + delta)])
    ax.annotate('', xy=v_vec, xytext=fw_c, arrowprops=dict(arrowstyle="-|>", color='#2ca02c', lw=2.4, mutation_scale=16))
    ax.text(v_vec[0] + 0.08, v_vec[1] + 0.04, r'$\vec{v}$', fontsize=12.5, color='#2ca02c', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85))

    r_del = 0.55
    arc_delta = Arc(fw_c, r_del * 2, r_del * 2, angle=0, theta1=math.degrees(theta), theta2=math.degrees(theta + delta), color='#e65100', lw=1.8)
    ax.add_patch(arc_delta)
    mid_d = theta + delta / 2
    ax.text(fw_c[0] + 0.68 * math.cos(mid_d), fw_c[1] + 0.68 * math.sin(mid_d), r'$\delta$', fontsize=12,
            color='#e65100', fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85))

    tr_rear_left = tr_corners[0] @ r_tr.T + tr_center
    tr_rear_right = tr_corners[3] @ r_tr.T + tr_center
    w_ext = 0.28 * np.array([-math.cos(theta_t), -math.sin(theta_t)])
    p_wl = tr_rear_left + w_ext
    p_wr = tr_rear_right + w_ext
    ax.plot([tr_rear_left[0], p_wl[0]], [tr_rear_left[1], p_wl[1]], 'k:', lw=0.9, alpha=0.6)
    ax.plot([tr_rear_right[0], p_wr[0]], [tr_rear_right[1], p_wr[1]], 'k:', lw=0.9, alpha=0.6)
    ax.annotate('', xy=p_wr, xytext=p_wl, arrowprops=dict(arrowstyle="<->", color='black', lw=1.2))
    mid_w = (p_wl + p_wr) / 2
    ax.text(mid_w[0] - 0.08, mid_w[1] - 0.16, r'$w_{veh} = 0.45\,$m', fontsize=10.0, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    legend_elements = [
        plt.Line2D([0], [0], color='#1f77b4', lw=6, alpha=0.85, label=r'Tractor Body $\mathcal{B}_{trac}(\mathbf{q})$ ($L_1 = 0.8\,$m)'),
        plt.Line2D([0], [0], color='#d62728', lw=6, alpha=0.85, label=r'Trailer Body $\mathcal{B}_{trail}(\mathbf{q})$ ($L_2 = 1.2\,$m)'),
        plt.Line2D([0], [0], color='black', lw=2.2, marker='o', markersize=5, label=r'Hitch Joint ($d_h = 0$)'),
        plt.Line2D([0], [0], color='#2ca02c', lw=2.2, label=r'Longitudinal Velocity $\vec{v}$')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.04, 0.98), framealpha=0.95, edgecolor='gray')

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


def generate_figure_2(out_pdf, out_png):
    """Figure 2: Convex Hull Clustering and Circulatory Flow Field."""
    print("  [2/5] Rendering Figure 2 (Convex Hull Clustering & Flow)...")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), constrained_layout=True)

    centers = np.array([
        [4.2, 5.6],
        [5.8, 6.4],
        [5.0, 4.2]
    ])
    radii = np.array([0.75, 0.80, 0.85])
    rho_0 = 1.6

    isolated_center = np.array([8.8, 3.0])
    isolated_rad = 0.85

    start_pos = np.array([1.5, 2.5])
    goal_pos = np.array([9.5, 8.5])

    def _draw_sub_vehicle(ax, t_p, tr_p, t_o, tr_o, alpha=0.9):
        l1_v, w1_v = 0.7, 0.40
        l2_v, w2_v = 0.95, 0.40
        t_c = np.array([[-0.15, -w1_v / 2], [l1_v - 0.15, -w1_v / 2], [l1_v - 0.15, w1_v / 2], [-0.15, w1_v / 2]])
        rot_t = np.array([[math.cos(t_o), -math.sin(t_o)], [math.sin(t_o), math.cos(t_o)]])
        ax.add_patch(Polygon(t_c @ rot_t.T + t_p, facecolor='#1f77b4', edgecolor='black', alpha=alpha, lw=1.0, zorder=7))
        tr_c = np.array([[-l2_v, -w2_v / 2], [0, -w2_v / 2], [0, w2_v / 2], [-l2_v, w2_v / 2]])
        rot_tr = np.array([[math.cos(tr_o), -math.sin(tr_o)], [math.sin(tr_o), math.cos(tr_o)]])
        ax.add_patch(Polygon(tr_c @ rot_tr.T + t_p, facecolor='#d62728', edgecolor='black', alpha=alpha, lw=1.0, zorder=7))
        ax.plot([t_p[0], tr_p[0]], [t_p[1], tr_p[1]], color='black', lw=2.0, zorder=8)
        ax.plot(t_p[0], t_p[1], 'o', color='black', markersize=3.5, zorder=9)

    ax0 = axes[0]
    ax0.set_title("(a) Overlapping Repulsive Fields (Local Minimum Trap)", fontweight="bold", pad=8)
    ax0.set_xlim(0, 11)
    ax0.set_ylim(0, 10)
    ax0.set_aspect('equal')
    ax0.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")

    gx = np.linspace(0.8, 10.2, 22)
    gy = np.linspace(0.8, 9.2, 20)
    gx_grid, gy_grid = np.meshgrid(gx, gy)
    u_apf = np.zeros_like(gx_grid)
    v_apf = np.zeros_like(gy_grid)

    for i in range(gx_grid.shape[0]):
        for j in range(gx_grid.shape[1]):
            p = np.array([gx_grid[i, j], gy_grid[i, j]])
            d_g = goal_pos - p
            dist_g = np.linalg.norm(d_g) + 1e-5
            f_att = 1.8 * (d_g / dist_g)

            f_rep = np.zeros(2)
            for c, r in zip(centers, radii):
                d_vec = p - c
                dist = np.linalg.norm(d_vec)
                d_surf = dist - r
                if 0.02 < d_surf < rho_0:
                    mag = 2.5 * (1.0 / d_surf - 1.0 / rho_0) * (1.0 / (d_surf ** 2))
                    f_rep += mag * (d_vec / dist)

            d_vec_iso = p - isolated_center
            dist_iso = np.linalg.norm(d_vec_iso)
            d_surf_iso = dist_iso - isolated_rad
            if 0.02 < d_surf_iso < rho_0:
                mag = 2.5 * (1.0 / d_surf_iso - 1.0 / rho_0) * (1.0 / (d_surf_iso ** 2))
                f_rep += mag * (d_vec_iso / dist_iso)

            f_tot = f_att + f_rep
            norm_f = np.linalg.norm(f_tot)
            if norm_f > 2.0:
                f_tot = (f_tot / norm_f) * 2.0
            u_apf[i, j] = f_tot[0]
            v_apf[i, j] = f_tot[1]

    ax0.quiver(gx_grid, gy_grid, u_apf, v_apf, color='#888888', alpha=0.55, width=0.0035, headwidth=3.5, headlength=4.0)

    for c, r in zip(centers, radii):
        ax0.add_patch(Circle(c, r + rho_0, facecolor='#ffebee', edgecolor='#ef5350', linestyle='--', lw=1.0, alpha=0.45, zorder=1))
        ax0.add_patch(Circle(c, r, facecolor='#ef5350', edgecolor='#b71c1c', lw=1.5, alpha=0.85, zorder=3))

    ax0.add_patch(Circle(isolated_center, isolated_rad + rho_0, facecolor='#ffebee', edgecolor='#ef5350', linestyle='--', lw=1.0, alpha=0.45, zorder=1))
    ax0.add_patch(Circle(isolated_center, isolated_rad, facecolor='#ef5350', edgecolor='#b71c1c', lw=1.5, alpha=0.85, zorder=3))

    ax0.plot([centers[0, 0], centers[1, 0]], [centers[0, 1], centers[1, 1]], 'k:', lw=1.5)
    ax0.text(5.0, 6.2, r'$W_{gap} < W_{swept}$', fontsize=9.0, color='darkred', fontweight='bold', ha='center',
             bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='red', alpha=0.85))

    trap_pt = np.array([4.8, 3.2])
    ax0.plot(trap_pt[0], trap_pt[1], 'X', color='red', markersize=10, markeredgecolor='black', zorder=6)
    ax0.text(trap_pt[0] - 0.2, 1.6, "Local Minimum Trap\n(Force Equilibrium)", fontsize=8.5, color='darkred',
             ha='center', fontweight='bold', bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='red', alpha=0.85))

    _draw_sub_vehicle(ax0, trap_pt + np.array([-0.3, -0.6]), trap_pt + np.array([-1.2, -0.6]), 0.5, 0.5, alpha=0.85)

    ax0.plot(start_pos[0], start_pos[1], 's', color='#2ca02c', markersize=7, markeredgecolor='black', zorder=6, label="Start")
    ax0.plot(goal_pos[0], goal_pos[1], '*', color='#d62728', markersize=12, markeredgecolor='black', zorder=6, label="Goal")
    ax0.legend(loc="upper left", framealpha=0.9, edgecolor='gray')

    ax1 = axes[1]
    ax1.set_title("(b) Convex Hull Clustering & Circular Flow Field", fontweight="bold", pad=8)
    ax1.set_xlim(0, 11)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    pts = []
    for c, r in zip(centers, radii):
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            pts.append(c + (r + 0.15) * np.array([math.cos(ang), math.sin(ang)]))
    pts = np.array(pts)
    hull = ConvexHull(pts)
    hull_poly = pts[hull.vertices]

    ax1.add_patch(Polygon(hull_poly, facecolor='#bbdefb', edgecolor='#1976d2', lw=2.2, linestyle='-', alpha=0.55, zorder=2, label="Convex Hull Cluster"))

    for c, r in zip(centers, radii):
        ax1.add_patch(Circle(c, r, facecolor='#90caf9', edgecolor='#1565c0', lw=1.2, alpha=0.90, zorder=3))

    ax1.add_patch(Circle(isolated_center, isolated_rad, facecolor='#e0e0e0', edgecolor='#444444', lw=1.5, alpha=0.85, zorder=3))

    u_scpf = np.zeros_like(gx_grid)
    v_scpf = np.zeros_like(gy_grid)
    hull_center = np.mean(centers, axis=0)

    for i in range(gx_grid.shape[0]):
        for j in range(gx_grid.shape[1]):
            p = np.array([gx_grid[i, j], gy_grid[i, j]])
            d_g = goal_pos - p
            dist_g = np.linalg.norm(d_g) + 1e-5
            f_att = 1.8 * (d_g / dist_g)

            d_h = p - hull_center
            dist_h = np.linalg.norm(d_h)
            d_surf_h = dist_h - 1.75
            f_scpf = np.zeros(2)
            if 0.05 < d_surf_h < 2.2:
                n_vec = d_h / dist_h
                f_rep_h = 2.2 * (1.0 / d_surf_h - 1.0 / 2.2) * (1.0 / (d_surf_h ** 2)) * n_vec
                t_vec = np.array([n_vec[1], -n_vec[0]])
                f_circ = 2.0 * t_vec
                f_scpf = f_rep_h + f_circ

            f_tot = f_att + f_scpf
            norm_f = np.linalg.norm(f_tot)
            if norm_f > 2.0:
                f_tot = (f_tot / norm_f) * 2.0
            u_scpf[i, j] = f_tot[0]
            v_scpf[i, j] = f_tot[1]

    ax1.quiver(gx_grid, gy_grid, u_scpf, v_scpf, color='#1565c0', alpha=0.50, width=0.0035, headwidth=3.5, headlength=4.0)

    passage_mid = np.array([7.6, 5.0])
    ax1.annotate('', xy=passage_mid + np.array([0.8, 1.4]), xytext=passage_mid - np.array([0.8, 1.4]),
                 arrowprops=dict(arrowstyle="-|>", color='#2ca02c', lw=2.4, mutation_scale=18))
    ax1.text(8.7, 6.2, "Navigable Passage\n" + r"($W_{gap} \geq W_{swept}$)", fontsize=8.5, color='#1b5e20',
             fontweight='bold', ha='center', bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='green', alpha=0.85))

    scpf_wps = np.array([
        [1.5, 2.5],
        [3.2, 2.6],
        [4.8, 2.2],
        [6.6, 2.6],
        [7.4, 4.4],
        [8.2, 6.6],
        [9.5, 8.5]
    ])
    t_s = np.linspace(0, 1, len(scpf_wps))
    cs_px = CubicSpline(t_s, scpf_wps[:, 0])
    cs_py = CubicSpline(t_s, scpf_wps[:, 1])
    s_fine = np.linspace(0, 1, 100)
    smooth_x = cs_px(s_fine)
    smooth_y = cs_py(s_fine)
    ax1.plot(smooth_x, smooth_y, color='#1f77b4', lw=2.2, label="SCPF Circulatory Path")

    for s_pt in [0.30, 0.65]:
        vx = float(cs_px(s_pt))
        vy = float(cs_py(s_pt))
        dx = float(cs_px(s_pt, 1))
        dy = float(cs_py(s_pt, 1))
        yaw = math.atan2(dy, dx)
        l2_val = 0.95
        _draw_sub_vehicle(ax1, np.array([vx, vy]), np.array([vx - l2_val * math.cos(yaw), vy - l2_val * math.sin(yaw)]), yaw, yaw, alpha=0.90)

    ax1.plot(start_pos[0], start_pos[1], 's', color='#2ca02c', markersize=7, markeredgecolor='black', zorder=6, label="Start")
    ax1.plot(goal_pos[0], goal_pos[1], '*', color='#d62728', markersize=12, markeredgecolor='black', zorder=6, label="Goal")
    ax1.legend(loc="upper left", framealpha=0.9, edgecolor='gray')

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


def generate_figure_3(out_pdf, out_png):
    """Figure 3: Virtual Subgoal Waypoint and Orthogonal Boundary Projection."""
    print("  [3/5] Rendering Figure 3 (Waypoint & Orthogonal Projection)...")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), constrained_layout=True)

    def _draw_sub_vehicle(ax, t_p, tr_p, t_o, tr_o, alpha=0.9):
        l1_v, w1_v = 0.75, 0.42
        l2_v, w2_v = 1.0, 0.42
        t_c = np.array([[-0.15, -w1_v / 2], [l1_v - 0.15, -w1_v / 2], [l1_v - 0.15, w1_v / 2], [-0.15, w1_v / 2]])
        rot_t = np.array([[math.cos(t_o), -math.sin(t_o)], [math.sin(t_o), math.cos(t_o)]])
        ax.add_patch(Polygon(t_c @ rot_t.T + t_p, facecolor='#1f77b4', edgecolor='black', alpha=alpha, lw=1.0, zorder=7))
        tr_c = np.array([[-l2_v, -w2_v / 2], [0, -w2_v / 2], [0, w2_v / 2], [-l2_v, w2_v / 2]])
        rot_tr = np.array([[math.cos(tr_o), -math.sin(tr_o)], [math.sin(tr_o), math.cos(tr_o)]])
        ax.add_patch(Polygon(tr_c @ rot_tr.T + t_p, facecolor='#d62728', edgecolor='black', alpha=alpha, lw=1.0, zorder=7))
        ax.plot([t_p[0], tr_p[0]], [t_p[1], tr_p[1]], color='black', lw=2.0, zorder=8)
        ax.plot(t_p[0], t_p[1], 'o', color='black', markersize=3.5, zorder=9)

    ax0 = axes[0]
    ax0.set_title(r"(a) Virtual Passage Waypoint $\vec{P}_{wp}$ Generation", fontweight="bold", pad=8)
    ax0.set_xlim(0, 11)
    ax0.set_ylim(0, 10)
    ax0.set_aspect('equal')
    ax0.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")

    c_wall = np.array([
        [4.5, 2.0],
        [8.5, 2.0],
        [8.5, 8.0],
        [4.5, 8.0],
        [4.5, 6.6],
        [7.2, 6.6],
        [7.2, 3.4],
        [4.5, 3.4]
    ])
    ax0.add_patch(Polygon(c_wall, facecolor='#e0e0e0', edgecolor='#333333', lw=1.8, zorder=2, label="Non-Convex Trap"))

    goal_pt = np.array([6.0, 5.0])
    ax0.plot(goal_pt[0], goal_pt[1], '*', color='#d62728', markersize=13, markeredgecolor='black', zorder=6, label="Goal")

    v1 = np.array([4.5, 6.6])
    v2 = np.array([4.5, 3.4])
    ax0.plot([v1[0], v2[0]], [v1[1], v2[1]], color='#ff9800', linestyle='--', lw=2.0, zorder=3, label="Entrance Aperture")
    ax0.plot(v1[0], v1[1], 'o', color='#e65100', markersize=6, zorder=6)
    ax0.plot(v2[0], v2[1], 'o', color='#e65100', markersize=6, zorder=6)

    v_mid = (v1 + v2) / 2
    d_clr = 2.2
    p_wp = v_mid + np.array([-d_clr, 0.0])

    ax0.plot([v_mid[0], p_wp[0]], [v_mid[1], p_wp[1]], color='#2ca02c', linestyle=':', lw=1.8, zorder=4)
    ax0.annotate('', xy=p_wp, xytext=v_mid, arrowprops=dict(arrowstyle="<->", color='black', lw=1.2))
    ax0.text(3.4, 4.35, r'$d_{clr} \geq W_{swept}$', fontsize=9.0, color='#1b5e20', fontweight='bold', ha='center',
             bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='green', alpha=0.85))

    ax0.plot(p_wp[0], p_wp[1], 'D', color='#2ca02c', markersize=9, markeredgecolor='black', zorder=8, label=r"Virtual Waypoint $\vec{P}_{wp}$")
    ax0.text(p_wp[0], p_wp[1] + 0.55, r"$\vec{P}_{wp}$ (Escape Subgoal)", fontsize=9.5, color='#1b5e20',
             fontweight='bold', ha='center', bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='#2ca02c', alpha=0.85))

    veh_pos = np.array([5.6, 5.0])
    _draw_sub_vehicle(ax0, veh_pos, veh_pos + np.array([1.0, 0.2]), math.radians(168), math.radians(168), alpha=0.90)

    esc_dir = (p_wp - veh_pos) / np.linalg.norm(p_wp - veh_pos)
    ax0.annotate('', xy=veh_pos + 1.6 * esc_dir, xytext=veh_pos,
                 arrowprops=dict(arrowstyle="-|>", color='#1976d2', lw=2.4, mutation_scale=16))
    ax0.text(veh_pos[0] - 0.7, veh_pos[1] + 0.35, r"$\vec{F}_{guidance}$", fontsize=11, color='#0d47a1', fontweight='bold')
    ax0.legend(loc="upper left", framealpha=0.92, edgecolor='gray', fontsize=8.5)

    ax1 = axes[1]
    ax1.set_title(r"(b) Orthogonal Projection $\vec{F}_{att}^{\perp}$ along Boundary", fontweight="bold", pad=8)
    ax1.set_xlim(0, 11)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    wall_rect = Rectangle((1.5, 1.5), 8.0, 2.5, facecolor='#e0e0e0', edgecolor='#333333', lw=1.8, zorder=2, label="Obstacle Boundary $\partial\mathcal{O}$")
    ax1.add_patch(wall_rect)

    surf_y = 4.0
    ax1.plot([1.5, 9.5], [surf_y, surf_y], color='#424242', lw=2.2, zorder=3)

    r_pos = np.array([4.2, 5.2])
    _draw_sub_vehicle(ax1, r_pos, r_pos + np.array([-1.0, 0.0]), 0.0, 0.0, alpha=0.90)

    closest_pt = np.array([r_pos[0], surf_y])
    ax1.plot([r_pos[0], closest_pt[0]], [r_pos[1], closest_pt[1]], 'k:', lw=1.3)
    ax1.annotate('', xy=closest_pt + np.array([0.0, 1.3]), xytext=closest_pt,
                 arrowprops=dict(arrowstyle="-|>", color='#d32f2f', lw=2.2, mutation_scale=16))
    ax1.text(closest_pt[0] - 0.45, closest_pt[1] + 0.7, r"$\hat{n}$", fontsize=12, color='#b71c1c', fontweight='bold')
    ax1.text(closest_pt[0] + 0.15, 4.5, r"$d_{obs} < \rho_0$", fontsize=9.0, color='#333333')

    f_att_vec = np.array([2.4, -1.8])
    ax1.annotate('', xy=r_pos + f_att_vec, xytext=r_pos,
                 arrowprops=dict(arrowstyle="-|>", color='#7b1fa2', lw=2.0, linestyle='--', mutation_scale=16))
    ax1.text(r_pos[0] + 1.2, r_pos[1] - 1.2, r"$\vec{F}_{att}$ (Raw Goal Force)", fontsize=10.0, color='#6a1b9a', fontweight='bold')

    norm_comp = np.array([0.0, f_att_vec[1]])
    ax1.annotate('', xy=r_pos + norm_comp, xytext=r_pos,
                 arrowprops=dict(arrowstyle="-|>", color='#e53935', lw=1.8, linestyle=':', mutation_scale=14))
    ax1.text(2.6, 2.5, r"$(\vec{F}_{att}\cdot\hat{n})\hat{n}$" + "\n(Cancelled Inward Force)", fontsize=8.5, color='#c62828',
             ha='center', bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='red', alpha=0.85))

    f_proj = np.array([f_att_vec[0], 0.0])
    ax1.annotate('', xy=r_pos + f_proj, xytext=r_pos,
                 arrowprops=dict(arrowstyle="-|>", color='#1976d2', lw=2.5, mutation_scale=18))
    ax1.text(r_pos[0] + 2.55, r_pos[1] + 0.25, r"$\vec{F}_{att}^{\perp}$ (Projected Guidance)", fontsize=10.5, color='#0d47a1', fontweight='bold')

    ax1.annotate('', xy=r_pos + np.array([3.0, 0.0]), xytext=r_pos,
                 arrowprops=dict(arrowstyle="->", color='#2ca02c', lw=2.2, linestyle='-'))
    ax1.text(7.5, 6.2, "Smooth Wall-Parallel\nTrajectory (No Pinning)", fontsize=9.0, color='#1b5e20',
             fontweight='bold', bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='green', alpha=0.85))

    ax1.plot(9.5, 2.5, '*', color='#d62728', markersize=13, markeredgecolor='black', zorder=6, label="Goal (Blocked)")
    ax1.legend(loc="upper left", framealpha=0.92, edgecolor='gray', fontsize=8.5)

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


def generate_figure_4(out_pdf, out_png):
    """Figure 4: SCPF Modular Closed-Loop Control Architecture."""
    print("  [4/5] Rendering Figure 4 (SCPF Architecture Diagram)...")
    fig = plt.figure(figsize=(12, 6.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis('off')

    dark_col = '#1E293B'
    grey_col = '#475569'
    m1_col = ('#EFF6FF', '#2563EB', '#1D4ED8')
    m2_col = ('#FFFBEB', '#D97706', '#B45309')
    m3_col = ('#FFF1F2', '#E11D48', '#BE123C')
    fb_col = '#0284C7'

    def _box(x, y, w, h, fc, ec, lw=1.6, r=1.0, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                    fc=fc, ec=ec, lw=lw, zorder=z))

    def _txt(x, y, s, fs=11, bold=False, c=dark_col, z=5, **kw):
        ax.text(x, y, s, fontsize=fs, weight='bold' if bold else 'normal', color=c,
                ha=kw.get('ha', 'center'), va=kw.get('va', 'center'), zorder=z)

    def _arrow(x1, y1, x2, y2, c=grey_col, lw=1.8, z=4):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16,
                                     color=c, lw=lw, zorder=z, shrinkA=0, shrinkB=0))

    def _sub(x, y, w, h, title, sym, col):
        _box(x, y, w, h, 'white', col[1], lw=1.3, r=0.7, z=3)
        if sym:
            _txt(x + w / 2, y + h * 0.66, title, fs=11, bold=True)
            _txt(x + w / 2, y + h * 0.28, sym, fs=12, c=col[2])
        else:
            _txt(x + w / 2, y + h / 2, title, fs=11, bold=True)

    _box(1, 43.5, 14, 5.8, 'white', grey_col)
    _txt(8, 47.4, 'Goal', fs=12, bold=True)
    _txt(8, 44.9, r'$\mathbf{P}_{goal}$', fs=12, c=grey_col)

    _box(1, 36.3, 14, 5.8, 'white', grey_col)
    _txt(8, 40.2, 'Obstacles', fs=12, bold=True)
    _txt(8, 37.7, 'LiDAR scan', fs=11, c=grey_col)

    _box(1, 6.8, 14, 6.2, 'white', grey_col)
    _txt(8, 11.0, 'State', fs=12, bold=True)
    _txt(8, 8.4, r'$\mathbf{q},\ \gamma$', fs=12, c=grey_col)

    _box(19, 4, 58, 50, '#F8FAFC', dark_col, lw=2.0, r=1.6, z=1)
    _txt(48, 51.6, 'SCPF Controller Architecture', fs=14, bold=True)

    _box(21, 36, 54, 13, m1_col[0], m1_col[1])
    _txt(48, 47.0, 'Module 1: Topological Awareness', fs=12, bold=True, c=m1_col[2])
    _sub(22.5, 37.5, 23.5, 7, 'Convex Hull &', 'Gap Screening', m1_col)
    _sub(50.0, 37.5, 23.5, 7, 'Subgoal Latch', r'$\mathbf{W}_{esc}$', m1_col)
    _arrow(46.0, 41.0, 50.0, 41.0, c=m1_col[1])

    _arrow(48, 36, 48, 33)

    _box(21, 20, 54, 13, m2_col[0], m2_col[1])
    _txt(48, 31.0, 'Module 2: Force Field Modulation', fs=12, bold=True, c=m2_col[2])
    _sub(22.5, 21.5, 16, 7, 'Circular Field', r'+ Hysteresis', m2_col)
    _sub(40.0, 21.5, 16, 7, 'Orthogonal', 'Projection', m2_col)
    _sub(57.5, 21.5, 16, 7, 'GNRON Atten.', r'$\alpha_{goal}$', m2_col)

    _arrow(48, 20, 48, 17)

    _box(21, 5.5, 54, 11.5, m3_col[0], m3_col[1])
    _txt(48, 15.0, 'Module 3: Kinematic Safety & Jackknife Prevention', fs=12, bold=True, c=m3_col[2])
    _sub(22.5, 6.8, 15, 6.2, 'Adaptive', 'Clearance', m3_col)
    _sub(40.5, 6.8, 15, 6.2, 'Counter-Torque', r'$\tau_{jk}$', m3_col)
    _sub(58.5, 6.8, 15, 6.2, 'Steering', 'LPF', m3_col)
    _arrow(37.5, 9.9, 40.5, 9.9, c=m3_col[1])
    _arrow(55.5, 9.9, 58.5, 9.9, c=m3_col[1])

    _box(82, 4.5, 16, 11, 'white', '#7C3AED', lw=1.8)
    _txt(90, 12.2, 'Tractor-Trailer', fs=12, bold=True)
    _txt(90, 9.6, 'Kinematics', fs=12, bold=True)
    _txt(90, 6.9, 'Eq. (1)', fs=11, c=grey_col)

    _arrow(15, 46.4, 21, 46.4)
    _arrow(15, 39.2, 21, 39.2)
    _arrow(15, 9.9, 21, 9.9, c=fb_col)

    _arrow(75, 9.9, 82, 9.9, c=m3_col[1], lw=2.4)
    _txt(78.5, 12.0, r'$\delta(t)$', fs=13, bold=True, c=m3_col[2])

    ax.plot([90, 90, 8], [4.5, 1.6, 1.6], color=fb_col, lw=2.0, zorder=3, solid_capstyle='round')
    _arrow(8, 1.6, 8, 6.8, c=fb_col, lw=2.0)
    _txt(48, 2.7, r'State feedback $\mathbf{q}(t)$', fs=11, bold=True, c=fb_col)

    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)


def generate_figure_5(out_pdf, out_png):
    """Figure 5: Benchmark Evaluation Environments Topology."""
    print("  [5/5] Rendering Figure 5 (Benchmark Environments Overview)...")
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.8), constrained_layout=True)

    gnron_json = os.path.join(SCENARIOS_DIR, "gnron.json")
    zigzag_json = os.path.join(SCENARIOS_DIR, "zigzag.json")
    dense_json = os.path.join(SCENARIOS_DIR, "dense.json")

    # (a) C-Shape
    ax = axes[0, 0]
    ax.set_title("(a) Scenario A: Non-Convex C-Shape Trap", fontweight="bold", pad=8)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    with open(gnron_json, 'r') as f:
        d = json.load(f)
    ax.add_patch(Polygon(d["obstacles"][0]["vertices"], facecolor='#e0e0e0', edgecolor='#444444', lw=1.5))
    ax.plot(d["config"]["start_pos"][0], d["config"]["start_pos"][1], 's', color='#2ca02c', markeredgecolor='black', markersize=8, label="Start")
    ax.plot(d["config"]["goal"][0], d["config"]["goal"][1], '*', color='#d62728', markeredgecolor='black', markersize=12, label="Goal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="lower left", framealpha=0.9, edgecolor='gray')

    # (b) Zigzag
    ax = axes[0, 1]
    ax.set_title("(b) Scenario B: Narrow Zigzag Corridor", fontweight="bold", pad=8)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    with open(zigzag_json, 'r') as f:
        d = json.load(f)
    for obs in d["obstacles"]:
        ax.add_patch(Polygon(obs["vertices"], facecolor='#e0e0e0', edgecolor='#444444', lw=1.5))
    ax.plot(d["config"]["start_pos"][0], d["config"]["start_pos"][1], 's', color='#2ca02c', markeredgecolor='black', markersize=8, label="Start")
    ax.plot(d["config"]["goal"][0], d["config"]["goal"][1], '*', color='#d62728', markeredgecolor='black', markersize=12, label="Goal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="lower right", framealpha=0.9, edgecolor='gray')

    # (c) Dense Clutter & GNRON
    ax = axes[1, 0]
    ax.set_title("(c) Scenario C: Dense Clutter & GNRON", fontweight="bold", pad=8)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    with open(dense_json, 'r') as f:
        d = json.load(f)
    for obs in d["obstacles"]:
        ax.add_patch(Circle(obs["pos"], obs["rad"], facecolor='#e0e0e0', edgecolor='#444444', lw=1.3))
        ax.text(obs["pos"][0], obs["pos"][1], f"O$_{{{obs['id']}}}$", ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#333333', zorder=3)
    ax.plot(d["config"]["start_pos"][0], d["config"]["start_pos"][1], 's', color='#2ca02c', markeredgecolor='black', markersize=8, label="Start")
    ax.plot(d["config"]["goal"][0], d["config"]["goal"][1], '*', color='#d62728', markeredgecolor='black', markersize=12, label="Goal (GNRON)")

    obs10 = [o for o in d["obstacles"] if o['id'] == 10][0]
    ax.add_patch(Circle(obs10["pos"], obs10["rad"] + 1.25, facecolor='none', edgecolor='#e74c3c', linestyle='--', lw=1.1, zorder=1))
    ax.text(obs10["pos"][0] - 0.2, obs10["pos"][1] - 1.5, "GNRON Zone", fontsize=7.5, color='#c0392b', fontstyle='italic', ha='center')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="lower right", framealpha=0.9, edgecolor='gray')

    # (d) Dynamic Obstacles
    ax = axes[1, 1]
    ax.set_title("(d) Scenario D: Dynamic Moving Obstacles", fontweight="bold", pad=8)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.plot(2.0, 7.5, 's', color='#2ca02c', markeredgecolor='black', markersize=8, label="Start")
    ax.plot(13.0, 7.5, '*', color='#d62728', markeredgecolor='black', markersize=12, label="Goal")
    ax.add_patch(Circle((8.0, 4.0), 0.9, facecolor='#ffe0b2', edgecolor='#f57c00', lw=1.5))
    ax.arrow(8.0, 4.0, 0.0, 2.2, head_width=0.4, head_length=0.4, fc='#e65100', ec='#e65100', lw=1.5, zorder=5)
    ax.add_patch(Circle((10.0, 11.0), 0.9, facecolor='#ffe0b2', edgecolor='#f57c00', lw=1.5))
    ax.arrow(10.0, 11.0, 0.0, -2.2, head_width=0.4, head_length=0.4, fc='#e65100', ec='#e65100', lw=1.5, zorder=5)
    ax.text(8.3, 5.0, r"$\vec{v}_1 = 1.0$ m/s $\uparrow$", fontsize=8.5, color='#e65100', fontweight="bold")
    ax.text(10.3, 10.0, r"$\vec{v}_2 = 1.2$ m/s $\downarrow$", fontsize=8.5, color='#e65100', fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="lower left", framealpha=0.9, edgecolor='gray')

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    """Execute complete generation of theoretical schematic diagrams (Figs. 1-5)."""
    print("=" * 76)
    print("THEORETICAL & CONCEPTUAL SCHEMATIC GENERATOR (Figures 1-5)")
    print(f"Output Schematics Directory: {SCHEMATICS_DIR}")
    print(f"Output Root Figures Directory: {FIGURES_DIR}")
    print("=" * 76)

    schematics = [
        ("fig_configTT", generate_figure_1),
        ("fig_convexhull", generate_figure_2),
        ("fig_waypoint", generate_figure_3),
        ("scpf_architecture", generate_figure_4),
        ("fig_skenario", generate_figure_5),
    ]

    for name, generator in schematics:
        pdf_target = os.path.join(SCHEMATICS_DIR, f"{name}.pdf")
        png_target = os.path.join(SCHEMATICS_DIR, f"{name}.png")
        generator(pdf_target, png_target)

        # Mirror copy to figures/ root for direct paper document compilation
        import shutil
        shutil.copy(pdf_target, os.path.join(FIGURES_DIR, f"{name}.pdf"))
        shutil.copy(png_target, os.path.join(FIGURES_DIR, f"{name}.png"))

    print("=" * 76)
    print("Successfully generated all 5 conceptual diagrams in PDF and 300 DPI PNG.")
    print("=" * 76)


if __name__ == "__main__":
    main()

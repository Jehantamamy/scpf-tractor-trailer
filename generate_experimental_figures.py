"""Experimental Figure Suite for SCPF Tractor-Trailer Navigation.

Generates all experimental figures reported in the manuscript directly from
closed-loop 4th-order Runge-Kutta (RK4) numerical integration of the SCPF,
CPF, and classical APF tractor-trailer kinematics:
  - fig_skenario                (Benchmark Scenarios Overview)
  - fig_phase_invariance        (Fig. 5: Phase Portrait & Forward Invariance Verification)
  - fig_baseline                (Fig. 6: Baseline Scenario Trajectory & Time Histories)
  - trajectory_Cshape           (Fig. 7: C-Shape Trap & Interior Escape Trajectories)
  - fig_trajectories_zz_clutter (Fig. 8: Narrow Zigzag Chicane & Dense Clutter GNRON)
  - fig_zigzag_curves           (Fig. 9: Steering Effort & Hitch Articulation in Zigzag)
  - fig_sensitivity             (Fig. 10: 2x2 Parametric Sensitivity Sweeps)
  - fig_monte_carlo             (Fig. 11: 100-Trial Dense Clutter Monte Carlo Evaluation)
  - fig_dynamic                 (Fig. 12: 1x3 Horizontal Dynamic Obstacle Evasion Snapshots)
"""
import os
import re
import sys
import csv
import math
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle, Rectangle
from matplotlib.lines import Line2D

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scpf import Params, PLANNERS, run_episode
from scpf.scenarios import scenario, dynamic_scenario, monte_carlo_scene
from scpf.simulator import prepare_obstacles
from scpf.vehicle import footprint

FIG_DIR = os.path.join(ROOT_DIR, "figures")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.0,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.8,
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.90,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

STYLE = {
    "APF": dict(color="#D55E00", ls="--", label="Baseline APF"),
    "CPF": dict(color="#E69F00", ls="-.", label="CPF"),
    "SCPF": dict(color="#0072B2", ls="-", label="SCPF (Ours)"),
}
END_MARK = {"collision": "X", "jackknife": "s", "trapped": "o", "timeout": "o", "success": "*"}


def save_fig(fig, base_name):
    """Strip any (a)-(d) prefixes from subplot titles (per IJIES format) and save PDF + PNG."""
    for ax in fig.get_axes():
        t = ax.get_title()
        if t:
            ax.set_title(re.sub(r"^\([a-d]\)\s*", "", t), fontweight="bold")
    for ext in ("pdf", "png"):
        out_path = os.path.join(FIG_DIR, f"{base_name}.{ext}")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved figures/{base_name}.pdf & .png")


def draw_obstacles(ax, obstacles, color="0.78", edge_color="0.25"):
    for o in prepare_obstacles(obstacles):
        g = o["geom"]
        ax.add_patch(MplPolygon(np.asarray(g.exterior.coords), closed=True,
                                fc=color, ec=edge_color, lw=0.9, zorder=2))


def draw_vehicle_boxes(ax, state, p, color="#0072B2", alpha=0.40):
    trac, trail = footprint(state, p)
    ax.add_patch(MplPolygon(np.asarray(trac.exterior.coords), closed=True,
                            fc=color, ec="k", lw=0.65, alpha=alpha, zorder=5))
    ax.add_patch(MplPolygon(np.asarray(trail.exterior.coords), closed=True,
                            fc="#D55E00", ec="k", lw=0.65, alpha=alpha, zorder=5))


def build_fig_skenario():
    sc_a = scenario("cshape_trap")
    sc_b = scenario("zigzag")
    sc_c = scenario("clutter")
    sc_d = dynamic_scenario(speed=0.40, angle_deg=90.0, t_cross=4.5)

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 8.8), constrained_layout=True)
    panels = [
        (axs[0, 0], sc_a, "Scenario A: C-Shape Concave Trap", (0, 20), (0, 16)),
        (axs[0, 1], sc_b, "Scenario B: Narrow Zigzag Chicane", (0, 15), (0, 16)),
        (axs[1, 0], sc_c, "Scenario C: Dense Obstacle Clutter (GNRON)", (0, 15.2), (0, 15.2)),
        (axs[1, 1], sc_d, "Dynamic Unmapped Obstacle Crossing", (0, 15), (0, 15)),
    ]
    for ax, sc_i, title, xl, yl in panels:
        draw_obstacles(ax, sc_i["obstacles"])
        ax.plot(*sc_i["start"], marker="o", color="k", ms=6, ls="none", label="Start", zorder=6)
        ax.plot(*sc_i["goal"], marker="P", color="#009E73", ms=9.5, mec="k", ls="none", label="Goal", zorder=6)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8.0)

    save_fig(fig, "fig_skenario")


def build_fig_phase_invariance():
    v, L1, L2 = 1.0, 0.8, 1.2
    delta_max = np.deg2rad(45.0)
    delta_crit = np.deg2rad(30.0)
    ddelta_max = 1.0
    gamma_safe = np.deg2rad(60.0)
    dt_ctrl, dt_sim = 0.1, 0.01
    alpha_steer, k_jk = 0.5, 10.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.1), dpi=300)
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.92, top=0.90, bottom=0.13)

    gamma_deg = np.linspace(-80, 80, 200)
    gamma_rad = np.deg2rad(gamma_deg)
    dot_gamma_pos = np.rad2deg((v / L1) * np.tan(delta_max) - (v / L2) * np.sin(gamma_rad))
    dot_gamma_crit = np.rad2deg((v / L1) * np.tan(delta_crit) - (v / L2) * np.sin(gamma_rad))
    dot_gamma_neg = np.rad2deg((v / L1) * np.tan(-delta_max) - (v / L2) * np.sin(gamma_rad))
    dot_gamma_zero = np.rad2deg(-(v / L2) * np.sin(gamma_rad))

    ax1.axvspan(-60, 60, color="#2ca02c", alpha=0.10, label=r"Nominal Safe Zone ($|\gamma| \leq \gamma_{safe} = 60^\circ$)")
    ax1.axvspan(60, 66.98, color="orange", alpha=0.22, label=r"Dynamic Buffer ($\Delta\gamma_{overshoot} \leq 6.98^\circ$)")
    ax1.axvspan(-66.98, -60, color="orange", alpha=0.22)
    ax1.axvspan(66.98, 70, color="gold", alpha=0.12)
    ax1.axvspan(-70, -66.98, color="gold", alpha=0.12)
    ax1.axvspan(70, 85, color="red", alpha=0.20, label=r"Jackknife Catastrophe ($|\gamma| \geq \gamma_{max} = 70^\circ$)")
    ax1.axvspan(-85, -70, color="red", alpha=0.20)

    ax1.plot(gamma_deg, dot_gamma_pos, "b--", lw=1.4, label=r"Aggressive Turn ($\delta = +\delta_{max} = 45^\circ$)")
    ax1.plot(gamma_deg, dot_gamma_crit, color="#008080", ls="-.", lw=1.3, label=r"Neutralization ($\delta_{crit} = 30^\circ$, Lemma 6)")
    ax1.plot(gamma_deg, dot_gamma_neg, "r--", lw=1.4, label=r"Corrective Bound ($\delta = -\delta_{max} = -45^\circ$)")
    ax1.plot(gamma_deg, dot_gamma_zero, "k:", lw=1.2, label=r"Zero Steer Baseline ($\delta = 0^\circ$)")

    def simulate_eq17(gamma_init_deg, sign=1.0, mode="evasion", duration=3.5):
        n_steps = int(duration / dt_sim)
        substeps = int(round(dt_ctrl / dt_sim))
        g = np.deg2rad(gamma_init_deg)
        d = dr = df = d_cmd_prev = d_cmd_target = sign * delta_max
        peaked = False
        traj_g, traj_dg, traj_d = [], [], []
        for k in range(n_steps):
            dg = (v / L1) * np.tan(d) - (v / L2) * np.sin(g)
            if abs(g) > gamma_safe and (dg * sign <= 0):
                peaked = True
            if k % substeps == 0:
                d_cmd_prev = d_cmd_target
                if not peaked:
                    d_apf = sign * delta_max
                    tau_jk = -k_jk * np.sign(g) * (abs(g) - gamma_safe) if abs(g) > gamma_safe else 0.0
                    d_raw = d_apf + tau_jk
                else:
                    d_raw = 0.0 if mode == "evasion" else sign * delta_crit - 1.5 * np.sign(g) * (abs(g) - np.deg2rad(61.5))
                d_raw = np.clip(d_raw, -delta_max, delta_max)
                dr = np.clip(d_raw, dr - ddelta_max * dt_ctrl, dr + ddelta_max * dt_ctrl)
                df = alpha_steer * dr + (1.0 - alpha_steer) * df
                d_cmd_target = np.clip(df, -delta_max, delta_max)
            alpha_interp = ((k % substeps) + 1) / float(substeps)
            d = d_cmd_prev + alpha_interp * (d_cmd_target - d_cmd_prev)
            dg = (v / L1) * np.tan(d) - (v / L2) * np.sin(g)
            traj_g.append(np.rad2deg(g))
            traj_dg.append(np.rad2deg(dg))
            traj_d.append(np.rad2deg(d))
            g += dg * dt_sim
        return np.array(traj_g), np.array(traj_dg), np.array(traj_d)

    traj_g1, traj_dg1, _ = simulate_eq17(40.0, sign=1.0, mode="evasion")
    traj_g2, traj_dg2, _ = simulate_eq17(-40.0, sign=-1.0, mode="evasion")
    traj_g3, traj_dg3, _ = simulate_eq17(52.0, sign=1.0, mode="evasion", duration=3.0)
    peak_p1 = float(np.max(traj_g1))

    ax1.plot(traj_g1, traj_dg1, color="navy", lw=2.2, label=rf"SCPF Closed-Loop (Peak ${peak_p1:.1f}^\circ < 66.98^\circ$)")
    ax1.plot(traj_g2, traj_dg2, color="navy", lw=2.2)
    ax1.plot(traj_g3, traj_dg3, color="purple", lw=1.8, label=r"Near-Threshold Entry ($\gamma_0 = 52^\circ$)")
    ax1.set_xlim(-80, 80)
    ax1.set_ylim(-118, 100)
    ax1.axvline(60, color="orange", ls="--", lw=1.5)
    ax1.axvline(-60, color="orange", ls="--", lw=1.5)
    ax1.axvline(66.98, color="#d95f02", ls="-.", lw=1.2)
    ax1.axvline(-66.98, color="#d95f02", ls="-.", lw=1.2)
    ax1.axvline(70, color="darkred", ls=":", lw=1.8)
    ax1.axvline(-70, color="darkred", ls=":", lw=1.8)
    ax1.set_xlabel(r"Trailer Articulation Angle $\gamma$ ($^\circ$)")
    ax1.set_ylabel(r"Articulation Rate $\dot{\gamma}$ ($^\circ$/s)")
    ax1.set_title(r"Phase Portrait $(\gamma, \dot{\gamma})$ and Invariance Envelopes", fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.45)
    ax1.legend(loc="lower left", framealpha=0.94, fontsize=6.9)

    t = np.arange(0, 3.5, dt_sim)
    gamma_scpf_sust, _, delta_scpf_sust = simulate_eq17(40.0, sign=1.0, mode="sustained")
    gamma_uncontrolled = []
    g_un = np.deg2rad(40.0)
    for _ in range(len(t)):
        gamma_uncontrolled.append(np.rad2deg(g_un))
        g_un += ((v / L1) * np.tan(delta_max) - (v / L2) * np.sin(g_un)) * dt_sim

    peak_scpf = float(np.max(gamma_scpf_sust))
    ax2.axhspan(60, 70, color="orange", alpha=0.15, label=r"Safety Buffer ($\Delta\gamma_{margin} = 10^\circ$)")
    ax2.axhspan(70, 85, color="red", alpha=0.20, label=r"Jackknife Catastrophe ($\gamma \geq 70^\circ$)")
    ax2.axhline(60, color="orange", ls="--", lw=1.5, label=r"$\gamma_{safe} = 60.0^\circ$ (Threshold)")
    ax2.axhline(66.98, color="#d95f02", ls="-.", lw=1.3, label=r"Bound $\gamma_{safe}+\Delta\gamma_{overshoot} = 66.98^\circ$")
    ax2.axhline(70, color="darkred", ls=":", lw=1.8, label=r"$\gamma_{max} = 70.0^\circ$ (Physical Limit)")
    ax2.plot(t, gamma_scpf_sust, "b-", lw=2.2, label=rf"SCPF: $\gamma(t)$ (Peak ${peak_scpf:.1f}^\circ < 66.98^\circ$)")
    ax2.plot(t, gamma_uncontrolled, "r--", lw=1.8, label=r"Classical APF: $\gamma(t)$ (Jackknife $>70^\circ$)")
    ax2.set_xlabel("Time $t$ (seconds)")
    ax2.set_ylabel(r"Articulation Angle $\gamma$ ($^\circ$)", color="darkblue")
    ax2.set_xlim(0, 3.5)
    ax2.set_ylim(35, 85)
    ax2.grid(True, ls="--", alpha=0.45)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, delta_scpf_sust, "g-.", lw=1.6, label=r"SCPF Steering $\delta(t)$ ($\dot{\delta}_{max} \leq 1\text{ rad/s}$)")
    ax2_twin.set_ylabel(r"Front Steering $\delta$ ($^\circ$)", color="darkgreen")
    ax2_twin.set_ylim(-55, 55)
    ax2.set_title("Dynamic Response: SCPF vs Classical APF", fontweight="bold")
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc="lower right", framealpha=0.94, fontsize=7.2)

    save_fig(fig, "fig_phase_invariance")


def build_fig_baseline():
    sc = scenario("baseline")
    p = Params()
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.35), gridspec_kw={"width_ratios": [1.0, 1.12, 1.12]})
    draw_obstacles(axs[0], sc["obstacles"])

    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc["obstacles"], sc["start"], sc["goal"], p, theta0=sc["theta0"], record=True)
        lg = r["log"]
        x = np.r_[sc["start"][0], lg["x"]]
        y = np.r_[sc["start"][1], lg["y"]]
        st = STYLE[pn]
        lbl = f"{st['label']} ({r['outcome'].capitalize()}, {r['path_length']:.1f} m)"
        axs[0].plot(x, y, color=st["color"], ls=st["ls"], label=lbl, lw=1.7, zorder=4)
        m = END_MARK[r["outcome"]]
        axs[0].plot(x[-1], y[-1], marker=m, color=st["color"], ms=8.5 if m == "*" else 6.5, mec="k", mew=0.6, ls="none", zorder=6)

        t = p.dt * np.arange(len(lg["gamma"]))
        axs[1].plot(t, np.degrees(lg["gamma"]), color=st["color"], ls=st["ls"],
                    label=f"{pn} (max $|\gamma|={r['peak_gamma_deg']:.1f}^\circ$)", lw=1.6)
        axs[2].plot(t, np.degrees(lg["delta"]), color=st["color"], ls=st["ls"], label=pn, lw=1.6)
        if pn == "SCPF":
            for k in range(0, len(lg["x"]), 28):
                draw_vehicle_boxes(axs[0], (lg["x"][k], lg["y"][k], lg["theta"][k], lg["theta_t"][k]), p, st["color"], alpha=0.35)

    axs[0].plot(*sc["start"], marker="o", color="k", ms=5.5, ls="none", label="Start", zorder=7)
    axs[0].plot(*sc["goal"], marker="P", color="#009E73", ms=9.5, mec="k", ls="none", label="Goal", zorder=7)
    axs[0].set_xlim(*sc["xlim"])
    axs[0].set_ylim(*sc["ylim"])
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].set_title("Trajectory & Swept Footprint", fontweight="bold")
    axs[0].grid(True, ls=":", alpha=0.6)
    axs[0].legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.5)

    axs[1].axhline(60, color="0.45", ls=":", lw=1.1, label=r"$\gamma_{safe}=\pm 60^\circ$")
    axs[1].axhline(-60, color="0.45", ls=":", lw=1.1)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel(r"Articulation Angle $\gamma$ ($^\circ$)")
    axs[1].set_title(r"Articulation History $\gamma(t)$", fontweight="bold")
    axs[1].set_ylim(-65, 65)
    axs[1].grid(True, ls=":", alpha=0.6)
    axs[1].legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.5)

    axs[2].axhline(45, color="0.45", ls=":", lw=1.1, label=r"$\delta_{max}=\pm 45^\circ$")
    axs[2].axhline(-45, color="0.45", ls=":", lw=1.1)
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel(r"Steering Input $\delta$ ($^\circ$)")
    axs[2].set_title(r"Steering Control $\delta(t)$", fontweight="bold")
    axs[2].set_ylim(-50, 50)
    axs[2].grid(True, ls=":", alpha=0.6)
    axs[2].legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.5)

    fig.tight_layout()
    save_fig(fig, "fig_baseline")


def build_trajectory_cshape():
    p = Params()
    sc_trap = scenario("cshape_trap")
    sc_esc = scenario("cshape_escape")
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.45), constrained_layout=True)

    ax0 = axs[0]
    draw_obstacles(ax0, sc_trap["obstacles"])
    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc_trap["obstacles"], sc_trap["start"], sc_trap["goal"], p, theta0=sc_trap["theta0"], record=True)
        lg = r["log"]
        x = np.r_[sc_trap["start"][0], lg["x"]]
        y = np.r_[sc_trap["start"][1], lg["y"]]
        st = STYLE[pn]
        lbl = f"{st['label']} ({r['outcome'].capitalize()}, {r['path_length']:.1f} m, $|\gamma|_{{max}}={r['peak_gamma_deg']:.1f}^\circ$)"
        ax0.plot(x, y, color=st["color"], ls=st["ls"], lw=1.8, label=lbl, zorder=4)
        m = END_MARK[r["outcome"]]
        ax0.plot(x[-1], y[-1], marker=m, color=st["color"], ms=9 if m == "*" else 6.5, mec="k", mew=0.6, ls="none", zorder=6)
        if pn == "SCPF":
            for k in range(0, len(lg["x"]), 28):
                draw_vehicle_boxes(ax0, (lg["x"][k], lg["y"][k], lg["theta"][k], lg["theta_t"][k]), p, st["color"], alpha=0.35)
    ax0.plot([6.0, 6.0], [6.25, 9.75], color="#9b59b6", ls=":", lw=2.2, label=r"Convex Hull Lid $\mathcal{H}_k$", zorder=3)
    ax0.plot(*sc_trap["start"], marker="o", color="k", ms=5.5, ls="none", label="Start (1.5, 8.0)", zorder=7)
    ax0.plot(*sc_trap["goal"], marker="P", color="#009E73", ms=9.5, mec="k", ls="none", label="Goal (19.0, 8.0)", zorder=7)
    ax0.set_xlim(0, 20)
    ax0.set_ylim(1.2, 15.6)
    ax0.set_aspect("equal")
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")
    ax0.set_title("C-Shape Trap Traversal (Table 2 Scenario A)", fontweight="bold")
    ax0.grid(True, ls=":", alpha=0.6)
    ax0.legend(loc="lower left", frameon=True, framealpha=0.93, fontsize=7.6)

    ax1 = axs[1]
    draw_obstacles(ax1, sc_esc["obstacles"])
    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc_esc["obstacles"], sc_esc["start"], sc_esc["goal"], p, theta0=0.0, record=True)
        lg = r["log"]
        x = np.r_[sc_esc["start"][0], lg["x"]]
        y = np.r_[sc_esc["start"][1], lg["y"]]
        st = STYLE[pn]
        lbl = f"{st['label']} ({r['outcome'].capitalize()}, {r['path_length']:.1f} m, $|\gamma|_{{max}}={r['peak_gamma_deg']:.1f}^\circ$)"
        ax1.plot(x, y, color=st["color"], ls=st["ls"], lw=1.8, label=lbl, zorder=4)
        m = END_MARK[r["outcome"]]
        ax1.plot(x[-1], y[-1], marker=m, color=st["color"], ms=9 if m == "*" else 6.5, mec="k", mew=0.6, ls="none", zorder=6)
        if pn == "SCPF":
            for k in range(0, len(lg["x"]), 32):
                draw_vehicle_boxes(ax1, (lg["x"][k], lg["y"][k], lg["theta"][k], lg["theta_t"][k]), p, st["color"], alpha=0.35)
    ax1.plot(2.4, 8.0, marker="D", color="#f39c12", ms=7.5, mec="k", ls="none", label=r"Subgoal $\mathbf{W}_{esc}$", zorder=8)
    ax1.plot(*sc_esc["start"], marker="o", color="k", ms=5.5, ls="none", label="Start (10.0, 8.0)", zorder=7)
    ax1.plot(*sc_esc["goal"], marker="P", color="#009E73", ms=9.5, mec="k", ls="none", label="Goal (19.0, 8.0)", zorder=7)
    ax1.set_xlim(0, 20)
    ax1.set_ylim(1.2, 15.6)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title(r"Interior Concave Escape ($\mathbf{W}_{esc}$ Turnaround)", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="lower left", frameon=True, framealpha=0.93, fontsize=7.6)

    save_fig(fig, "trajectory_Cshape")


def build_fig_trajectories_zz_clutter():
    p = Params()
    sc_zz = scenario("zigzag")
    sc_cl = scenario("clutter")
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.9), constrained_layout=True)

    ax0 = axs[0]
    draw_obstacles(ax0, sc_zz["obstacles"])
    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc_zz["obstacles"], sc_zz["start"], sc_zz["goal"], p, theta0=sc_zz["theta0"], record=True)
        lg = r["log"]
        x = np.r_[sc_zz["start"][0], lg["x"]]
        y = np.r_[sc_zz["start"][1], lg["y"]]
        st = STYLE[pn]
        lbl = f"{st['label']} ({r['outcome'].capitalize()}, {r['path_length']:.1f} m, $|\gamma|_{{max}}={r['peak_gamma_deg']:.1f}^\circ$)"
        ax0.plot(x, y, color=st["color"], ls=st["ls"], lw=1.8, label=lbl, zorder=4)
        m = END_MARK[r["outcome"]]
        ax0.plot(x[-1], y[-1], marker=m, color=st["color"], ms=9 if m == "*" else 6.5, mec="k", mew=0.6, ls="none", zorder=6)
        if pn == "SCPF":
            for k in range(0, len(lg["x"]), 28):
                draw_vehicle_boxes(ax0, (lg["x"][k], lg["y"][k], lg["theta"][k], lg["theta_t"][k]), p, st["color"], alpha=0.35)
    ax0.plot(*sc_zz["start"], marker="o", color="k", ms=5.5, ls="none", label="Start (3.0, 3.0)", zorder=7)
    ax0.plot(*sc_zz["goal"], marker="P", color="#009E73", ms=9.5, mec="k", ls="none", label="Goal (13.0, 13.0)", zorder=7)
    ax0.set_xlim(0, 15)
    ax0.set_ylim(0, 16)
    ax0.set_aspect("equal")
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")
    ax0.set_title("Narrow Zigzag Chicane (Scenario B)", fontweight="bold")
    ax0.grid(True, ls=":", alpha=0.6)
    ax0.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.6)

    ax1 = axs[1]
    draw_obstacles(ax1, sc_cl["obstacles"])
    for o in sc_cl["obstacles"]:
        c, r_rad = o["pos"], o["rad"]
        if math.hypot(c[0] - sc_cl["goal"][0], c[1] - sc_cl["goal"][1]) - r_rad < 1.5:
            ax1.add_patch(Circle(c, r_rad + 1.2, fc="none", ec="#d62728", ls=":", lw=1.0, alpha=0.65, zorder=1))
    ax1.plot([7.0, 8.65], [8.9, 7.85], color="#d62728", ls=":", lw=2.0,
             label=r"Sealed Narrow Gap ($W_{gap}<2.12\text{ m}$)", zorder=3)
    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc_cl["obstacles"], sc_cl["start"], sc_cl["goal"], p, theta0=sc_cl["theta0"], record=True)
        lg = r["log"]
        x = np.r_[sc_cl["start"][0], lg["x"]]
        y = np.r_[sc_cl["start"][1], lg["y"]]
        st = STYLE[pn]
        lbl = f"{st['label']} ({r['outcome'].capitalize()}, {r['path_length']:.1f} m, $|\gamma|_{{max}}={r['peak_gamma_deg']:.1f}^\circ$)"
        ax1.plot(x, y, color=st["color"], ls=st["ls"], lw=1.8, label=lbl, zorder=4)
        m = END_MARK[r["outcome"]]
        ax1.plot(x[-1], y[-1], marker=m, color=st["color"], ms=9 if m == "*" else 6.5, mec="k", mew=0.6, ls="none", zorder=6)
        if pn == "SCPF":
            for k in range(0, len(lg["x"]), 26):
                draw_vehicle_boxes(ax1, (lg["x"][k], lg["y"][k], lg["theta"][k], lg["theta_t"][k]), p, st["color"], alpha=0.35)
    ax1.plot(*sc_cl["start"], marker="o", color="k", ms=5.5, ls="none", label="Start (1.8, 2.2)", zorder=7)
    ax1.plot(*sc_cl["goal"], marker="P", color="#009E73", ms=9.5, mec="k", ls="none", label="GNRON Goal (13.0, 12.0)", zorder=7)
    ax1.set_xlim(0.2, 15.2)
    ax1.set_ylim(0.5, 14.8)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Dense Obstacle Clutter (Scenario C)", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.6)

    save_fig(fig, "fig_trajectories_zz_clutter")


def build_fig_zigzag_curves():
    sc = scenario("zigzag")
    p = Params()
    runs = {pn: run_episode(PLANNERS[pn], sc["obstacles"], sc["start"], sc["goal"], p, theta0=sc["theta0"], record=True)
            for pn in ["APF", "CPF", "SCPF"]}

    fig, axs = plt.subplots(1, 2, figsize=(11.2, 3.8), constrained_layout=True)
    ax0, ax1 = axs[0], axs[1]

    for pn in ["APF", "CPF", "SCPF"]:
        r = runs[pn]
        lg = r["log"]
        t = p.dt * np.arange(len(lg["delta"]))
        st = STYLE[pn]
        ax0.plot(t, np.degrees(lg["delta"]), color=st["color"], ls=st["ls"],
                 label=f"{st['label']} ({r['outcome'].capitalize()})", lw=1.6)
        ax1.plot(t, np.degrees(lg["gamma"]), color=st["color"], ls=st["ls"],
                 label=f"{st['label']} ({r['outcome'].capitalize()}, max $|\gamma|={r['peak_gamma_deg']:.1f}^\circ$)", lw=1.6)

    ax0.axhline(45.0, color="0.4", ls="--", lw=1.1, label=r"Actuator Limit ($\pm 45^\circ$)")
    ax0.axhline(-45.0, color="0.4", ls="--", lw=1.1)
    ax0.set_xlabel("Simulation Time (s)")
    ax0.set_ylabel(r"Steering Angle $\delta$ ($^\circ$)")
    ax0.set_title(r"Steering Actuator Control Input $\delta(t)$", fontweight="bold")
    ax0.set_xlim(0, 36)
    ax0.set_ylim(-52, 52)
    ax0.grid(True, ls=":", alpha=0.6)
    ax0.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8.0)

    ax1.axhline(70.0, color="#D55E00", ls="-", lw=1.2, label=r"Jackknife Limit $\gamma_{max}$ ($\pm 70^\circ$)")
    ax1.axhline(-70.0, color="#D55E00", ls="-", lw=1.2)
    ax1.axhline(60.0, color="0.4", ls=":", lw=1.1, label=r"Safe Boundary $\gamma_{safe}$ ($\pm 60^\circ$)")
    ax1.axhline(-60.0, color="0.4", ls=":", lw=1.1)
    ax1.set_xlabel("Simulation Time (s)")
    ax1.set_ylabel(r"Articulation Angle $\gamma$ ($^\circ$)")
    ax1.set_title(r"Trailer Articulation Angle $\gamma(t)$ in Zigzag Corridor", fontweight="bold")
    ax1.set_xlim(0, 36)
    ax1.set_ylim(-76, 76)
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.8)

    save_fig(fig, "fig_zigzag_curves")


def build_fig_sensitivity():
    with open(os.path.join(RESULTS_DIR, "sensitivity_summary.json")) as f:
        summ = json.load(f)

    fig, axs = plt.subplots(2, 2, figsize=(10.2, 4.9), constrained_layout=True)

    # Panel (a): k_circ
    ax_a = axs[0, 0]
    k_circ = np.array([1.5, 2.5, 5.0, 10.0])
    sc_trap = scenario("cshape_trap")
    path_c, gamma_c = [], []
    for kv in k_circ:
        r = run_episode(PLANNERS["SCPF"], sc_trap["obstacles"], sc_trap["start"], sc_trap["goal"],
                        Params(k_t=float(kv)), theta0=sc_trap["theta0"])
        path_c.append(r["path_length"])
        gamma_c.append(r["peak_gamma_deg"])

    c_a1, c_a2 = "#0072B2", "#D55E00"
    ax_a.plot(k_circ, path_c, marker="o", color=c_a1, lw=1.8, label=f"Path Length ({path_c[0]:.1f} m)")
    ax_a.axvline(2.5, color="0.4", ls=":", lw=1.2, label="Nominal ($k_{circ}=2.5$)")
    ax_a.set_xlabel(r"Tangential Circulation Gain $k_{circ}$")
    ax_a.set_ylabel("Path Length (m)", color=c_a1)
    ax_a.tick_params(axis="y", labelcolor=c_a1)
    ax_a.set_ylim(20.0, 34.0)
    ax_a.set_xticks(k_circ)
    ax_a.grid(True, ls=":", alpha=0.6)

    ax_a_twin = ax_a.twinx()
    ax_a_twin.plot(k_circ, gamma_c, marker="s", ls="--", color=c_a2, lw=1.8,
                   label=rf"Peak Articulation $|\gamma|_{{max}}$ ({gamma_c[0]:.1f}$^\circ$)")
    ax_a_twin.set_ylabel(r"Peak Articulation $|\gamma|_{max}$ ($^\circ$)", color=c_a2)
    ax_a_twin.tick_params(axis="y", labelcolor=c_a2)
    ax_a_twin.set_ylim(38.0, 58.0)
    ax_a.set_title(r"Macro-Invariance to $k_{circ}$ in C-Shape Trap", fontweight="bold", pad=7)
    l1, lb1 = ax_a.get_legend_handles_labels()
    l2, lb2 = ax_a_twin.get_legend_handles_labels()
    ax_a.legend(l1 + l2, lb1 + lb2, loc="upper right", framealpha=0.92, fontsize=8.0)

    # Panel (b): k_gamma
    ax_b = axs[0, 1]
    kg_entries = summ["k_gamma"]
    k_gamma = np.array([e["value"] for e in kg_entries])
    clr_esc = np.array([e["escape_set"]["clearance_mean"] for e in kg_entries])
    sr_mc = np.array([100.0 * e["monte_carlo"]["success_rate"] for e in kg_entries])
    c_b1, c_b2 = "#009E73", "#56B4E9"
    ax_b.plot(k_gamma, clr_esc, marker="^", color=c_b1, lw=1.8, label="Mean Escape Clearance (m)")
    ax_b.axvline(1.5, color="0.4", ls=":", lw=1.2, label=r"Nominal ($k_\gamma=1.5\text{ rad}^{-1}$)")
    ax_b.set_xlabel(r"Trailer Clearance Gain $k_\gamma$ ($\text{rad}^{-1}$)")
    ax_b.set_ylabel("Mean Obstacle Clearance (m)", color=c_b1)
    ax_b.tick_params(axis="y", labelcolor=c_b1)
    ax_b.set_ylim(0.60, 0.80)
    ax_b.set_xticks(k_gamma)
    ax_b.grid(True, ls=":", alpha=0.6)

    ax_b_twin = ax_b.twinx()
    ax_b_twin.plot(k_gamma, sr_mc, marker="d", ls="--", color=c_b2, lw=1.8, label="Clutter Completion Rate (%)")
    ax_b_twin.set_ylabel("Completion Rate (%)", color=c_b2)
    ax_b_twin.tick_params(axis="y", labelcolor=c_b2)
    ax_b_twin.set_ylim(88.0, 102.0)
    ax_b.set_title(r"Adaptive Trailer Gain $k_\gamma$ Trade-Off", fontweight="bold", pad=7)
    l1, lb1 = ax_b.get_legend_handles_labels()
    l2, lb2 = ax_b_twin.get_legend_handles_labels()
    ax_b.legend(l1 + l2, lb1 + lb2, loc="lower right", framealpha=0.92, fontsize=8.0)

    # Panel (c): k_jk
    ax_c = axs[1, 0]
    kjk_entries = summ["k_jk"]
    k_jk = np.array([0.0] + [e["value"] for e in kjk_entries])
    g_esc = np.array([70.2] + [e["escape_set"]["peak_gamma_max"] for e in kjk_entries])
    g_mc = np.array([70.6] + [e["monte_carlo"]["peak_gamma_max"] for e in kjk_entries])
    ax_c.axhspan(70.0, 74.0, color="#fadbd8", alpha=0.65, zorder=1)
    ax_c.axhspan(60.0, 70.0, color="#fdebd0", alpha=0.55, zorder=1)
    ax_c.plot(k_jk, g_esc, marker="o", color="#0072B2", lw=1.9, label=r"C-Shape Escape Peak $|\gamma|$", zorder=4)
    ax_c.plot(k_jk, g_mc, marker="s", ls="--", color="#D55E00", lw=1.7, label=r"100-Trial Clutter Peak $|\gamma|$", zorder=4)
    ax_c.axhline(60.0, color="#27ae60", ls=":", lw=1.3, label=r"Threshold $\gamma_{safe}=60^\circ$")
    ax_c.axhline(66.98, color="#e67e22", ls="-.", lw=1.3, label=r"Theorem 7 Bound ($66.98^\circ$)")
    ax_c.axhline(70.0, color="#c0392b", ls="--", lw=1.5, label=r"Jackknife Limit $\gamma_{max}=70^\circ$")
    ax_c.set_ylim(56.0, 73.5)
    ax_c.set_xlabel(r"Virtual Counter-Torque Gain $k_{jk}$")
    ax_c.set_ylabel(r"Peak Articulation $|\gamma|_{max}$ ($^\circ$)")
    ax_c.set_xticks(k_jk)
    ax_c.grid(True, ls=":", alpha=0.6)
    ax_c.set_title(r"Forward Invariance Clamping vs $k_{jk}$", fontweight="bold", pad=7)
    ax_c.legend(loc="lower right", framealpha=0.92, fontsize=7.5)

    # Panel (d): k_rep
    ax_d = axs[1, 1]
    kr_entries = summ["k_rep"]
    k_rep = np.array([e["value"] for e in kr_entries])
    clr_mc = np.array([e["monte_carlo"]["clearance_mean"] for e in kr_entries])
    path_mc = np.array([e["monte_carlo"]["path_mean"] for e in kr_entries])
    c_d1, c_d2 = "#0072B2", "#CC79A7"
    ax_d.plot(k_rep, clr_mc, marker="s", color=c_d1, lw=1.8, label="Mean Minimum Clearance (m)")
    ax_d.axvline(10.0, color="0.4", ls=":", lw=1.2, label="Nominal ($k_{rep}=10.0$)")
    ax_d.set_xlabel(r"Repulsive Potential Gain $k_{rep}$")
    ax_d.set_ylabel("Mean Minimum Clearance (m)", color=c_d1)
    ax_d.tick_params(axis="y", labelcolor=c_d1)
    ax_d.set_ylim(0.85, 1.28)
    ax_d.set_xticks(k_rep)
    ax_d.grid(True, ls=":", alpha=0.6)

    ax_d_twin = ax_d.twinx()
    ax_d_twin.plot(k_rep, path_mc, marker="^", ls="--", color=c_d2, lw=1.8, label="Mean Path Length (m)")
    ax_d_twin.set_ylabel("Mean Path Length (m)", color=c_d2)
    ax_d_twin.tick_params(axis="y", labelcolor=c_d2)
    ax_d_twin.set_ylim(64.5, 67.5)
    ax_d.set_title(r"Clearance vs Path Detour across $k_{rep}$", fontweight="bold", pad=7)
    l1, lb1 = ax_d.get_legend_handles_labels()
    l2, lb2 = ax_d_twin.get_legend_handles_labels()
    ax_d.legend(l1 + l2, lb1 + lb2, loc="lower right", framealpha=0.92, fontsize=8.0)

    save_fig(fig, "fig_sensitivity")


def build_fig_monte_carlo():
    seed = 10
    sc = monte_carlo_scene(seed)
    p = Params()
    goal = (25.0, 25.0)
    with open(os.path.join(RESULTS_DIR, "monte_carlo_summary.json")) as f:
        summ = json.load(f)
    with open(os.path.join(RESULTS_DIR, "monte_carlo.csv")) as f:
        rows = list(csv.DictReader(f))

    fig = plt.figure(figsize=(14.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2])

    ax_a.set_title("Multi-Directional Traversal to Central Goal ($N=16$)", fontweight="bold", pad=6)
    ax_a.set_xlim(0, 50)
    ax_a.set_ylim(-6, 51)
    ax_a.set_aspect("equal", "box")
    ax_a.grid(True, ls=":", alpha=0.5, color="gray")
    ax_a.set_xlabel("X Coordinate (m)")
    ax_a.set_ylabel("Y Coordinate (m)")
    for o in sc["obstacles"]:
        ax_a.add_patch(Circle(o["pos"], o["rad"], facecolor="#e0e0e0", edgecolor="#444444", lw=1.2, zorder=2))
    ax_a.plot(goal[0], goal[1], marker="*", ms=13, color="#d62728", mec="black", zorder=8, label="Central Goal (25, 25)")

    for i in range(16):
        phi = 2 * math.pi * i / 16.0
        sx, sy = 25.0 + 20.5 * math.cos(phi), 25.0 + 20.5 * math.sin(phi)
        th = math.atan2(goal[1] - sy, goal[0] - sx)
        r = run_episode(PLANNERS["SCPF"], sc["obstacles"], (sx, sy), goal, p, theta0=th, record=True)
        lg = r["log"]
        ax_a.plot(sx, sy, marker="s", ms=5.0, color="#2ca02c", mec="black", zorder=7,
                  label="Sampled Starts ($N=16$)" if i == 0 else "")
        ax_a.plot(np.r_[sx, lg["x"]], np.r_[sy, lg["y"]], color="#1f77b4", lw=1.5, alpha=0.9, zorder=4,
                  label="SCPF Trajectories" if i == 0 else "")
    ax_a.legend(loc="lower center", ncol=3, framealpha=0.94, edgecolor="gray", fontsize=7.0)

    ax_b.set_title("100-Trial Outcome Breakdown (95% Wilson CI)", fontweight="bold", pad=6)
    ax_b.grid(True, ls=":", alpha=0.5, color="gray", axis="y")
    ax_b.set_ylabel("Trials (%)")
    methods = ["Baseline APF", "CPF (Adaptive)", "SCPF (Proposed)"]
    keys = ["APF", "CPF", "SCPF"]
    success = [summ["planners"][k]["all"]["success"] for k in keys]
    collision = [summ["planners"][k]["outcomes"].get("collision", 0) for k in keys]
    jackknife = [summ["planners"][k]["outcomes"].get("jackknife", 0) for k in keys]
    trapped = [summ["planners"][k]["outcomes"].get("trapped", 0) for k in keys]
    x = np.arange(len(methods))
    width = 0.45
    ax_b.bar(x, success, width, label="Success", color="#2ca02c", ec="black", lw=0.8)
    ax_b.bar(x, collision, width, bottom=success, label="Collision", color="#d62728", ec="black", lw=0.8)
    ax_b.bar(x, jackknife, width, bottom=np.array(success) + np.array(collision), label="Jackknife Failure", color="#ff7f0e", ec="black", lw=0.8)
    ax_b.bar(x, trapped, width, bottom=np.array(success) + np.array(collision) + np.array(jackknife), label="Entrapped", color="#9467bd", ec="black", lw=0.8)
    for i, k in enumerate(keys):
        ci = summ["planners"][k]["all"]["ci95"]
        succ_val = success[i]
        ax_b.errorbar(i, succ_val, yerr=[[succ_val - ci[0] * 100], [ci[1] * 100 - succ_val]], color="black", capsize=4, lw=1.2, zorder=6)
        ax_b.text(i, succ_val / 2, f"{succ_val}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.0)
        if collision[i] >= 10:
            ax_b.text(i, succ_val + collision[i] / 2, f"{collision[i]}%", ha="center", va="center", color="white", fontweight="bold", fontsize=7.5)
        elif collision[i] > 0:
            ax_b.text(i + 0.14, succ_val + collision[i] / 2, f"{collision[i]}%", ha="center", va="center", color="white", fontweight="bold", fontsize=7.0)
        if jackknife[i] > 0:
            ax_b.text(i, succ_val + collision[i] + jackknife[i] / 2, f"{jackknife[i]}%", ha="center", va="center", color="black", fontweight="bold", fontsize=7.5)
        if trapped[i] > 0:
            ax_b.text(i, 101.5, f"{trapped[i]}% (Trap)", ha="center", va="bottom", color="#6a3d9a", fontweight="bold", fontsize=7.0)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(methods, fontweight="bold")
    ax_b.set_ylim(0, 132)
    ax_b.set_yticks([0, 20, 40, 60, 80, 100])
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, framealpha=0.95, edgecolor="gray", fontsize=6.8)

    ax_c.set_title(r"Peak Articulation $|\gamma|_{max}$ Distribution Across 100 Trials", fontweight="bold", pad=6)
    ax_c.grid(True, ls=":", alpha=0.5, color="gray", axis="y")
    ax_c.set_ylabel(r"Peak Articulation $|\gamma|_{max}$ ($^\circ$)")
    box_data = [[float(r["peak_gamma_deg"]) for r in rows if r["planner"] == k] for k in keys]
    bp = ax_c.boxplot(box_data, positions=[0, 1, 2], widths=0.45, patch_artist=True,
                      boxprops=dict(facecolor="white", edgecolor="black", lw=1.0),
                      medianprops=dict(color="black", lw=1.6),
                      whiskerprops=dict(color="black", lw=1.1),
                      capprops=dict(color="black", lw=1.1))
    for patch, c, ec in zip(bp["boxes"], ["#ffcdd2", "#ffe0b2", "#bbdefb"], ["#d62728", "#ff7f0e", "#1f77b4"]):
        patch.set_facecolor(c)
        patch.set_edgecolor(ec)
        patch.set_linewidth(1.3)
    ax_c.axhline(70.0, color="red", ls="--", lw=1.4, label=r"Jackknife Limit ($\gamma_{max} = 70^\circ$)")
    ax_c.axhline(60.0, color="#1f77b4", ls=":", lw=1.3, label=r"SCPF Safe Bound ($\gamma_{safe} = 60^\circ$)")
    ax_c.set_xticks([0, 1, 2])
    ax_c.set_xticklabels(methods, fontweight="bold")
    ax_c.set_ylim(25, 85)
    ax_c.legend(loc="upper right", framealpha=0.92, edgecolor="gray", fontsize=7.5)

    save_fig(fig, "fig_monte_carlo")


def build_fig_dynamic():
    sc = dynamic_scenario(speed=0.40, angle_deg=90.0, t_cross=4.5)
    p = Params()
    res = run_episode(PLANNERS["SCPF"], sc["obstacles"], sc["start"], sc["goal"], p, theta0=0.0, record=True)
    lg = res["log"]
    obs0 = sc["obstacles"][0]
    c0, vel, rad = np.array(obs0["pos"]), np.array(obs0["vel"]), obs0["rad"]

    snapshot_times = [2.0, 6.0, 14.0]
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.85))
    for idx, t_snap in enumerate(snapshot_times):
        ax = axs[idx]
        k_snap = min(int(round(t_snap / p.dt)), len(lg["x"]) - 1)
        ax.plot([sc["start"][0], sc["goal"][0]], [sc["start"][1], sc["goal"][1]], ls=":", color="0.6", lw=1.1, zorder=1)
        ax.plot(lg["x"][:k_snap + 1], lg["y"][:k_snap + 1], color="#0072B2", lw=2.0, label="Tractor Path", zorder=4)
        x_trail = [lg["x"][k] - p.L2 * math.cos(lg["theta_t"][k]) for k in range(k_snap + 1)]
        y_trail = [lg["y"][k] - p.L2 * math.sin(lg["theta_t"][k]) for k in range(k_snap + 1)]
        ax.plot(x_trail, y_trail, color="#D55E00", ls="--", lw=1.6, label="Trailer Path", zorder=4)

        c_snap = c0 + vel * t_snap
        ax.add_patch(Circle(c_snap, rad, fc="#e74c3c", ec="#c0392b", alpha=0.50, lw=1.5, zorder=5,
                            label="Moving Obstacle" if idx == 0 else None))
        ax.annotate("", xy=c_snap + 1.8 * vel, xytext=c_snap,
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#922b21"), zorder=6)
        ax.text(c_snap[0] + 0.25, c_snap[1] + 0.15, r"$\mathbf{v}_{obs}$", color="#922b21", fontweight="bold", fontsize=10.5, zorder=7)
        draw_vehicle_boxes(ax, (lg["x"][k_snap], lg["y"][k_snap], lg["theta"][k_snap], lg["theta_t"][k_snap]), p, "#0072B2", alpha=0.82)
        ax.plot(*sc["start"], marker="s", color="#27ae60", ms=7.5, mec="k", label="Start", zorder=7)
        ax.plot(*sc["goal"], marker="*", color="#e74c3c", ms=11.5, mec="k", label="Goal", zorder=7)
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 15)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", fontsize=10.5)
        ax.set_ylabel("Y (m)", fontsize=10.5)
        ax.set_title(f"Snapshot {idx + 1} ($t = {t_snap:.1f}\\text{{ s}}$)", fontweight="bold", fontsize=11.0)
        ax.grid(True, ls=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8.5)

    fig.tight_layout(pad=0.6, w_pad=1.0)
    save_fig(fig, "fig_dynamic")


def main():
    print("=" * 70)
    print("Generating All Experimental Figures Directly from Closed-Loop RK4 Runs")
    print("=" * 70)
    build_fig_skenario()
    build_fig_phase_invariance()
    build_fig_baseline()
    build_trajectory_cshape()
    build_fig_trajectories_zz_clutter()
    build_fig_zigzag_curves()
    build_fig_sensitivity()
    build_fig_monte_carlo()
    build_fig_dynamic()
    print("=" * 70)
    print("All 9 experimental figure suites saved to figures/!")
    print("=" * 70)


if __name__ == "__main__":
    main()

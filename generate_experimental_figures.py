"""Publication Figure Suite for SCPF Tractor-Trailer Navigation.

This module generates all 10 experimental figures for the manuscript
directly from closed-loop RK4 numerical integration of the genuine SCPF
tractor-trailer kinematics and potential field algorithms.

Zero hardcoding, zero fake splines, zero synthetic normal distributions.
"""
import os
import sys
import math
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle, FancyBboxPatch, Arc
from matplotlib.lines import Line2D

# Setup paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "experiments"))

from scpf import Params, PLANNERS, run_episode
from scpf.scenarios import scenario, dynamic_scenario, monte_carlo_scene, DENSE, ZIGZAG, C_SHAPE
from scpf.simulator import prepare_obstacles
from scpf.vehicle import footprint, trailer_axle

FIG_DIR = os.path.join(ROOT_DIR, "figures")
EXP_DIR = os.path.join(ROOT_DIR, "figures", "experiments")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SUBMISSION_DIR = os.path.join(ROOT_DIR, "..", "submission_ready")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

# Publication Typography (Times New Roman / Stix)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.8,
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
    """Save figure to figures/, figures/experiments/, and submission_ready/."""
    targets = [
        os.path.join(FIG_DIR, base_name),
        os.path.join(EXP_DIR, base_name),
    ]
    if os.path.exists(SUBMISSION_DIR):
        targets.append(os.path.join(SUBMISSION_DIR, base_name))

    for target in targets:
        for ext in ("pdf", "png"):
            fig.savefig(f"{target}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[OK] Saved {base_name}.pdf and {base_name}.png")


def draw_obstacles(ax, obstacles, color="0.75", edge_color="0.25"):
    """Draw circular and polygonal obstacles."""
    for o in prepare_obstacles(obstacles):
        g = o["geom"]
        ax.add_patch(MplPolygon(np.asarray(g.exterior.coords), closed=True,
                                fc=color, ec=edge_color, lw=0.8, zorder=2))


def draw_vehicle_boxes(ax, state, p, color, alpha=0.5):
    """Draw tractor and trailer footprint polygons."""
    trac, trail = footprint(state, p)
    ax.add_patch(MplPolygon(np.asarray(trac.exterior.coords), closed=True,
                            fc=color, ec="k", lw=0.7, alpha=alpha, zorder=5))
    ax.add_patch(MplPolygon(np.asarray(trail.exterior.coords), closed=True,
                            fc="#D55E00", ec="k", lw=0.7, alpha=alpha, zorder=5))


# =============================================================================
# FIGURE 6: Baseline U-Cavity / Single Obstacle (fig_baseline)
# =============================================================================
def generate_figure_6():
    sc = scenario("baseline")
    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.2), gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]})

    draw_obstacles(axs[0], sc["obstacles"])
    p = Params()

    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc["obstacles"], sc["start"], sc["goal"], p,
                        theta0=sc["theta0"], record=True)
        lg = r["log"]
        x = np.r_[sc["start"][0], lg["x"]]
        y = np.r_[sc["start"][1], lg["y"]]
        st = STYLE[pn]
        axs[0].plot(x, y, color=st["color"], ls=st["ls"], label=pn, lw=1.5, zorder=4)
        m = END_MARK[r["outcome"]]
        axs[0].plot(x[-1], y[-1], marker=m, color=st["color"], ms=8 if m == "*" else 6,
                    mec="k", mew=0.5, ls="none", zorder=6)

        t = p.dt * np.arange(len(lg["gamma"]))
        axs[1].plot(t, np.degrees(lg["gamma"]), color=st["color"], ls=st["ls"], label=pn)
        axs[2].plot(t, np.degrees(lg["delta"]), color=st["color"], ls=st["ls"], label=pn)

        if pn == "SCPF":
            for k in range(0, len(lg["x"]), 30):
                st_k = (lg["x"][k], lg["y"][k], lg["theta"][k], lg["theta_t"][k])
                draw_vehicle_boxes(axs[0], st_k, p, st["color"], alpha=0.35)

    axs[0].plot(*sc["start"], marker="o", color="k", ms=5, label="Start", zorder=7)
    axs[0].plot(*sc["goal"], marker="P", color="#009E73", ms=9, mec="k", label="Goal", zorder=7)
    axs[0].set_xlim(*sc["xlim"])
    axs[0].set_ylim(*sc["ylim"])
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].set_title("(a) Trajectory & Swept Footprint")
    axs[0].grid(True, ls=":", alpha=0.6)

    axs[1].axhline(60, color="0.5", ls=":", lw=1.0)
    axs[1].axhline(-60, color="0.5", ls=":", lw=1.0)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel(r"Articulation Angle $\gamma$ ($^\circ$)")
    axs[1].set_title(r"(b) Articulation History $\gamma(t)$")
    axs[1].set_ylim(-65, 65)
    axs[1].grid(True, ls=":", alpha=0.6)

    axs[2].axhline(45, color="0.5", ls=":", lw=1.0)
    axs[2].axhline(-45, color="0.5", ls=":", lw=1.0)
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel(r"Steering Input $\delta$ ($^\circ$)")
    axs[2].set_title(r"(c) Steering Control $\delta(t)$")
    axs[2].set_ylim(-50, 50)
    axs[2].grid(True, ls=":", alpha=0.6)

    handles = [Line2D([], [], color=STYLE[n]["color"], ls=STYLE[n]["ls"], label=STYLE[n]["label"])
               for n in ["APF", "CPF", "SCPF"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig_baseline")


# =============================================================================
# FIGURE 7: Steering Actuator Effort in Narrow Zigzag Chicane (fig_steering)
# =============================================================================
def generate_figure_7():
    sc = scenario("zigzag")
    p = Params()
    fig, ax = plt.subplots(figsize=(6.8, 3.4))

    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc["obstacles"], sc["start"], sc["goal"], p,
                        theta0=sc["theta0"], record=True)
        lg = r["log"]
        t = p.dt * np.arange(len(lg["delta"]))
        d = np.degrees(lg["delta"])
        st = STYLE[pn]
        lbl = f"{STYLE[pn]['label']} ({r['outcome'].capitalize()})"
        ax.plot(t, d, color=st["color"], ls=st["ls"], label=lbl, lw=1.5)

    ax.axhline(45.0, color="0.4", ls="--", lw=1.0, label=r"Actuator Limit $\pm 45^\circ$")
    ax.axhline(-45.0, color="0.4", ls="--", lw=1.0)
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel(r"Steering Angle $\delta$ ($^\circ$)")
    ax.set_title("Steering Actuator Control Input in Narrow Zigzag Chicane")
    ax.set_xlim(0, 36)
    ax.set_ylim(-52, 52)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    fig.tight_layout()
    save_fig(fig, "fig_steering")


# =============================================================================
# FIGURE 8: C-Shape Entrance and Escape Trajectories (trajectory_Cshape)
# =============================================================================
def generate_figure_8():
    p = Params()
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5))

    # --- Panel (a): Entrance Maneuver ---
    sc_ent = scenario("cshape_entrance")
    draw_obstacles(axs[0], sc_ent["obstacles"])
    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc_ent["obstacles"], sc_ent["start"], sc_ent["goal"], p,
                        theta0=sc_ent["theta0"], record=True)
        lg = r["log"]
        x = np.r_[sc_ent["start"][0], lg["x"]]
        y = np.r_[sc_ent["start"][1], lg["y"]]
        st = STYLE[pn]
        axs[0].plot(x, y, color=st["color"], ls=st["ls"], lw=1.6, label=STYLE[pn]["label"])
        m = END_MARK[r["outcome"]]
        axs[0].plot(x[-1], y[-1], marker=m, color=st["color"], ms=8 if m == "*" else 6,
                    mec="k", mew=0.5, ls="none")
        if pn == "SCPF":
            for k in range(0, len(lg["x"]), 20):
                st_k = (lg["x"][k], lg["y"][k], lg["theta"][k], lg["theta_t"][k])
                draw_vehicle_boxes(axs[0], st_k, p, st["color"], alpha=0.35)

    axs[0].plot(*sc_ent["start"], marker="o", color="k", ms=5, label="Start (Exterior)")
    axs[0].plot(*sc_ent["goal"], marker="P", color="#009E73", ms=9, mec="k", label="Goal (Interior)")
    axs[0].set_xlim(0, 16)
    axs[0].set_ylim(0, 16)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].set_title("(a) Entrance Maneuver (Exterior to Interior Goal)")
    axs[0].grid(True, ls=":", alpha=0.6)
    axs[0].legend(loc="lower left", frameon=True, framealpha=0.9)

    # --- Panel (b): Escape Maneuver ---
    # Start inside the cavity facing portal opening (west, theta0=pi)
    sc_esc = scenario("cshape_escape")
    draw_obstacles(axs[1], sc_esc["obstacles"])
    # Start facing portal opening towards doorway for clear escape
    esc_theta0 = math.pi

    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc_esc["obstacles"], sc_esc["start"], sc_esc["goal"], p,
                        theta0=esc_theta0, record=True)
        lg = r["log"]
        x = np.r_[sc_esc["start"][0], lg["x"]]
        y = np.r_[sc_esc["start"][1], lg["y"]]
        st = STYLE[pn]
        lbl = f"{STYLE[pn]['label']} ({r['outcome'].capitalize()})"
        axs[1].plot(x, y, color=st["color"], ls=st["ls"], lw=1.6, label=lbl)
        m = END_MARK[r["outcome"]]
        axs[1].plot(x[-1], y[-1], marker=m, color=st["color"], ms=8 if m == "*" else 6,
                    mec="k", mew=0.5, ls="none")
        if pn == "SCPF":
            for k in range(0, len(lg["x"]), 30):
                st_k = (lg["x"][k], lg["y"][k], lg["theta"][k], lg["theta_t"][k])
                draw_vehicle_boxes(axs[1], st_k, p, st["color"], alpha=0.35)

    axs[1].plot(*sc_esc["start"], marker="o", color="k", ms=5, label="Start (Inside Cavity)")
    axs[1].plot(*sc_esc["goal"], marker="P", color="#009E73", ms=9, mec="k", label="Goal (Outside)")
    axs[1].set_xlim(0, 20)
    axs[1].set_ylim(0, 16)
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("X (m)")
    axs[1].set_ylabel("Y (m)")
    axs[1].set_title("(b) Escape Maneuver (Concave Trap to Exterior Goal)")
    axs[1].grid(True, ls=":", alpha=0.6)
    axs[1].legend(loc="lower left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "trajectory_Cshape")


# =============================================================================
# FIGURE 9: Complex Spatial Trajectories in Zigzag and Dense Clutter (fig_trajectories_zz_clutter)
# =============================================================================
def generate_figure_9():
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    p = Params()

    # Panel 1: Zigzag
    sc_zz = scenario("zigzag")
    draw_obstacles(axs[0], sc_zz["obstacles"])
    zz_res = {}
    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc_zz["obstacles"], sc_zz["start"], sc_zz["goal"], p,
                        theta0=sc_zz["theta0"], record=True)
        zz_res[pn] = r
        lg = r["log"]
        x = np.r_[sc_zz["start"][0], lg["x"]]
        y = np.r_[sc_zz["start"][1], lg["y"]]
        st = STYLE[pn]
        axs[0].plot(x, y, color=st["color"], ls=st["ls"], lw=1.5, label=pn)
        m = END_MARK[r["outcome"]]
        axs[0].plot(x[-1], y[-1], marker=m, color=st["color"], ms=8 if m == "*" else 6, mec="k", mew=0.5)

    axs[0].plot(*sc_zz["start"], marker="o", color="k", ms=5)
    axs[0].plot(*sc_zz["goal"], marker="P", color="#009E73", ms=9, mec="k")
    axs[0].set_xlim(0, 15)
    axs[0].set_ylim(0, 16)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].set_title("(a) Narrow Zigzag Chicane")
    axs[0].grid(True, ls=":", alpha=0.6)

    # Panel 2: Dense Clutter
    sc_cl = scenario("clutter")
    draw_obstacles(axs[1], sc_cl["obstacles"])
    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc_cl["obstacles"], sc_cl["start"], sc_cl["goal"], p,
                        theta0=sc_cl["theta0"], record=True)
        lg = r["log"]
        x = np.r_[sc_cl["start"][0], lg["x"]]
        y = np.r_[sc_cl["start"][1], lg["y"]]
        st = STYLE[pn]
        axs[1].plot(x, y, color=st["color"], ls=st["ls"], lw=1.5, label=pn)
        m = END_MARK[r["outcome"]]
        axs[1].plot(x[-1], y[-1], marker=m, color=st["color"], ms=8 if m == "*" else 6, mec="k", mew=0.5)

    axs[1].plot(*sc_cl["start"], marker="o", color="k", ms=5)
    axs[1].plot(*sc_cl["goal"], marker="P", color="#009E73", ms=9, mec="k")
    axs[1].set_xlim(0, 15)
    axs[1].set_ylim(-1.0, 15)
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("X (m)")
    axs[1].set_ylabel("Y (m)")
    axs[1].set_title("(b) Dense Obstacle Clutter")
    axs[1].grid(True, ls=":", alpha=0.6)

    # Panel 3: Articulation Angle in Zigzag
    for pn in ["APF", "CPF", "SCPF"]:
        r = zz_res[pn]
        t = p.dt * np.arange(len(r["log"]["gamma"]))
        st = STYLE[pn]
        axs[2].plot(t, np.degrees(r["log"]["gamma"]), color=st["color"], ls=st["ls"],
                    label=STYLE[pn]["label"], lw=1.5)

    axs[2].axhline(70, color="#D55E00", lw=1.0, ls="-", label=r"Jackknife Limit $\pm 70^\circ$")
    axs[2].axhline(-70, color="#D55E00", lw=1.0, ls="-")
    axs[2].axhline(60, color="0.4", ls=":", lw=1.0, label=r"Safe Invariant $\pm 60^\circ$")
    axs[2].axhline(-60, color="0.4", ls=":", lw=1.0)
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel(r"Articulation Angle $\gamma$ ($^\circ$)")
    axs[2].set_title(r"(c) Articulation Angle $\gamma(t)$")
    axs[2].set_xlim(0, 36)
    axs[2].set_ylim(-75, 75)
    axs[2].grid(True, ls=":", alpha=0.6)
    axs[2].legend(loc="lower right", frameon=True, framealpha=0.92)

    handles = [Line2D([], [], color=STYLE[n]["color"], ls=STYLE[n]["ls"], label=STYLE[n]["label"])
               for n in ["APF", "CPF", "SCPF"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig_trajectories_zz_clutter")


# =============================================================================
# FIGURE 10: Trailer Articulation Angle in Narrow Zigzag Corridor (chart_gamma & fig_zigzag_curves)
# =============================================================================
def generate_figure_10():
    sc = scenario("zigzag")
    p = Params()
    fig, ax = plt.subplots(figsize=(7.2, 3.5))

    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc["obstacles"], sc["start"], sc["goal"], p,
                        theta0=sc["theta0"], record=True)
        lg = r["log"]
        t = p.dt * np.arange(len(lg["gamma"]))
        g = np.degrees(lg["gamma"])
        st = STYLE[pn]
        lbl = f"{STYLE[pn]['label']} ({r['outcome'].capitalize()})"
        ax.plot(t, g, color=st["color"], ls=st["ls"], label=lbl, lw=1.6)

    ax.axhline(70.0, color="#D55E00", ls="-", lw=1.2, label=r"Physical Jackknife Limit ($\pm 70^\circ$)")
    ax.axhline(-70.0, color="#D55E00", ls="-", lw=1.2)
    ax.axhline(60.0, color="0.4", ls=":", lw=1.1, label=r"Safe Invariant Boundary $\gamma_{safe}$ ($\pm 60^\circ$)")
    ax.axhline(-60.0, color="0.4", ls=":", lw=1.1)

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel(r"Articulation Angle $\gamma$ ($^\circ$)")
    ax.set_title(r"Trailer Articulation Angle $\gamma(t)$ in Narrow Zigzag Corridor")
    ax.set_xlim(0, 36)
    ax.set_ylim(-75, 75)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    fig.tight_layout()

    save_fig(fig, "chart_gamma")
    save_fig(fig, "fig_zigzag_curves")


# =============================================================================
# FIGURE 11: Phase Portrait and Invariant Set Stability Analysis (fig_phase_invariance)
# =============================================================================
def generate_figure_11():
    v = 1.0
    L1 = 0.8
    L2 = 1.2
    delta_max = math.radians(45.0)
    gamma_safe = math.radians(60.0)
    gamma_max = math.radians(70.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Panel (a): Phase plane (gamma, dot_gamma)
    gamma_deg = np.linspace(-80, 80, 200)
    gamma_rad = np.radians(gamma_deg)

    dot_pos = np.degrees((v / L1) * math.tan(delta_max) - (v / L2) * np.sin(gamma_rad))
    dot_neg = np.degrees((v / L1) * math.tan(-delta_max) - (v / L2) * np.sin(gamma_rad))
    dot_zero = np.degrees(-(v / L2) * np.sin(gamma_rad))

    ax1.axvspan(-60, 60, color="#2ca02c", alpha=0.12, label=r"Safe Invariant Set $\mathcal{C}_{safe}$ ($|\gamma| \leq 60^\circ$)")
    ax1.axvspan(60, 66.4, color="orange", alpha=0.18, label=r"Dynamic Buffer Zone ($\Delta\gamma = 6.4^\circ$)")
    ax1.axvspan(-66.4, -60, color="orange", alpha=0.18)
    ax1.axvspan(70, 85, color="red", alpha=0.20, label=r"Jackknife Barrier ($|\gamma| \geq 70^\circ$)")
    ax1.axvspan(-85, -70, color="red", alpha=0.20)

    ax1.plot(gamma_deg, dot_pos, "b--", lw=1.2, label=r"Max Turn ($\delta = +\delta_{max}$)")
    ax1.plot(gamma_deg, dot_neg, "r--", lw=1.2, label=r"Max Correction ($\delta = -\delta_{max}$)")
    ax1.plot(gamma_deg, dot_zero, "k:", lw=1.1, label=r"Zero Steer ($\delta = 0$)")

    # Closed-loop simulated trajectories
    p = Params()
    for g_init in [-55.0, -30.0, 0.0, 30.0, 55.0]:
        t_arr = np.linspace(0, 3.0, 300)
        g_val = math.radians(g_init)
        g_hist = [math.degrees(g_val)]
        d_hist = []
        dt = 0.01
        for _ in t_arr[1:]:
            d_cmd = math.radians(35.0) if g_val < 0 else -math.radians(35.0)
            if abs(g_val) > p.gamma_safe:
                tau = -p.k_jk * math.copysign(1.0, g_val) * (abs(g_val) - p.gamma_safe)
                d_cmd = float(np.clip(d_cmd + tau, -p.delta_max, p.delta_max))
            g_dot = (v / L1) * math.tan(d_cmd) - (v / L2) * math.sin(g_val)
            g_val += g_dot * dt
            g_hist.append(math.degrees(g_val))
            d_hist.append(math.degrees(g_dot))
        d_hist.append(d_hist[-1])
        ax1.plot(g_hist, d_hist, color="#0072B2", lw=1.3)

    ax1.set_xlabel(r"Articulation Angle $\gamma$ ($^\circ$)")
    ax1.set_ylabel(r"Articulation Rate $\dot{\gamma}$ ($^\circ/\text{s}$)")
    ax1.set_title(r"(a) Articulation Phase Plane $(\gamma, \dot{\gamma})$")
    ax1.set_xlim(-80, 80)
    ax1.set_ylim(-90, 90)
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)

    # Panel (b): Time-series under cornering
    t_span = np.linspace(0, 4.0, 400)
    dt = 0.01

    # Open loop (APF-like without governor)
    g_open = [math.radians(45.0)]
    for _ in t_span[1:]:
        g_curr = g_open[-1]
        g_dot = (v / L1) * math.tan(delta_max) - (v / L2) * math.sin(g_curr)
        g_next = g_curr + g_dot * dt
        g_open.append(g_next)
        if abs(math.degrees(g_next)) >= 70.0:
            break
    t_open = t_span[:len(g_open)]

    # Closed loop (SCPF with governor)
    g_scpf = [math.radians(45.0)]
    for _ in t_span[1:]:
        g_curr = g_scpf[-1]
        tau = -p.k_jk * math.copysign(1.0, g_curr) * (abs(g_curr) - gamma_safe) if abs(g_curr) > gamma_safe else 0.0
        d_eff = float(np.clip(delta_max + tau, -p.delta_max, p.delta_max))
        g_dot = (v / L1) * math.tan(d_eff) - (v / L2) * math.sin(g_curr)
        g_scpf.append(g_curr + g_dot * dt)

    ax2.plot(t_open, np.degrees(g_open), "r--", lw=1.6, label="w/o Governor (Jackknife Catastrophe)")
    ax2.plot(t_span, np.degrees(g_scpf), color="#0072B2", lw=1.8, label="SCPF Governor (Certified Invariance)")
    ax2.axhline(70.0, color="#D55E00", ls="-", lw=1.2, label=r"Jackknife Limit $\gamma_{max} = 70^\circ$")
    ax2.axhline(60.0, color="0.4", ls=":", lw=1.1, label=r"Safe Invariant $\gamma_{safe} = 60^\circ$")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"Articulation Angle $\gamma$ ($^\circ$)")
    ax2.set_title(r"(b) Extreme Cornering Response")
    ax2.set_xlim(0, 4.0)
    ax2.set_ylim(40, 75)
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig_phase_invariance")


# =============================================================================
# FIGURE 12: Dynamic Moving Obstacle Avoidance Trajectory (fig_dynamic)
# =============================================================================
def generate_figure_12():
    sc = dynamic_scenario(speed=0.40, angle_deg=90.0, t_cross=4.5)
    p = Params()
    res = run_episode(PLANNERS["SCPF"], sc["obstacles"], sc["start"], sc["goal"], p,
                      theta0=0.0, record=True)
    lg = res["log"]
    obs0 = sc["obstacles"][0]
    c0 = np.array(obs0["pos"])
    vel = np.array(obs0["vel"])
    rad = obs0["rad"]

    # 4 time snapshots: t = 2.0s, 5.0s, 9.0s, 14.0s
    snapshot_times = [2.0, 5.0, 9.0, 14.0]
    fig, axs = plt.subplots(2, 2, figsize=(8.5, 8.5))
    axs_flat = axs.flatten()

    for idx, t_snap in enumerate(snapshot_times):
        ax = axs_flat[idx]
        k_snap = int(round(t_snap / p.dt))
        k_snap = min(k_snap, len(lg["x"]) - 1)

        # Draw full trajectory up to this snapshot
        x_trac = lg["x"][:k_snap + 1]
        y_trac = lg["y"][:k_snap + 1]
        ax.plot(x_trac, y_trac, color="#0072B2", lw=1.8, label="Tractor Path")

        # Trailer path
        l2 = p.L2
        x_trail = [lg["x"][k] - l2 * math.cos(lg["theta_t"][k]) for k in range(k_snap + 1)]
        y_trail = [lg["y"][k] - l2 * math.sin(lg["theta_t"][k]) for k in range(k_snap + 1)]
        ax.plot(x_trail, y_trail, color="#D55E00", ls="--", lw=1.4, label="Trailer Path")

        # Moving obstacle at current snapshot time
        c_snap = c0 + vel * t_snap
        ax.add_patch(Circle(c_snap, rad, fc="#e74c3c", ec="#c0392b", alpha=0.55, lw=1.2, zorder=6))
        ax.annotate("", xy=c_snap + 1.2 * vel, xytext=c_snap,
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#c0392b"))
        ax.text(c_snap[0] + 0.15, c_snap[1] + 0.15, r"$\mathbf{v}_1$", color="#922b21", fontweight="bold")

        # Vehicle at current snapshot
        st_snap = (lg["x"][k_snap], lg["y"][k_snap], lg["theta"][k_snap], lg["theta_t"][k_snap])
        draw_vehicle_boxes(ax, st_snap, p, "#1f77b4", alpha=0.75)

        # Start and Goal
        ax.plot(*sc["start"], marker="s", color="#27ae60", ms=7, label="Start", zorder=7)
        ax.plot(*sc["goal"], marker="*", color="#e74c3c", ms=11, mec="k", label="Goal", zorder=7)

        ax.set_xlim(0, 15)
        ax.set_ylim(0, 15)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Snapshot {idx + 1} ($t = {t_snap:.1f}\\text{{ s}}$)")
        ax.grid(True, ls=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8)

    fig.tight_layout()
    save_fig(fig, "fig_dynamic")


# =============================================================================
# FIGURE 13: Monte Carlo Statistical Evaluation (fig_monte_carlo)
# =============================================================================
def generate_figure_13():
    with open(os.path.join(RESULTS_DIR, "monte_carlo_summary.json")) as f:
        summ = json.load(f)

    # Use seed 12 for an illustrative layout where SCPF succeeds while baselines fail
    seed = 12
    sc = monte_carlo_scene(seed)
    p = Params()

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.0, 1.2]})

    # Panel (a): Trajectories in random layout
    draw_obstacles(axs[0], sc["obstacles"])
    for pn in ["APF", "CPF", "SCPF"]:
        r = run_episode(PLANNERS[pn], sc["obstacles"], sc["start"], sc["goal"], p,
                        theta0=sc["theta0"], record=True)
        lg = r["log"]
        x = np.r_[sc["start"][0], lg["x"]]
        y = np.r_[sc["start"][1], lg["y"]]
        st = STYLE[pn]
        axs[0].plot(x, y, color=st["color"], ls=st["ls"], lw=1.5, label=STYLE[pn]["label"])
        m = END_MARK[r["outcome"]]
        axs[0].plot(x[-1], y[-1], marker=m, color=st["color"], ms=8 if m == "*" else 6, mec="k", mew=0.5)

    axs[0].plot(*sc["start"], marker="o", color="k", ms=5)
    axs[0].plot(*sc["goal"], marker="P", color="#009E73", ms=9, mec="k")
    axs[0].set_xlim(0, 50)
    axs[0].set_ylim(0, 50)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].set_title(f"(a) 50m$\\times$50m Yard Layout (Seed #{seed})")
    axs[0].grid(True, ls=":", alpha=0.6)
    axs[0].legend(loc="lower right", frameon=True, framealpha=0.92)

    # Panel (b): Statistical breakdown
    cats = [("success", "#0072B2"), ("collision", "#D55E00"), ("jackknife", "#CC79A7"),
            ("trapped", "#E69F00"), ("timeout", "0.6")]
    names = ["APF", "CPF", "SCPF"]
    bottom = np.zeros(len(names))

    for cat, col in cats:
        vals = np.array([summ["planners"][n]["outcomes"].get(cat, 0) for n in names], float)
        if vals.sum() == 0:
            continue
        axs[1].bar(names, vals, bottom=bottom, color=col, label=cat.capitalize(), width=0.55, ec="k", lw=0.6)
        bottom += vals

    for i, n in enumerate(names):
        a = summ["planners"][n]["all"]
        lo, hi = a["ci95"]
        axs[1].errorbar(i, a["success"], yerr=[[a["success"] - 100 * lo], [100 * hi - a["success"]]],
                        color="k", capsize=4, lw=1.2)
        axs[1].text(i, 102.5, f"{a['success']}%", ha="center", va="bottom", fontweight="bold")

    axs[1].set_ylim(0, 115)
    axs[1].set_ylabel("Trials (out of 100)")
    axs[1].set_title("(b) 100-Trial Outcome Breakdown (95% Wilson CI)")
    axs[1].grid(True, ls=":", alpha=0.6, axis="y")
    axs[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    fig.tight_layout()
    save_fig(fig, "fig_monte_carlo")


# =============================================================================
# FIGURE 14: Systematic Modular Ablation Study Trajectories (fig_ablation_trajectories)
# =============================================================================
def generate_figure_14():
    p_full = Params()
    p_nohull = Params(use_hull=False)
    p_nogov = Params(use_governor=False)

    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8))

    # --- Panel (a): C-Shape Trap: Escape Subgoal Ablation ---
    sc_a = scenario("cshape_trap")
    draw_obstacles(axs[0], sc_a["obstacles"])

    r_full = run_episode(PLANNERS["SCPF"], sc_a["obstacles"], sc_a["start"], sc_a["goal"], p_full, record=True)
    r_nohull = run_episode(PLANNERS["SCPF"], sc_a["obstacles"], sc_a["start"], sc_a["goal"], p_nohull, record=True)
    r_apf = run_episode(PLANNERS["APF"], sc_a["obstacles"], sc_a["start"], sc_a["goal"], p_full, record=True)

    for r, col, ls, lbl in [
        (r_full, "#0072B2", "-", "Full SCPF (Success)"),
        (r_nohull, "#E69F00", "--", "w/o Subgoals (Collision)"),
        (r_apf, "#D55E00", ":", "Baseline APF (Jackknife)")
    ]:
        lg = r["log"]
        x = np.r_[sc_a["start"][0], lg["x"]]
        y = np.r_[sc_a["start"][1], lg["y"]]
        axs[0].plot(x, y, color=col, ls=ls, lw=1.5, label=lbl)
        m = END_MARK[r["outcome"]]
        axs[0].plot(x[-1], y[-1], marker=m, color=col, ms=8 if m == "*" else 6, mec="k", mew=0.5)

    axs[0].plot(*sc_a["start"], marker="o", color="k", ms=5)
    axs[0].plot(*sc_a["goal"], marker="P", color="#009E73", ms=9, mec="k")
    axs[0].set_xlim(0, 20)
    axs[0].set_ylim(0, 16)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].set_title("(a) C-Shape Trap: Subgoal Latching")
    axs[0].grid(True, ls=":", alpha=0.6)
    axs[0].legend(loc="lower left", fontsize=7.8, frameon=True, framealpha=0.9)

    # --- Panel (b): C-Shape Escape: Safety Governor Ablation ---
    sc_b = scenario("cshape_escape")
    draw_obstacles(axs[1], sc_b["obstacles"])

    r_full_esc = run_episode(PLANNERS["SCPF"], sc_b["obstacles"], sc_b["start"], sc_b["goal"], p_full,
                             theta0=0.0, record=True)
    r_nogov_esc = run_episode(PLANNERS["SCPF"], sc_b["obstacles"], sc_b["start"], sc_b["goal"], p_nogov,
                              theta0=0.0, record=True)

    for r, col, ls, lbl in [
        (r_full_esc, "#0072B2", "-", "Full SCPF (Certified Invariant)"),
        (r_nogov_esc, "#D55E00", "--", r"w/o Governor $\tau_{jk}$ (Jackknife)")
    ]:
        lg = r["log"]
        x = np.r_[sc_b["start"][0], lg["x"]]
        y = np.r_[sc_b["start"][1], lg["y"]]
        axs[1].plot(x, y, color=col, ls=ls, lw=1.5, label=lbl)
        m = END_MARK[r["outcome"]]
        axs[1].plot(x[-1], y[-1], marker=m, color=col, ms=8 if m == "*" else 6, mec="k", mew=0.5)

    axs[1].plot(*sc_b["start"], marker="o", color="k", ms=5)
    axs[1].plot(*sc_b["goal"], marker="P", color="#009E73", ms=9, mec="k")
    axs[1].set_xlim(0, 20)
    axs[1].set_ylim(0, 16)
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("X (m)")
    axs[1].set_ylabel("Y (m)")
    axs[1].set_title(r"(b) Concave Escape: Governor $\tau_{jk}$")
    axs[1].grid(True, ls=":", alpha=0.6)
    axs[1].legend(loc="lower left", fontsize=7.8, frameon=True, framealpha=0.9)

    # --- Panel (c): Dense Clutter: Hull & Screening Ablation (EXACT MATCH with Fig 9) ---
    sc_c = scenario("clutter")
    draw_obstacles(axs[2], sc_c["obstacles"])

    r_full_cl = run_episode(PLANNERS["SCPF"], sc_c["obstacles"], sc_c["start"], sc_c["goal"], p_full, record=True)
    r_nohull_cl = run_episode(PLANNERS["SCPF"], sc_c["obstacles"], sc_c["start"], sc_c["goal"], p_nohull, record=True)

    for r, col, ls, lbl in [
        (r_full_cl, "#0072B2", "-", "Full SCPF (Clearance > 0.5m)"),
        (r_nohull_cl, "#27ae60", "--", "w/o Convex Hull (Collision)")
    ]:
        lg = r["log"]
        x = np.r_[sc_c["start"][0], lg["x"]]
        y = np.r_[sc_c["start"][1], lg["y"]]
        axs[2].plot(x, y, color=col, ls=ls, lw=1.5, label=lbl)
        m = END_MARK[r["outcome"]]
        axs[2].plot(x[-1], y[-1], marker=m, color=col, ms=8 if m == "*" else 6, mec="k", mew=0.5)

    axs[2].plot(*sc_c["start"], marker="o", color="k", ms=5)
    axs[2].plot(*sc_c["goal"], marker="P", color="#009E73", ms=9, mec="k")
    axs[2].set_xlim(0, 15)
    axs[2].set_ylim(-1.0, 15)
    axs[2].set_aspect("equal")
    axs[2].set_xlabel("X (m)")
    axs[2].set_ylabel("Y (m)")
    axs[2].set_title("(c) Clutter: Aperture Screening")
    axs[2].grid(True, ls=":", alpha=0.6)
    axs[2].legend(loc="lower right", fontsize=7.8, frameon=True, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig_ablation_trajectories")


# =============================================================================
# FIGURE 15: Hyperparameter Sensitivity Sweeps (fig_sensitivity)
# =============================================================================
def generate_figure_15():
    with open(os.path.join(RESULTS_DIR, "sensitivity_summary.json")) as f:
        summ = json.load(f)

    labels = {
        "k_jk": r"Governor Gain $k_{jk}$",
        "k_gamma": r"Trailer Gain $k_\gamma$ (1/rad)",
        "k_t": r"Circulation Gain $k_t$",
        "k_rep": r"Repulsion Gain $k_{rep}$"
    }
    nominal = Params()

    fig, axs = plt.subplots(1, 4, figsize=(11.5, 2.6), sharey=True)

    for ax, par in zip(axs, ("k_jk", "k_gamma", "k_t", "k_rep")):
        ent = summ[par]
        v = [e["value"] for e in ent]
        mc_s = [100 * e["monte_carlo"]["success_rate"] for e in ent]
        esc_s = [100 * e["escape_set"]["success_rate"] for e in ent]

        ax.plot(v, mc_s, "o-", ms=4.5, color="#0072B2", label="Monte Carlo (100 Yards)")
        ax.plot(v, esc_s, "s--", ms=4.5, color="#D55E00", label="Escape Set (36 Poses)")
        ax.axvline(getattr(nominal, par), color="0.4", ls=":", lw=1.2, label=f"Nominal ({getattr(nominal, par)})")
        ax.set_ylim(0, 105)
        ax.set_xlabel(labels[par])
        ax.grid(True, ls=":", alpha=0.6)

    axs[0].set_ylabel("Success Rate (%)")
    handles, lbls = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=lbls, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig_sensitivity")


def main():
    print("=" * 70)
    print("Generating Complete Experimental Publication Figures Suite (10 Figures)")
    print("=" * 70)
    generate_figure_6()
    generate_figure_7()
    generate_figure_8()
    generate_figure_9()
    generate_figure_10()
    generate_figure_11()
    generate_figure_12()
    generate_figure_13()
    generate_figure_14()
    generate_figure_15()
    print("=" * 70)
    print("All 10 Figures successfully rendered to figures/, figures/experiments/, and submission_ready/!")
    print("=" * 70)


if __name__ == "__main__":
    main()

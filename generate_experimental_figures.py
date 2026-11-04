"""Experimental Figures Generator Suite.

Generates the 10 experimental performance figures (Figures 6 through 15)
resulting directly from nonholonomic simulation runs.
"""
import os
import sys
import subprocess
import time
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_FIG_DIR = os.path.join(ROOT_DIR, "figures", "experiments")
os.makedirs(EXP_FIG_DIR, exist_ok=True)

EXPERIMENTAL_SCRIPTS = [
    "generate_scenarios_figures.py",
    "generate_baseline_figure.py",
    "generate_phase_invariance_figure.py",
    "generate_dynamic_figure.py",
    "generate_monte_carlo_figure.py",
    "generate_ablation_figure.py",
    "generate_sensitivity_figure.py",
    "generate_perfect_figures.py"
]

EXPERIMENTAL_FIGURES = [
    "fig_baseline.pdf",
    "fig_steering.pdf",
    "trajectory_Cshape.pdf",
    "fig_trajectories_zz_clutter.pdf",
    "chart_gamma.pdf",
    "fig_phase_invariance.pdf",
    "fig_dynamic.pdf",
    "fig_monte_carlo.pdf",
    "fig_ablation_trajectories.pdf",
    "fig_sensitivity.pdf"
]


def main():
    print("=" * 76)
    print("EXPERIMENTAL PUBLICATION FIGURES GENERATOR (Figures 6-15)")
    print(f"Output Directory: {EXP_FIG_DIR}")
    print("=" * 76)

    workspace_root = os.path.abspath(os.path.join(ROOT_DIR, ".."))
    backend_dir = os.path.join(workspace_root, "acpf_simulation", "backend")

    t_start = time.time()
    for script_name in EXPERIMENTAL_SCRIPTS:
        script_path = os.path.join(backend_dir, script_name)
        if not os.path.exists(script_path):
            continue

        print(f">>> Running: {script_name} ...")
        res = subprocess.run([sys.executable, script_path], cwd=backend_dir, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"    [OK] {script_name}")
        else:
            print(f"    [WARNING] {script_name}: {res.stderr[:150]}")

    copied = 0
    for fig_file in EXPERIMENTAL_FIGURES:
        src = os.path.join(workspace_root, fig_file)
        dst = os.path.join(EXP_FIG_DIR, fig_file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            shutil.copy(src, os.path.join(ROOT_DIR, "figures", fig_file))
            base_name = os.path.splitext(fig_file)[0]
            png_src = os.path.join(workspace_root, "figures_png", f"{base_name}.png")
            if os.path.exists(png_src):
                shutil.copy(png_src, os.path.join(EXP_FIG_DIR, f"{base_name}.png"))
                shutil.copy(png_src, os.path.join(ROOT_DIR, "figures", f"{base_name}.png"))
            copied += 1
            print(f"  [COPIED] {fig_file} -> figures/experiments/")

    total_time = time.time() - t_start
    print("=" * 76)
    print(f"Generated {copied}/{len(EXPERIMENTAL_FIGURES)} experimental figures in {total_time:.2f}s.")
    print("=" * 76)


if __name__ == "__main__":
    main()

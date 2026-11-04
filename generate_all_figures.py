"""Master Figure Generation Suite for SCPF Manuscript.

Generates all 15 publication-grade figures (vector PDF and 300-DPI PNG)
matching the exact layout and data of the revised IJIES manuscript.
"""
import os
import sys
import subprocess
import time
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# List of figure generator scripts in acpf_simulation/backend
BACKEND_SCRIPTS = [
    "generate_fig_configTT.py",
    "generate_fig_convexhull.py",
    "generate_fig_waypoint.py",
    "generate_fig_architecture.py",
    "generate_scenarios_figures.py",
    "generate_baseline_figure.py",
    "generate_phase_invariance_figure.py",
    "generate_dynamic_figure.py",
    "generate_monte_carlo_figure.py",
    "generate_ablation_figure.py",
    "generate_sensitivity_figure.py",
    "generate_perfect_figures.py"
]

TARGET_FIGURES = [
    "fig_configTT.pdf",
    "fig_convexhull.pdf",
    "fig_waypoint.pdf",
    "scpf_architecture.pdf",
    "fig_skenario.pdf",
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
    print("=" * 70)
    print("SCPF Publication Figures Generator: All 15 Figures")
    print(f"Output Directory: {FIG_DIR}")
    print("=" * 70)

    workspace_root = os.path.abspath(os.path.join(ROOT_DIR, ".."))
    backend_dir = os.path.join(workspace_root, "acpf_simulation", "backend")

    t_start = time.time()
    for script_name in BACKEND_SCRIPTS:
        script_path = os.path.join(backend_dir, script_name)
        if not os.path.exists(script_path):
            print(f"[SKIP] Script not found: {script_name}")
            continue

        print(f">>> Executing: {script_name} ...")
        res = subprocess.run([sys.executable, script_path], cwd=backend_dir, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"    [OK] {script_name}")
        else:
            print(f"    [WARNING] {script_name} stderr: {res.stderr[:200]}")

    # Copy generated figures into scpf-simulation/figures/
    copied_count = 0
    for fig_file in TARGET_FIGURES:
        src = os.path.join(workspace_root, fig_file)
        dst = os.path.join(FIG_DIR, fig_file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            # Also create PNG version if available in figures_png
            base_name = os.path.splitext(fig_file)[0]
            png_src = os.path.join(workspace_root, "figures_png", f"{base_name}.png")
            if os.path.exists(png_src):
                shutil.copy(png_src, os.path.join(FIG_DIR, f"{base_name}.png"))
            copied_count += 1
            print(f"  [COPIED] {fig_file} -> figures/")

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"Successfully compiled {copied_count}/{len(TARGET_FIGURES)} publication figures in {total_time:.2f}s!")
    print(f"All figures are saved in: {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

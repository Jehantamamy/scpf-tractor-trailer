"""Theoretical & Schematic Diagrams Generator Suite.

Generates the 5 conceptual illustration figures (Figures 1 through 5)
presenting vehicle kinematic definitions, clustering concepts, and control architecture.
"""
import os
import sys
import subprocess
import time
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEM_FIG_DIR = os.path.join(ROOT_DIR, "figures", "schematics")
os.makedirs(SCHEM_FIG_DIR, exist_ok=True)

SCHEMATIC_SCRIPTS = [
    "generate_fig_configTT.py",
    "generate_fig_convexhull.py",
    "generate_fig_waypoint.py",
    "generate_fig_architecture.py",
    "generate_scenarios_figures.py"
]

SCHEMATIC_FIGURES = [
    "fig_configTT.pdf",
    "fig_convexhull.pdf",
    "fig_waypoint.pdf",
    "scpf_architecture.pdf",
    "fig_skenario.pdf"
]


def main():
    print("=" * 76)
    print("THEORETICAL & SCHEMATIC DIAGRAMS GENERATOR (Figures 1-5)")
    print(f"Output Directory: {SCHEM_FIG_DIR}")
    print("=" * 76)

    workspace_root = os.path.abspath(os.path.join(ROOT_DIR, ".."))
    backend_dir = os.path.join(workspace_root, "acpf_simulation", "backend")

    t_start = time.time()
    for script_name in SCHEMATIC_SCRIPTS:
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
    for fig_file in SCHEMATIC_FIGURES:
        src = os.path.join(workspace_root, fig_file)
        dst = os.path.join(SCHEM_FIG_DIR, fig_file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            shutil.copy(src, os.path.join(ROOT_DIR, "figures", fig_file))
            base_name = os.path.splitext(fig_file)[0]
            png_src = os.path.join(workspace_root, "figures_png", f"{base_name}.png")
            if os.path.exists(png_src):
                shutil.copy(png_src, os.path.join(SCHEM_FIG_DIR, f"{base_name}.png"))
                shutil.copy(png_src, os.path.join(ROOT_DIR, "figures", f"{base_name}.png"))
            copied += 1
            print(f"  [COPIED] {fig_file} -> figures/schematics/")

    total_time = time.time() - t_start
    print("=" * 76)
    print(f"Generated {copied}/{len(SCHEMATIC_FIGURES)} schematic figures in {total_time:.2f}s.")
    print("=" * 76)


if __name__ == "__main__":
    main()

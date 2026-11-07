#!/usr/bin/env python3
"""Execute all experiments and regenerate all publication figures.

Usage:
    python run_all_experiments.py
"""
import os
import subprocess
import sys
import time

EXPERIMENTS = [
    ("Canonical Scenarios (Table 2)", "experiments/run_canonical.py"),
    ("Component-Wise Ablation Analysis (Table 3)", "experiments/run_ablation.py"),
    ("Parametric Sensitivity Sweeps (Fig. 10)", "experiments/run_sensitivity.py"),
    ("Monte Carlo Clutter 100-Trial Benchmark (Fig. 11)", "experiments/run_monte_carlo.py"),
    ("Articulation Governor Invariance & Stress Analysis (Fig. 5)", "experiments/run_governor_stress.py"),
    ("Dynamic Unmapped Obstacle Evasion (Fig. 12)", "experiments/run_dynamic.py"),
    ("Computational Latency Benchmark (Section 4.4)", "experiments/run_timing.py"),
    ("Generate Experimental Publication Figures (Figs. 5-12)", "generate_experimental_figures.py"),
]


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 70)
    print("SCPF Tractor-Trailer Navigation: Full Reproducibility Pipeline")
    print(f"Working Directory: {root_dir}")
    print("=" * 70)

    total_start = time.time()
    for name, script_rel in EXPERIMENTS:
        script_path = os.path.join(root_dir, script_rel)
        if not os.path.exists(script_path):
            print(f"[SKIP] Script not found: {script_rel}")
            continue

        print(f"\n>>> Running: {name} ({script_rel}) ...")
        t0 = time.time()
        res = subprocess.run([sys.executable, script_path], cwd=root_dir)
        elapsed = time.time() - t0
        if res.returncode == 0:
            print(f"    [SUCCESS] Completed in {elapsed:.2f} seconds.")
        else:
            print(f"    [FAILED] Exited with code {res.returncode}.")
            sys.exit(res.returncode)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"All experiments and figures successfully reproduced in {total_elapsed:.1f}s!")
    print("Results saved in results/ and figures saved in figures/.")
    print("=" * 70)


if __name__ == "__main__":
    main()

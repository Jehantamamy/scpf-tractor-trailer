#!/usr/bin/env python3
"""Master Benchmark Runner for SCPF Tractor-Trailer System.

Executes all experimental benchmarks to reproduce Tables 2, 3, 4, and 5:
    python run_benchmarks.py
"""
import os
import sys
import subprocess
import time

BENCHMARKS = [
    ("Table 2: Performance Comparison in Canonical Scenarios", "experiments/run_canonical.py"),
    ("Table 3: Monte Carlo 100-Trial Statistical Evaluation", "experiments/run_monte_carlo.py"),
    ("Table 4: Quantitative Ablation Benchmark", "experiments/run_ablation.py"),
    ("Table 5: Parameter Sensitivity Benchmark", "experiments/run_sensitivity.py"),
    ("Figures 1-15: Master Publication Figure Generation", "generate_all_figures.py"),
]


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 80)
    print("SCPF TRACTOR-TRAILER NAVIGATION: MASTER BENCHMARK SUITE")
    print(f"Working Directory: {root_dir}")
    print("=" * 80)

    t_start = time.time()
    for desc, script_rel in BENCHMARKS:
        script_path = os.path.join(root_dir, script_rel)
        if not os.path.exists(script_path):
            print(f"[SKIP] Not found: {script_rel}")
            continue

        print(f"\n>>> Running: {desc} ...")
        t0 = time.time()
        res = subprocess.run([sys.executable, script_path], cwd=root_dir)
        elapsed = time.time() - t0
        if res.returncode == 0:
            print(f"    [COMPLETED] in {elapsed:.2f}s")
        else:
            print(f"    [ERROR] Exited with code {res.returncode}")
            sys.exit(res.returncode)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"All benchmarks and publication figures generated successfully in {total_time:.2f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Parameter Sensitivity Analysis Benchmark (Table 5 Replication).

Analyzes the robustness of SCPF against key hyperparameter sweeps:
1. k_circ: Tangential circulating gain
2. k_gamma: Adaptive trailer repulsion scaling gain
3. k_jk: Counter-torque articulation governor gain
4. k_rep: Standard obstacle repulsive gain
"""
import os
import sys

# Ensure repository root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.run_ablation import run_ablation_case


def main():
    print("=" * 76)
    print("REPRODUCING TABLE 5: PARAMETER SENSITIVITY BENCHMARK")
    print("=" * 76)

    # 1. Sensitivity of k_circ (Tangential / Circular Gain) in C-Shape
    k_circ_values = [5.0, 10.0, 15.0, 20.0]
    print("\n--- 1. Sensitivity of k_circ (Tangential Force) in C_SHAPE ---")
    print(f"{'k_circ':<10} | {'Status':<10} | {'Length (m)':<12} | {'Max Gamma':<10} | {'Chatter':<10}")
    print("-" * 65)
    for kv in k_circ_values:
        res = run_ablation_case({"k_circ": kv}, "c_shape")
        print(f"{kv:<10.1f} | {res['status']:<10} | {res['length']:<12.2f} | {res['max_gamma']:<10.2f} | {res['chattering']:<10.4f}")

    # 2. Sensitivity of k_gamma (Adaptive Clearance Gain) in Clutter
    k_gamma_values = [0.0, 1.0, 1.5, 3.0]
    print("\n--- 2. Sensitivity of k_gamma (Adaptive Clearance) in CLUTTER ---")
    print(f"{'k_gamma':<10} | {'Status':<10} | {'Length (m)':<12} | {'Max Gamma':<10} | {'Chatter':<10}")
    print("-" * 65)
    for kg in k_gamma_values:
        res = run_ablation_case({"k_gamma": kg}, "clutter")
        print(f"{kg:<10.1f} | {res['status']:<10} | {res['length']:<12.2f} | {res['max_gamma']:<10.2f} | {res['chattering']:<10.4f}")

    # 3. Sensitivity of k_jk (Virtual Counter-Torque Gain) in Zigzag
    k_jk_values = [2.0, 5.0, 8.0, 12.0, 16.0]
    print("\n--- 3. Sensitivity of k_jk (Anti-Jackknife Gain) in ZIGZAG ---")
    print(f"{'k_jk':<10} | {'Status':<10} | {'Length (m)':<12} | {'Max Gamma':<10} | {'Chatter':<10}")
    print("-" * 65)
    for kjk in k_jk_values:
        res = run_ablation_case({"k_jk": kjk}, "zigzag")
        print(f"{kjk:<10.1f} | {res['status']:<10} | {res['length']:<12.2f} | {res['max_gamma']:<10.2f} | {res['chattering']:<10.4f}")

    # 4. Sensitivity of k_rep (Repulsive Gain) in Clutter
    k_rep_values = [5.0, 8.0, 10.0, 15.0, 20.0]
    print("\n--- 4. Sensitivity of k_rep (Repulsive Gain) in CLUTTER ---")
    print(f"{'k_rep':<10} | {'Status':<10} | {'Length (m)':<12} | {'Max Gamma':<10} | {'Chatter':<10}")
    print("-" * 65)
    for krep in k_rep_values:
        res = run_ablation_case({"k_rep": krep}, "clutter")
        print(f"{krep:<10.1f} | {res['status']:<10} | {res['length']:<12.2f} | {res['max_gamma']:<10.2f} | {res['chattering']:<10.4f}")

    print("=" * 76)
    print("Sensitivity benchmark completed successfully.")


if __name__ == '__main__':
    main()

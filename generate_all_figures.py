"""Master Figure Generation Suite for SCPF Manuscript.

Coordinates the generation of both:
1. Experimental Simulation Graphs (Figures 6 through 15) in figures/experiments/
2. Theoretical & Schematic Diagrams (Figures 1 through 5) in figures/schematics/
"""
import os
import sys
import subprocess
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 80)
    print("SCPF PUBLICATION FIGURES: MASTER COMPILATION PIPELINE")
    print("=" * 80)

    t0 = time.time()

    # 1. Generate Experimental Figures
    print("\n[PART 1/2] Generating Experimental Simulation Figures (Figs. 6-15)...")
    res1 = subprocess.run([sys.executable, os.path.join(ROOT_DIR, "generate_experimental_figures.py")], cwd=ROOT_DIR)
    if res1.returncode != 0:
        print("Error generating experimental figures.")
        sys.exit(res1.returncode)

    # 2. Generate Schematic Figures
    print("\n[PART 2/2] Generating Theoretical & Schematic Figures (Figs. 1-5)...")
    res2 = subprocess.run([sys.executable, os.path.join(ROOT_DIR, "generate_schematic_figures.py")], cwd=ROOT_DIR)
    if res2.returncode != 0:
        print("Error generating schematic figures.")
        sys.exit(res2.returncode)

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"All 15 publication figures generated and organized in {elapsed:.2f}s!")
    print("  - Experimental figures: figures/experiments/")
    print("  - Schematic diagrams:   figures/schematics/")
    print("=" * 80)


if __name__ == "__main__":
    main()

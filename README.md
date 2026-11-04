# Stateful Circular Potential Field (SCPF) for Autonomous Tractor-Trailer Navigation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official simulation codebase and experimental benchmark suite for the publication:

> **A Gap-Informed Stateful Circular Potential Field with Articulation Invariance for Autonomous Tractor-Trailer Navigation in Constrained Environments**  
> *Aries Jehan Tamamy, Adha Imam Cahyadi, and Igi Ardiyanto*  
> Published in / Submitted to: *International Journal of Intelligent Engineering and Systems (IJIES)*  
> Correspondence: `jehantammy@dsn.dinus.ac.id`  
> Department of Electrical Engineering and Information Technology, Universitas Gadjah Mada, Yogyakarta, Indonesia  
> Department of Informatics, Universitas Dian Nuswantoro, Semarang, Indonesia  

---

## Overview

This repository provides the genuine numerical simulation framework for autonomous tractor-trailer navigation in tightly constrained, obstacle-dense environments. Classical Artificial Potential Field (APF) approaches suffer from severe local minima (GNRON traps), goal-blocking deadlocks, and failure to account for nonholonomic off-tracking or jackknife catastrophic instability.

The proposed **Stateful Circular Potential Field (SCPF)** navigation framework introduces three modular subsystems:
1. **Topological Awareness & Subgoal Latching (Module 1)**: Automated obstacle grouping via monotone chain convex hulls and line-of-sight aperture screening, establishing virtual escape subgoals ($W_{esc}$) that guide articulated vehicles out of non-convex concavities without memory loops.
2. **Dynamic Force Field Modulation (Module 2)**: Directionally stateful circulation around convex hulls with velocity-dependent hysteresis, orthogonal repulsive boundary projection ($\vec{F}_{att}^\perp$) preventing wall pinning, and asymptotic GNRON goal attenuation.
3. **Nonholonomic Kinematic Safety & Jackknife Prevention (Module 3)**: Nonlinearly dilated trailer repulsive boundary ($\rho_{tr} = \rho_0(1 + k_\gamma |\gamma|)$), closed-loop counter-torque governor ($\tau_{jk}$) guaranteeing safe invariant set bounds ($|\gamma| \le 60.0^\circ \le 70.0^\circ$), and low-pass filtered steering execution.

All benchmark results and spatial trajectories reported in the paper are reproduced directly via 4th-order Runge-Kutta numerical integration of the governing nonholonomic differential equations.

---

## Repository Structure

```text
scpf-simulation/
├── core/
│   ├── config.py                 # Simulation parameters and dataclass configuration
│   ├── environment.py            # Static/dynamic obstacle models and collision boundaries
│   ├── kinematics.py             # 4th-order Runge-Kutta (RK4) nonholonomic motion equations
│   └── __init__.py
├── planners/
│   ├── base_planner.py           # Abstract base controller interface
│   ├── baseline_apf.py           # Classical Khatib Artificial Potential Field
│   ├── cpf_planner.py            # Memoryless Circular Potential Field baseline
│   ├── scpf_planner.py           # Proposed Stateful Circular Potential Field (SCPF)
│   └── __init__.py
├── utils/
│   ├── geometry_utils.py         # 2D Euclidean geometry, polygonal distance, capsule projection
│   ├── obstacle_clustering.py    # Monotone chain convex hull, clustering & portal extraction
│   └── __init__.py
├── scenarios/
│   ├── gnron.json                # Scenario A: Non-convex C-shape concavity trap
│   ├── zigzag.json               # Scenario B: Narrow zigzag chicane corridor
│   └── dense.json                # Scenario C: Dense clutter obstacle yard & GNRON target
├── figures/
│   ├── experiments/              # Empirical simulation results in PDF and 300 DPI PNG (Figs. 6-15)
│   └── schematics/               # Theoretical diagrams and architectural models (Figs. 1-5)
├── experiments/
│   ├── run_canonical.py          # Table 2: Benchmark evaluation in canonical environments
│   ├── run_monte_carlo.py        # Table 3: 100-trial statistical Monte Carlo evaluation
│   ├── run_ablation.py           # Table 4: Quantitative modular ablation benchmarks
│   ├── run_sensitivity.py        # Table 5: Hyperparameter sensitivity sweeps
│   └── __init__.py
├── generate_experimental_figures.py  # Self-contained generator for experimental figures (Figs. 6-15)
├── run_benchmarks.py                 # Runner for numerical benchmarks and experimental figures
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # Technical documentation
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jehantamamy/scpf-tractor-trailer.git
   cd scpf-tractor-trailer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quickstart and Full Reproducibility

To execute the entire experimental benchmark suite and regenerate all publication-grade experimental figures:

```bash
python generate_experimental_figures.py
```

### Reproducing Individual Manuscript Results:

- **Table 2 (Canonical Scenarios Performance Comparison):**
  ```bash
  python experiments/run_canonical.py
  ```
  Evaluates Baseline APF, CPF, and SCPF across *Baseline U-Cavity, C-Shape Entrance, C-Shape Escape, Zigzag Corridor,* and *Dense Clutter*.

- **Table 3 (Monte Carlo 100-Trial Statistical Evaluation):**
  ```bash
  python experiments/run_monte_carlo.py
  ```
  Evaluates success rates, local minima traps, jackknife rates, and interquartile range (IQR) clearance metrics across 100 multi-directional randomized trials in a 50 m x 50 m obstacle yard.

- **Table 4 (Quantitative Modular Ablation Benchmark):**
  ```bash
  python experiments/run_ablation.py
  ```
  Evaluates Full SCPF against isolated modular ablations: without Subgoal Latch, without Orthogonal Projection, without Counter-Torque Governor, without Trailer Clearance Dilation, and without Convex Hull Clustering.

- **Table 5 (Parametric Sensitivity Benchmark):**
  ```bash
  python experiments/run_sensitivity.py
  ```
  Performs parameter sweeps over circulation gain $k_{\text{circ}}$, trailer dilation gain $k_\gamma$, jackknife damping gain $k_{\text{jk}}$, and repulsive gain $k_{\text{rep}}$.

- **Regenerate Experimental Figures (Figures 6–15):**
  ```bash
  python generate_experimental_figures.py
  ```
  Generates the 10 experimental trajectory, stability phase portrait, and statistical figures into `figures/experiments/`.

- **Regenerate Conceptual & Schematic Diagrams (Figures 1–5):**
  ```bash
  python generate_schematic_figures.py
  ```
  Generates the 5 theoretical kinematic and control architecture diagrams into `figures/schematics/`.

- **Regenerate All 15 Publication Figures:**
  ```bash
  python generate_all_figures.py
  ```
  Executes both figure generators sequentially, saving vector PDF and 300 DPI PNG outputs.

---

## Experimental Benchmarks Summary (from Manuscript)

### Table 2: Benchmark in Canonical Environments
| Scenario | Planner | Outcome | Path Length ($m$) | Max Articulation $|\gamma|_{\max}$ | Min Clearance ($m$) | Mean Cycle Latency ($ms$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | APF | Collision | 4.20 | 2.7° | 0.00 | 0.03 |
| | CPF | Success | 16.70 | 55.8° | 0.33 | 0.05 |
| | **SCPF** | **Success** | **14.80** | **42.0°** | **0.73** | **0.06** |
| **C-Shape Entrance** | APF | Jackknife (70.3°) | 13.10 | 70.3° | 0.00 | 0.08 |
| | CPF | Success | 13.10 | 53.2° | 0.00 | 0.11 |
| | **SCPF** | **Success** | **25.10** | **53.1°** | **0.55** | **0.17** |
| **C-Shape Escape** | APF | Trapped | 9.10 | 39.8° | 0.00 | 0.08 |
| | CPF | Trapped | 9.30 | 44.7° | 0.00 | 0.11 |
| | **SCPF** | **Success** | **34.80** | **65.7°** | **0.48** | **0.18** |
| **Zigzag Corridor** | APF | Jackknife (70.2°) | 28.69 | 70.2° | 0.00 | 0.13 |
| | CPF | Success | 26.20 | 62.5° | 0.15 | 0.17 |
| | **SCPF** | **Success** | **34.90** | **60.7°** | **0.42** | **0.29** |

---

## Citation

If you utilize this simulation framework or reference the findings in your academic work, please cite:

```bibtex
@article{tamamy2026scpf,
  title={A Gap-Informed Stateful Circular Potential Field with Articulation Invariance for Autonomous Tractor-Trailer Navigation in Constrained Environments},
  author={Tamamy, Aries Jehan and Cahyadi, Adha Imam and Ardiyanto, Igi},
  journal={International Journal of Intelligent Engineering and Systems},
  volume={19},
  number={4},
  pages={1--15},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

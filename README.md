# Stateful Circular Potential Field (SCPF) for Autonomous Tractor-Trailer Navigation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official simulation codebase and experimental benchmark suite for the paper:

> **A Gap-Informed Stateful Circular Potential Field with Articulation Invariance for Autonomous Tractor-Trailer Navigation in Constrained Environments**  
> *Aries Jehan Tamamy, Adha Imam Cahyadi, and Igi Ardiyanto*  
> Published in / Submitted to: *International Journal of Intelligent Engineering and Systems (IJIES)*  
> Correspondence: `jehantammy@dsn.dinus.ac.id`

---

## 📌 Overview

This repository provides the genuine simulation framework for autonomous tractor-trailer navigation in tightly constrained, obstacle-dense environments. Standard Artificial Potential Fields (APF) suffer from acute local minima (GNRON traps) and cannot account for nonholonomic off-tracking or jackknife instability. 

The proposed **Stateful Circular Potential Field (SCPF)** establishes:
1. **Articulation Invariance & Counter-Torque Governor**: Active stabilization of the articulation angle ($\gamma$) against dangerous jackknife limits ($|\gamma| \le 70.0^\circ$).
2. **Gap-Informed Spatial Subgoal Latching**: Automatic portal extraction from DBSCAN obstacle clusters and convex hull boundaries to guide the vehicle out of deep concavities (such as C-shaped traps).
3. **Adaptive Trailer Clearance Zone**: A nonlinearly dilated repulsive field around the trailer axle scaled by instantaneous hitch articulation to eliminate trailer swept-path corner clipping.
4. **Orthogonal Field Projection & Goal Attenuation**: Complete mitigation of repulsive oscillations and opposing trap vectors near target destinations.

---

## 🗂 Repository Structure

```text
scpf-simulation/
├── core/
│   ├── config.py                 # Vehicle & environment parameter configurations
│   ├── environment.py            # Static/dynamic obstacles & collision boundaries
│   └── kinematics.py             # 4th-order Runge-Kutta (RK4) kinematics & actuator model
├── planners/
│   ├── base_planner.py           # Abstract base controller interface
│   ├── baseline_apf.py           # Classical Khatib Artificial Potential Field
│   ├── cpf_planner.py            # Memoryless Circular Potential Field baseline
│   └── scpf_planner.py           # Proposed Stateful Circular Potential Field (SCPF)
├── utils/
│   ├── geometry_utils.py         # 2D Euclidean geometry & segment projection routines
│   └── obstacle_clustering.py    # DBSCAN spatial clustering, convex hull & portal extraction
├── scenarios/
│   ├── gnron.json                # Scenario A: C-shape concavity trap
│   ├── zigzag.json               # Scenario B: Narrow zigzag chicane corridor
│   └── dense.json                # Scenario C: Dense clutter obstacle yard
├── figures/
│   ├── experiments/              # Experimental simulation outputs (Figs. 6-15)
│   └── schematics/               # Theoretical diagrams & vehicle schematics (Figs. 1-5)
├── experiments/
│   ├── run_canonical.py          # Table 2: Benchmark in canonical environments
│   ├── run_monte_carlo.py        # Table 3: 100-trial statistical evaluation
│   ├── run_ablation.py           # Table 4: Component ablation benchmark
│   └── run_sensitivity.py        # Table 5: Parameter sensitivity analysis
├── generate_experimental_figures.py  # Render experimental figures (Figs. 6-15)
├── generate_schematic_figures.py     # Render theoretical schematics (Figs. 1-5)
├── generate_all_figures.py           # Master script compiling all 15 figures
├── run_benchmarks.py                 # One-command runner for all benchmarks and figures
├── requirements.txt                  # Python library dependencies
├── LICENSE                           # MIT License
└── README.md                         # Project documentation

```

---

## ⚙️ Installation

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

## 🚀 Quickstart & Full Reproducibility

To run all benchmarks and regenerate all 15 publication figures in a single command:

```bash
python run_benchmarks.py
```

### Reproducing Individual Tables and Figures:

- **Table 2 (Canonical Scenarios Performance Comparison):**
  ```bash
  python experiments/run_canonical.py
  ```
  Evaluates Baseline APF, CPF, and SCPF across *Baseline, C-Shape Entrance, C-Shape Escape, Zigzag Corridor,* and *Dense Clutter*.

- **Table 3 (Monte Carlo 100-Trial Statistical Evaluation):**
  ```bash
  python experiments/run_monte_carlo.py
  ```
  Evaluates success rates, local minima traps, jackknife rates, and IQR metrics across 100 multi-directional trials.

- **Table 4 (Quantitative Ablation Benchmark):**
  ```bash
  python experiments/run_ablation.py
  ```
  Evaluates Full SCPF vs. No Hysteresis, No Orthogonal Projection, and Base APF.

- **Table 5 (Parameter Sensitivity Benchmark):**
  ```bash
  python experiments/run_sensitivity.py
  ```
  Sweeps parameters $k_{\text{circ}}$, $k_\gamma$, $k_{\text{jk}}$, and $k_{\text{rep}}$.

- **Generate Experimental Figures (Figures 6–15):**
  ```bash
  python generate_experimental_figures.py
  ```
  Renders the 10 experimental trajectory and benchmark graphs into `figures/experiments/`.

- **Generate Theoretical & Schematic Diagrams (Figures 1–5):**
  ```bash
  python generate_schematic_figures.py
  ```
  Renders the 5 kinematic and architectural schematics into `figures/schematics/`.

- **Regenerate All Publication Figures:**
  ```bash
  python generate_all_figures.py
  ```
  Compiles and organizes all 15 figures.


---

## 📊 Summary of Experimental Results (from Manuscript)

### Table 2: Performance in Canonical Environments
| Scenario | Planner | Status | Path Length ($m$) | Max Articulation $|\gamma|_{\max}$ | Min Clearance ($m$) | Mean Latency ($ms$) |
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

## 📜 Citation

If you use this simulation software or refer to our findings in your research, please cite our paper:

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

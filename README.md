# Stateful Circular Potential Field (SCPF) for Autonomous Tractor-Trailer Navigation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official simulation codebase and experimental benchmark suite for the publication:

> **A Gap-Informed Stateful Circular Potential Field with Articulation Invariance for Autonomous Tractor-Trailer Navigation in Constrained Environments**  
> *Aries Jehan Tamamy, Adha Imam Cahyadi, and Igi Ardiyanto*  
> *International Journal of Intelligent Engineering and Systems (IJIES)*  
> Correspondence: `jehantammy@dsn.dinus.ac.id`  
> Department of Electrical and Information Engineering, Universitas Gadjah Mada, Yogyakarta, Indonesia  
> Department of Informatics, Universitas Dian Nuswantoro, Semarang, Indonesia  

---

## Overview

This repository provides the complete numerical simulation engine, experiment runners, and raw benchmark logs for autonomous tractor-trailer navigation in tightly constrained, obstacle-dense environments. Classical Artificial Potential Field (APF) and memoryless Circular Potential Field (CPF) controllers suffer from concave local minima, Goal Non-Reachable with Obstacles Nearby (GNRON) deadlocks, and inability to account for nonholonomic trailer off-tracking or mechanical jackknife instability.

The **Stateful Circular Potential Field (SCPF)** framework integrates three coupled layers:
1. **Topological Perception & Kinematic Gap Screening (Section 3.1–3.2)**: Groups raw range scans via Euclidean clustering ($d_{thresh} = 1.2\text{ m}$), extracts rolling convex hulls ($\mathcal{H}_k$) via Andrew's Monotone Chain algorithm, screens candidate openings against the transient lateral swept envelope ($W_{gap} \ge W_{swept} = w_{veh} + L_2\sin\gamma_{max} + d_{margin} = 0.60 + 1.128 + 0.392 = 2.12\text{ m}$, Assumption 4), and latches escape subgoals ($\mathbf{W}_{esc}$) outside non-convex enclosures.
2. **Field Modulation & GNRON Resolution (Section 3.3–3.5)**: Applies directional hysteresis with survival urgency scaling, neutralizes inward boundary penetration via orthogonal attractive projection ($\mathbf{F}_{att}'$), and resolves GNRON deadlocks via target-proximity repulsive attenuation ($\beta_{goal}(d_g) = \min(1, (d_g / (\rho_0 \eta_g))^2)$) with terminal stopping ball $\epsilon_{goal} = 0.5\text{ m}$.
3. **Kinematic Safety Regulation & Articulation Invariance (Section 3.6)**: Expands trailer repulsive clearance dynamically ($\rho_{dyn\_trailer} = \rho_{trailer\_base}(1 + k_\gamma |\gamma|/\gamma_{max})$) and enforces a virtual counter-torque governor ($\tau_{jk}$) processed through an explicit **4-stage discrete actuation pipeline** (raw demand synthesis, slew-rate limiter $|\dot{\delta}| \le \dot{\delta}_{max} = 1.0\text{ rad/s}$, first-order low-pass filter $\alpha_{steer} = 0.5$, and travel saturation $|\delta| \le \delta_{max} = 45.0^\circ$). As proven in Lemma 6 ($\delta_{crit}(60^\circ) = 30.0^\circ$) and Theorem 7, dynamic overshoot above the intervention threshold $\gamma_{safe} = 60.0^\circ$ is bounded by $\Delta\gamma_{overshoot} \le 6.98^\circ < 10.0^\circ$, certifying forward invariance of $|\gamma(t)| < \gamma_{max} = 70.0^\circ$.

All trajectories, tables, and experimental figures are generated directly via closed-loop 4th-order Runge-Kutta (RK4) integration ($\Delta t = 0.1\text{ s}$).

---

## Repository Structure

```text
scpf-tractor-trailer/
├── scpf/                             # Core simulation & controller package
│   ├── config.py                     # System & algorithm parameters (Table 1)
│   ├── vehicle.py                    # Nonholonomic tractor-trailer kinematics & footprints
│   ├── geometry.py                   # 2D polygon/circle distance & collision checks
│   ├── perception.py                 # 2D LiDAR raycasting, clustering, convex hull & gap screening
│   ├── planners.py                   # APF, CPF, and proposed SCPF (with ablation flags)
│   ├── simulator.py                  # Closed-loop RK4 episode runner & 4-stage actuator pipeline
│   ├── scenarios.py                  # Canonical scenes (Baseline, C-Shape, Zigzag, Clutter), Dynamic & Monte Carlo
│   └── __init__.py
├── experiments/                      # Reproducible benchmark evaluation scripts
│   ├── common.py                     # Multiprocessing runner, Wilson CI, and exact McNemar test helpers
│   ├── run_canonical.py              # Table 2 & Figs. 6–9: Canonical scenario evaluations
│   ├── run_ablation.py               # Table 3: Single-component ablation suite
│   ├── run_sensitivity.py            # Fig. 10: Parametric sensitivity sweeps (k_circ, k_gamma, k_jk, k_rep)
│   ├── run_monte_carlo.py            # Section 4.3 & Fig. 11: 100-trial dense clutter Monte Carlo benchmark
│   ├── run_governor_stress.py        # Section 3.6 & Fig. 5: Analytical vs. adversarial overshoot & robustness
│   ├── run_dynamic.py                # Section 4.4 & Fig. 12: Unmapped moving obstacle evasion
│   ├── run_escape_set.py             # Multi-pose interior C-shape escape verification
│   ├── run_timing.py                 # Section 4.4: Single-core per-step computational latency benchmark
│   └── __init__.py
├── results/                          # Machine-readable CSV & JSON experimental logs
│   ├── canonical.csv                 # Quantitative metrics for Table 2
│   ├── canonical_logs.json           # Full time-series trajectories for canonical runs
│   ├── ablation.csv                  # Single-component ablation runs (Table 3)
│   ├── ablation_summary.json         # Aggregated ablation statistics
│   ├── sensitivity.csv               # Full sensitivity sweep runs
│   ├── sensitivity_summary.json      # Aggregated sensitivity metrics (Fig. 10)
│   ├── monte_carlo.csv               # All 300 paired runs (100 seeds x 3 planners) & feasibility flags
│   ├── monte_carlo_summary.json      # Wilson 95% CIs and exact paired McNemar p-values (Fig. 11)
│   ├── governor_bound.json           # Analytic ZOH bound vs. adversarial simulation peaks
│   ├── robustness.csv                # Stress tests under reduced slew rate, delay, and sensor noise
│   ├── robustness_summary.json       # Aggregated robustness summary
│   ├── dynamic.csv                   # Dynamic obstacle crossing runs across speeds/angles/phases
│   ├── dynamic_logs.json             # Trajectory logs for dynamic evasion
│   ├── dynamic_summary.json          # Aggregated dynamic avoidance summary
│   ├── escape_set.csv                # Multi-pose C-shape interior escape results
│   └── timing.json                   # Per-step CPU execution latency statistics
├── figures/                          # Experimental figures generated from closed-loop RK4 runs (.pdf & .png)
│   ├── fig_skenario.*                # Benchmark scenarios overview
│   ├── fig_phase_invariance.*        # Fig. 5: Phase portrait & forward invariance verification
│   ├── fig_baseline.*                # Fig. 6: Baseline scenario trajectories, gamma(t), delta(t)
│   ├── trajectory_Cshape.*           # Fig. 7: C-Shape trap & interior turnaround escape trajectories
│   ├── fig_trajectories_zz_clutter.* # Fig. 8: Narrow zigzag chicane & 20-obstacle GNRON dense clutter
│   ├── fig_zigzag_curves.*           # Fig. 9: Steering effort delta(t) & articulation gamma(t) in zigzag
│   ├── fig_sensitivity.*             # Fig. 10: 2x2 parametric sensitivity sweeps
│   ├── fig_monte_carlo.*             # Fig. 11: 100-trial dense clutter Monte Carlo evaluation
│   └── fig_dynamic.*                 # Fig. 12: 1x3 horizontal dynamic obstacle evasion snapshots
├── generate_experimental_figures.py  # Generates all experimental figures in figures/
├── run_all_experiments.py            # Master pipeline to run all experiments and regenerate figures
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # Documentation
```

---

## Installation

```bash
git clone https://github.com/Jehantamamy/scpf-tractor-trailer.git
cd scpf-tractor-trailer
pip install -r requirements.txt
```

---

## Reproducing All Experiments and Figures

To run the complete experimental suite from scratch and regenerate all CSV/JSON logs in `results/` and all experimental figures in `figures/`:

```bash
python run_all_experiments.py
```

Or run individual experiment modules:

```bash
# 1. Canonical Scenarios (Table 2, Figs. 6-9)
python experiments/run_canonical.py

# 2. Single-Component Ablation Study (Table 3)
python experiments/run_ablation.py

# 3. Parametric Sensitivity Sweeps (Fig. 10)
python experiments/run_sensitivity.py

# 4. 100-Trial Dense Clutter Monte Carlo Benchmark (Section 4.3, Fig. 11)
python experiments/run_monte_carlo.py

# 5. Governor Forward Invariance & Adversarial Stress Tests (Section 3.6, Fig. 5)
python experiments/run_governor_stress.py

# 6. Dynamic Crossing Obstacle Evasion (Section 4.4, Fig. 12)
python experiments/run_dynamic.py

# 7. Computational Latency Benchmark (Section 4.4)
python experiments/run_timing.py

# 8. Regenerate All Experimental Figures (Figs. 5-12)
python generate_experimental_figures.py
```

---

## Summary of Experimental Results

### 1. Canonical Scenarios (Table 2)

| Scenario | Algorithm | Status | Path Length (m) | Max $\|\gamma\|$ (°) | Chattering (rad/step) | $T_{calc}$ (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | APF | Collision | 4.6 | 3.2 | 0.0133 | 0.18 |
| | CPF | Collision | 4.6 | 6.7 | 0.0350 | 0.22 |
| | **SCPF** | **Success** | **17.3** | **39.8** | **0.0135** | **1.25** |
| **A. C-Shape Trap** | APF | Jackknife | 12.7 | 72.5* | 0.0287 | 0.18 |
| | CPF | Collision | 11.6 | 27.7 | 0.0372 | 0.22 |
| | **SCPF** | **Success** | **26.7** | **48.2** | **0.0143** | **1.28** |
| **B. Zigzag Corridor** | APF | Jackknife | 12.1 | 70.5* | 0.0689 | 0.19 |
| | CPF | Collision | 14.1 | 42.3 | 0.0227 | 0.22 |
| | **SCPF** | **Success** | **34.8** | **56.3** | **0.0194** | **1.24** |
| **C. Dense Clutter (GNRON)** | APF | Collision | 18.3 | 67.6 | 0.0456 | 0.21 |
| | CPF | Collision | 7.4 | 26.7 | 0.0192 | 0.24 |
| | **SCPF** | **Success** | **15.6** | **34.2** | **0.0348** | **1.26** |

*\*Indicates mechanical jackknife failure ($|\gamma| \ge \gamma_{max} = 70.0^\circ$). Interior C-Shape turnaround escape via $\mathbf{W}_{esc}$ (Fig. 7b) completes in $38.4\text{ m}$ with peak $|\gamma| = 65.5^\circ < 70.0^\circ$.*

### 2. 100-Trial Dense Clutter Monte Carlo Benchmark (Section 4.3, Fig. 11)
Across 100 paired randomized layouts in a $50\text{ m} \times 50\text{ m}$ yard populated with $35\text{--}48$ circular obstacles (mean $41.4 \pm 2.8$ obstacles, all 100 scenes verified feasible via $W_{swept}$-inflated grid connectivity):

| Planner | Success Rate | 95% Wilson CI | Collisions | Jackknifes ($\ge 70^\circ$) | Trapped | Max $\|\gamma\|$ (°) | Paired McNemar $p$ (vs. SCPF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline APF** | 57.0% (57/100) | [47.2%, 66.3%] | 31 | 12 | 0 | 72.6° | $p = 5.82 \times 10^{-11}$ |
| **Standard CPF** | 56.0% (56/100) | [46.2%, 65.3%] | 44 | 0 | 0 | 50.2° | $p = 2.84 \times 10^{-10}$ |
| **Proposed SCPF** | **92.0% (92/100)** | **[85.0%, 95.9%]** | **6** | **0** | **2** | **66.2°** | — |

---

## Citation

```bibtex
@article{tamamy2026scpf,
  title={A Gap-Informed Stateful Circular Potential Field with Articulation Invariance for Autonomous Tractor-Trailer Navigation in Constrained Environments},
  author={Tamamy, Aries Jehan and Cahyadi, Adha Imam and Ardiyanto, Igi},
  journal={International Journal of Intelligent Engineering and Systems},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

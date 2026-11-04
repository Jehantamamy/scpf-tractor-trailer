"""Ablation Study Benchmark (Table 4 Replication).

Evaluates the contribution of core SCPF modules:
1. Full Proposed SCPF
2. No Hysteresis (Memoryless circulation)
3. No Orthogonal Projection (Standard attractive force)
4. Base APF (No stateful memory, no projection)
"""
import os
import sys
import json
import math
import numpy as np

# Ensure repository root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.config import SimulationConfig
from core.environment import SimulationEnvironment
from core.kinematics import rk4_step
from planners.scpf_planner import SCPFPlanner
from utils.obstacle_clustering import process_obstacle_clustering


def run_ablation_case(config_kwargs, env_type="c_shape"):
    config = SimulationConfig(**config_kwargs)
    env = SimulationEnvironment(config)

    json_map = {"c_shape": "gnron", "zigzag": "zigzag", "clutter": "dense"}
    json_path = os.path.join(ROOT_DIR, "scenarios", f"{json_map[env_type]}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)

    config.start_pos = data["config"]["start_pos"]
    config.goal = data["config"]["goal"]
    if "field_size" in data["config"]:
        config.field_size = data["config"]["field_size"]

    env.obstacles = []
    for obs in data["obstacles"]:
        if obs["type"] == "circle":
            env.obstacles.append({"id": obs["id"], "type": "circle", "pos": np.array(obs["pos"]), "rad": obs["rad"], "vel": np.array([0.0, 0.0])})
        elif obs["type"] == "polygon":
            env.obstacles.append({"id": obs["id"], "type": "polygon", "vertices": np.array(obs["vertices"]), "vel": np.array([0.0, 0.0])})

    planner = SCPFPlanner(config)
    t_pos = np.array(config.start_pos, dtype=float)
    t_ori = float(config.start_ori)
    tr_ori = float(config.start_ori)

    dt = config.dt
    v = config.v_tractor
    L1 = config.tractor_length
    L2 = config.trailer_length

    step = 0
    max_steps = 3000

    path = []
    chattering = 0.0
    prev_delta = 0.0
    max_gamma = 0.0
    status = "Timeout"

    while step < max_steps:
        path.append(t_pos.copy())
        dist_to_goal = np.linalg.norm(t_pos - np.array(config.goal))
        if dist_to_goal < 0.5:
            status = "Success"
            break

        final_obstacles = env.obstacles
        escape_wps = []
        if config.use_merge or config.use_cluster:
            final_obstacles, _, escape_wps = process_obstacle_clustering(
                env.obstacles, np.array(config.goal), t_pos,
                robot_radius=1.0, start_ori=t_ori,
                use_merge=config.use_merge, use_cluster=config.use_cluster
            )

        try:
            delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, final_obstacles, escape_wps)
        except TypeError:
            delta, _, _ = planner.calculate_control(t_pos, t_ori, tr_ori, final_obstacles)

        if step > 0:
            chattering += abs(delta - prev_delta)
        prev_delta = delta

        gamma = (t_ori - tr_ori + np.pi) % (2 * np.pi) - np.pi
        max_gamma = max(max_gamma, abs(math.degrees(gamma)))

        new_x, new_y, new_t_ori, new_tr_ori = rk4_step(t_pos[0], t_pos[1], t_ori, tr_ori, v, delta, L1, L2, dt)
        t_pos[0], t_pos[1] = new_x, new_y
        t_ori = new_t_ori
        tr_ori = new_tr_ori

        step += 1

    path_length = sum(np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path)))
    avg_chattering = (chattering / step) if step > 0 else 0.0

    return {
        "status": status,
        "length": path_length,
        "max_gamma": max_gamma,
        "chattering": avg_chattering,
        "steps": step
    }


def main():
    print("=" * 80)
    print("REPRODUCING TABLE 4: QUANTITATIVE ABLATION BENCHMARK")
    print("=" * 80)

    configs = [
        {"name": "Full SCPF", "kwargs": {"use_hysteresis": True, "use_orthogonal_projection": True}},
        {"name": "No Hysteresis", "kwargs": {"use_hysteresis": False, "use_orthogonal_projection": True}},
        {"name": "No Projection", "kwargs": {"use_hysteresis": True, "use_orthogonal_projection": False}},
        {"name": "Base APF", "kwargs": {"use_hysteresis": False, "use_orthogonal_projection": False}},
    ]

    environments = ["c_shape", "zigzag", "clutter"]

    for env_type in environments:
        print(f"\nScenario: {env_type.upper()}")
        print(f"{'Configuration':<18} | {'Status':<10} | {'Length (m)':<12} | {'Max Gamma':<10} | {'Chatter':<10}")
        print("-" * 70)
        for c in configs:
            res = run_ablation_case(c["kwargs"], env_type)
            print(f"{c['name']:<18} | {res['status']:<10} | {res['length']:<12.2f} | {res['max_gamma']:<10.2f} | {res['chattering']:<10.4f}")

    print("=" * 80)
    print("Ablation study completed successfully.")


if __name__ == '__main__':
    main()

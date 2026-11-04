# File: backend/core/config.py
from pydantic import BaseModel
from typing import List

class SimulationConfig(BaseModel):
    # Pengaturan Area & Robot
    field_size: List[float] = [15.0, 15.0]
    start_pos: List[float] = [2.0, 2.0]
    start_ori: float = 0.785
    goal: List[float] = [13.0, 13.0]
    
    # 🚀 BARU: Parameter Dimensi Radius & Iterasi (Menggantikan Magic Numbers)
    tractor_length: float = 0.8
    trailer_length: float = 1.2
    tractor_radius: float = 0.4  # Radius aman traktor
    trailer_radius: float = 0.6  # Radius aman trailer
    max_steps: int = 5000        # Batas timeout universal

    v_tractor: float = 1.0
    dt: float = 0.1
    stop_threshold: float = 0.5
    
    # Pengaturan Rintangan Default
    num_static_obs: int = 4
    num_dynamic_obs: int = 2
    obs_radius_range: List[float] = [0.8, 1.5]

    # BATASAN MEKANIK & SAFETY
    max_steer_deg: float = 30.0 
    jackknife_threshold_deg: float = 60.0
    stop_on_collision: bool = False
    stop_on_jackknife: bool = False

    # PARAMETER TUNING DASAR APF / ACPF
    k_att: float = 5.0
    k_rep: float = 10.0
    rho_0: float = 3.0
    rho_trailer: float = 1.5
    k_circ: float = 10.0       
    max_circular: float = 50.0 
    k_jk: float = 8.0
    gamma_safe: float = 50.0
    k_gamma: float = 1.5

    # PARAMETER ADVANCED ACPF
    k_att_overdrive: float = 5.0
    gnron_range_scale: float = 0.8
    gnron_power: float = 2.0
    tangential_gain: float = 2.5
    repulsive_gain: float = 1.0
    k_vel: float = 1.5
    
    # --- Fitur Tambahan ---
    use_merge: bool = False
    use_cluster: bool = False
    use_hysteresis: bool = True
    use_orthogonal_projection: bool = True
    visual_mode: str = "fluid"

    class Config:
        extra = "ignore"
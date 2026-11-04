# File: backend/core/environment.py
import numpy as np
import random
from .config import SimulationConfig

class SimulationEnvironment:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.obstacles = []
        self.generate_obstacles()

    def generate_obstacles(self):
        """Membuat rintangan statis dan dinamis secara acak berdasarkan config"""
        self.obstacles = []
        max_x, max_y = self.config.field_size
        
        # Area aman agar rintangan tidak menumpuk di start atau goal
        safe_margin = 3.0 
        start_pos = np.array(self.config.start_pos)
        goal_pos = np.array(self.config.goal)

        total_obs = self.config.num_static_obs + self.config.num_dynamic_obs
        
        for i in range(total_obs):
            is_dynamic = i >= self.config.num_static_obs
            
            while True:
                rad = random.uniform(*self.config.obs_radius_range)
                pos_x = random.uniform(rad, max_x - rad)
                pos_y = random.uniform(rad, max_y - rad)
                pos = np.array([pos_x, pos_y])

                # Cek agar tidak nabrak start atau goal
                if np.linalg.norm(pos - start_pos) < (rad + safe_margin): continue
                if np.linalg.norm(pos - goal_pos) < (rad + safe_margin): continue
                
                # Setup velocity untuk rintangan dinamis
                vel = np.array([0.0, 0.0])
                if is_dynamic:
                    vel = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])

                self.obstacles.append({
                    "id": i,
                    "pos": pos,
                    "rad": rad,
                    "vel": vel,
                    "is_dynamic": is_dynamic
                })
                break

    def update_dynamic_obstacles(self):
        """Menggerakkan rintangan dinamis dan memantulkannya jika kena tembok"""
        max_x, max_y = self.config.field_size
        dt = self.config.dt

        for obs in self.obstacles:
            if obs["is_dynamic"]:
                obs["pos"] += obs["vel"] * dt
                
                # Pantulkan dari dinding (X)
                if obs["pos"][0] - obs["rad"] < 0 or obs["pos"][0] + obs["rad"] > max_x:
                    obs["vel"][0] *= -1
                    obs["pos"] += obs["vel"] * dt # Koreksi posisi

                # Pantulkan dari dinding (Y)
                if obs["pos"][1] - obs["rad"] < 0 or obs["pos"][1] + obs["rad"] > max_y:
                    obs["vel"][1] *= -1
                    obs["pos"] += obs["vel"] * dt # Koreksi posisi

    def calculate_global_field_quiver(self, grid_resolution=0.8):
        """Menghitung vektor medan (U, V) untuk dikirim ke Vue sebagai latar quiver"""
        max_x, max_y = self.config.field_size
        X, Y = np.meshgrid(np.arange(0.5, max_x, grid_resolution), 
                           np.arange(0.5, max_y, grid_resolution))
        
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        goal = np.array(self.config.goal)
        
        # Gaya Tarik
        dx = goal[0] - X
        dy = goal[1] - Y
        dist_goal = np.hypot(dx, dy) + 1e-5
        U += 5.0 * dx / dist_goal
        V += 5.0 * dy / dist_goal
        
        # Gaya Tolak dan Melingkar dari rintangan
        k_rep, k_circ, circular_radius = 15.0, 15.0, 3.0
        
        for obs in self.obstacles:
            ox, oy = obs["pos"]
            rad = obs["rad"]
            
            vx = X - ox
            vy = Y - oy
            dist_center = np.hypot(vx, vy) + 1e-5
            dist_bound = dist_center - rad
            
            mask_rep = (dist_bound >= 0) & (dist_bound < circular_radius)
            mask_in = dist_bound < 0
            
            U[mask_in] += 50.0 * vx[mask_in] / dist_center[mask_in]
            V[mask_in] += 50.0 * vy[mask_in] / dist_center[mask_in]
            
            dir_out_x = vx[mask_rep] / dist_center[mask_rep]
            dir_out_y = vy[mask_rep] / dist_center[mask_rep]
            
            U[mask_rep] += k_rep * dir_out_x / (dist_bound[mask_rep]**2 + 1e-5)
            V[mask_rep] += k_rep * dir_out_y / (dist_bound[mask_rep]**2 + 1e-5)
            
            tx, ty = -dir_out_y, dir_out_x
            cross_p = dir_out_x * (goal[1] - oy) - dir_out_y * (goal[0] - ox)
            sign = np.where(cross_p > 0, 1.0, -1.0)
            
            U[mask_rep] += sign * k_circ * tx / (dist_bound[mask_rep] + 1e-5)
            V[mask_rep] += sign * k_circ * ty / (dist_bound[mask_rep] + 1e-5)
            
        # Logarithmic scaling agar panah tidak terlalu panjang/tumpang tindih
        mag = np.hypot(U, V) + 1e-5
        mag_scaled = np.clip(np.log1p(mag) * 0.8, 0, 5.0)
        
        U_vis = (U / mag) * mag_scaled
        V_vis = (V / mag) * mag_scaled
        
        # Mengembalikan data dalam bentuk list of dicts agar mudah diproses di frontend JSON
        quiver_data = []
        rows, cols = X.shape
        for i in range(rows):
            for j in range(cols):
                quiver_data.append({
                    "x": float(X[i, j]), "y": float(Y[i, j]),
                    "u": float(U_vis[i, j]), "v": float(V_vis[i, j])
                })
        return quiver_data
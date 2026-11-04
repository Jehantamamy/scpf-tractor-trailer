# File: backend/planners/base_planner.py
from abc import ABC, abstractmethod
import numpy as np

class BasePlanner(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def calculate_control(self, t_pos, t_ori, tr_ori, obstacles):
        pass

    def calculate_field_matrix(self, X, Y, obstacles):
        """
        Fungsi Vektorisasi NumPy. 
        Secara default akan memicu error agar main.py tahu planner ini belum di-upgrade.
        """
        raise NotImplementedError("Planner ini belum mendukung komputasi matriks.")
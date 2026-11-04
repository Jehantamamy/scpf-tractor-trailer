"""Abstract Base Planner Interface."""
from abc import ABC, abstractmethod
import numpy as np


class BasePlanner(ABC):
    """Base class for potential field motion planners."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def calculate_control(self, t_pos: np.ndarray, t_ori: float, tr_ori: float, obstacles):
        """Compute steering control command and potential forces."""
        pass

    def calculate_field_matrix(self, X: np.ndarray, Y: np.ndarray, obstacles):
        """Compute vectorized 2D potential field over a spatial grid."""
        raise NotImplementedError("Vector field calculation not implemented.")
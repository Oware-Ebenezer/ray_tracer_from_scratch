from dataclasses import dataclass
import numpy as np

@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray
    
    def at(self, distance: float) -> np.ndarray:
        """Calculate the point along the ray at a given distance."""
        return self.origin + distance * self.direction
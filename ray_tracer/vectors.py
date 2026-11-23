import numpy as np

def vec(x: float, y: float, z:float) -> np.ndarray:
    """Create a 3D vector using floating-point numbers."""
    return  np.array([x, y, z], dtype=float)

def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    
    if length == 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    
    return vector / length
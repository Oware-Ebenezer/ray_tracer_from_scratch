"""Ray-plane intersection (Member 2)."""
import numpy as np
from .sphere import HitRecord   # shared hit-result class (defined in sphere.py)
from ray_tracer.vectors import normalize

DEFAULT_T_MIN = 1e-4
# If |normal . direction| is smaller than this, the ray is treated as parallel.
PARALLEL_EPSILON = 1e-8


class Plane:
    """An infinite flat plane defined by one point on it and a normal vector."""

    def __init__(self, point, normal, material=None):
        point = np.asarray(point, dtype=float)
        if point.shape != (3,):
            raise ValueError("Plane point must contain exactly three coordinates (x, y, z).")
        self.point = point
        self.normal = normalize(normal)   # raises ValueError for a zero-length normal
        self.material = material

    def hit(self, ray, t_min: float = DEFAULT_T_MIN, t_max: float = np.inf):
        """Return a HitRecord for the hit with t_min < t < t_max, or None.

        A point P is on the plane when  (P - point) . normal = 0.
        Substituting P = origin + t * direction and solving for t gives
            t = ((point - origin) . normal) / (direction . normal)
        """
        denominator = np.dot(ray.direction, self.normal)

        # denominator ~ 0 means the ray runs parallel to the plane: no hit.
        if abs(denominator) < PARALLEL_EPSILON:
            return None

        t = np.dot(self.point - ray.origin, self.normal) / denominator

        # Reject hits behind the origin (or too close) and beyond t_max.
        if not (t_min < t < t_max):
            return None

        # HitRecord.from_ray flips the normal if the ray came from the back side.
        return HitRecord.from_ray(ray, t, self.normal, self.material)

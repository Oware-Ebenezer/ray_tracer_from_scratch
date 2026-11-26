"""Ray-sphere intersection (Member 2).

Also defines HitRecord, the result object returned by every hit() method.
plane.py imports HitRecord from this file.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class HitRecord:
    """Everything the renderer needs to know about one ray-surface hit.

    t           -- distance along the ray (hit point = origin + t * direction)
    point       -- exact 3D position of the hit
    normal      -- unit surface normal, always pointing AGAINST the incoming ray
    front_face  -- True if the ray hit the outside of the surface,
                   False if it hit from the inside (e.g. a ray inside a sphere)
    material    -- material of the object that was hit (None until one is assigned)
    """
    t: float
    point: np.ndarray
    normal: np.ndarray
    front_face: bool
    material: object = None

    @classmethod
    def from_ray(cls, ray, t, outward_normal, material=None):
        """Build a HitRecord and work out front/back face from the ray.

        outward_normal must be a unit vector pointing out of the surface.
        If the ray travels against it (dot product <= 0) we hit the front face.
        A dot product of exactly 0 is a grazing (tangent) hit from outside, so it
        counts as front face. Otherwise we hit the back face, so the normal is
        flipped to face the ray.
        """
        point = ray.at(t)
        front_face = bool(np.dot(ray.direction, outward_normal) <= 0.0)
        normal = outward_normal if front_face else -outward_normal
        return cls(t=float(t), point=point, normal=normal,
                   front_face=front_face, material=material)


# Smallest distance we accept as a hit. Stops a ray that has just left a
# surface from hitting that same surface again because of rounding errors.
DEFAULT_T_MIN = 1e-4


class Sphere:
    """A sphere defined by its centre and radius."""

    def __init__(self, center, radius: float, material=None):
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError("Sphere center must contain exactly three coordinates (x, y, z).")
        if radius <= 0:
            raise ValueError("Sphere radius must be positive.")
        self.center = center
        self.radius = float(radius)
        self.material = material

    def hit(self, ray, t_min: float = DEFAULT_T_MIN, t_max: float = np.inf):
        """Return a HitRecord for the nearest hit with t_min < t < t_max, or None.

        A point on the ray is  P(t) = origin + t * direction.
        A point is on the sphere when  |P - center|^2 = radius^2.
        Substituting gives a quadratic equation  a*t^2 + 2*half_b*t + c = 0.
        """
        oc = ray.origin - self.center                 # vector from sphere centre to ray origin

        a = np.dot(ray.direction, ray.direction)      # equals 1 for normalized directions
        half_b = np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2

        # discriminant < 0 -> miss, = 0 -> tangent (one touch point), > 0 -> two hits
        discriminant = half_b ** 2 - a * c
        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)

        # Try the nearer root first. If it is not in the valid range (e.g. it is
        # behind the origin because the ray starts inside), try the farther root.
        t = (-half_b - sqrt_disc) / a
        if not (t_min < t < t_max):
            t = (-half_b + sqrt_disc) / a
            if not (t_min < t < t_max):
                return None

        # Outward unit normal: from the centre towards the hit point.
        hit_point = ray.at(t)
        outward_normal = (hit_point - self.center) / self.radius

        return HitRecord.from_ray(ray, t, outward_normal, self.material)

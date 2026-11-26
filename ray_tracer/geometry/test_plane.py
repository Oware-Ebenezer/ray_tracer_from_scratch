import unittest
import numpy as np

from ray_tracer.ray import Ray
from ray_tracer.vectors import normalize
from ray_tracer.geometry.plane import Plane


def make_ray(origin, direction):
    """Helper: build a Ray with a unit-length direction, like the camera does."""
    return Ray(origin=np.array(origin, dtype=float), direction=normalize(direction))


class PlaneTests(unittest.TestCase):
    def setUp(self):
        # Horizontal floor at y = 0 with the normal pointing up
        self.floor = Plane(point=(0, 0, 0), normal=(0, 1, 0))

    def test_direct_hit(self):
        hit = self.floor.hit(make_ray((0, 3, 0), (0, -1, 0)))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.t, 3.0)

    def test_hit_position(self):
        hit = self.floor.hit(make_ray((2, 3, -1), (0, -1, 0)))
        np.testing.assert_allclose(hit.point, (2, 0, -1))

    def test_normal_faces_ray_from_above(self):
        hit = self.floor.hit(make_ray((0, 3, 0), (0, -1, 0)))
        np.testing.assert_allclose(hit.normal, (0, 1, 0))
        self.assertTrue(hit.front_face)

    def test_normal_flips_when_hit_from_below(self):
        hit = self.floor.hit(make_ray((0, -3, 0), (0, 1, 0)))
        np.testing.assert_allclose(hit.normal, (0, -1, 0))
        self.assertFalse(hit.front_face)

    def test_parallel_ray_misses(self):
        self.assertIsNone(self.floor.hit(make_ray((0, 2, 0), (1, 0, 0))))

    def test_nearly_parallel_ray_misses(self):
        self.assertIsNone(self.floor.hit(make_ray((0, 2, 0), (1, -1e-10, 0))))

    def test_ray_travelling_away(self):
        self.assertIsNone(self.floor.hit(make_ray((0, 3, 0), (0, 1, 0))))

    def test_angled_hit(self):
        hit = self.floor.hit(make_ray((0, 2, 0), (1, -1, 0)))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.t, 2 * np.sqrt(2))
        np.testing.assert_allclose(hit.point, (2, 0, 0), atol=1e-12)

    def test_non_unit_normal_is_normalized(self):
        plane = Plane((0, 0, 0), (0, 5, 0))
        self.assertAlmostEqual(np.linalg.norm(plane.normal), 1.0)

    def test_zero_normal_is_rejected(self):
        with self.assertRaises(ValueError):
            Plane((0, 0, 0), (0, 0, 0))

    def test_t_max_is_respected(self):
        self.assertIsNone(self.floor.hit(make_ray((0, 3, 0), (0, -1, 0)), t_max=2.0))


if __name__ == "__main__":
    unittest.main()

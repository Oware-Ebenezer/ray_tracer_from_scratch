import unittest
import numpy as np

from ray_tracer.camera import Camera
from ray_tracer.ray import Ray
from ray_tracer.vectors import normalize
from ray_tracer.geometry.sphere import Sphere


def make_ray(origin, direction):
    """Helper: build a Ray with a unit-length direction, like the camera does."""
    return Ray(origin=np.array(origin, dtype=float), direction=normalize(direction))


class SphereTests(unittest.TestCase):
    def setUp(self):
        # Unit sphere sitting 5 units in front of the origin (camera looks down -z)
        self.sphere = Sphere(center=(0, 0, -5), radius=1.0)

    def test_direct_hit(self):
        hit = self.sphere.hit(make_ray((0, 0, 0), (0, 0, -1)))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.t, 4.0)

    def test_hit_position_and_normal(self):
        hit = self.sphere.hit(make_ray((0, 0, 0), (0, 0, -1)))
        np.testing.assert_allclose(hit.point, (0, 0, -4))
        np.testing.assert_allclose(hit.normal, (0, 0, 1))

    def test_miss(self):
        self.assertIsNone(self.sphere.hit(make_ray((0, 3, 0), (0, 0, -1))))

    def test_tangent_ray(self):
        hit = self.sphere.hit(make_ray((1, 0, 0), (0, 0, -1)))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.t, 5.0)
        np.testing.assert_allclose(hit.point, (1, 0, -5))
        np.testing.assert_allclose(hit.normal, (1, 0, 0))

    def test_ray_pointing_away(self):
        self.assertIsNone(self.sphere.hit(make_ray((0, 0, 0), (0, 0, 1))))

    def test_angled_hit_lies_on_sphere(self):
        ray = make_ray((0, 0, 0), (0.1, 0, -1))
        hit = self.sphere.hit(ray)
        self.assertIsNotNone(hit)
        distance_to_centre = np.linalg.norm(hit.point - self.sphere.center)
        self.assertAlmostEqual(distance_to_centre, 1.0)
        np.testing.assert_allclose(hit.point, ray.at(hit.t))
        self.assertAlmostEqual(np.linalg.norm(hit.normal), 1.0)

    def test_front_face_from_outside(self):
        hit = self.sphere.hit(make_ray((0, 0, 0), (0, 0, -1)))
        self.assertTrue(hit.front_face)

    def test_ray_starting_inside_hits_far_side(self):
        hit = self.sphere.hit(make_ray((0, 0, -5), (0, 0, -1)))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.t, 1.0)
        np.testing.assert_allclose(hit.point, (0, 0, -6))

    def test_back_face_from_inside(self):
        hit = self.sphere.hit(make_ray((0, 0, -5), (0, 0, -1)))
        self.assertFalse(hit.front_face)
        # Outward normal is (0,0,-1); it is flipped to face the ray.
        np.testing.assert_allclose(hit.normal, (0, 0, 1))

    def test_t_min_and_t_max_are_respected(self):
        ray = make_ray((0, 0, 0), (0, 0, -1))
        self.assertIsNone(self.sphere.hit(ray, t_max=3.0))
        self.assertIsNone(self.sphere.hit(ray, t_min=7.0))

    def test_material_is_passed_through(self):
        sphere = Sphere((0, 0, -5), 1.0, material="red")
        self.assertEqual(sphere.hit(make_ray((0, 0, 0), (0, 0, -1))).material, "red")

    def test_invalid_radius_is_rejected(self):
        with self.assertRaises(ValueError):
            Sphere((0, 0, 0), 0)

    def test_works_with_camera_centre_ray(self):
        camera = Camera(1280, 720, 60, (0, 0, 0), (0, 0, -1))
        hit = self.sphere.hit(camera.generate_ray(640, 360))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.t, 4.0)


if __name__ == "__main__":
    unittest.main()

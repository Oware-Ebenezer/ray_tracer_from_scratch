import unittest
import numpy as np

from ray_tracer.camera import Camera


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(1280, 720, 60, (0, 0, 0), (0, 0, -1))

    def test_aspect_ratio_16_by_9(self):
        self.assertAlmostEqual(self.camera.aspect_ratio, 16 / 9)

    def test_exact_centre_ray_follows_view_direction(self):
        ray = self.camera.generate_ray(640, 360)
        np.testing.assert_allclose(ray.direction, (0, 0, -1), atol=1e-12)

    def test_ray_origin_matches_camera_position(self):
        moved = Camera(640, 480, 60, (2, 3, 4), (2, 3, 3))
        ray = moved.generate_pixel_ray(10, 10)
        np.testing.assert_allclose(ray.origin, (2, 3, 4))

    def test_all_sampled_directions_are_unit_length(self):
        for x, y in [(0, 0), (639, 359), (1279, 719), (400, 200)]:
            ray = self.camera.generate_pixel_ray(x, y)
            self.assertAlmostEqual(np.linalg.norm(ray.direction), 1.0)

    def test_corner_rays_point_to_correct_sides(self):
        top_left = self.camera.generate_pixel_ray(0, 0).direction
        bottom_right = self.camera.generate_pixel_ray(1279, 719).direction
        self.assertLess(top_left[0], 0)
        self.assertGreater(top_left[1], 0)
        self.assertGreater(bottom_right[0], 0)
        self.assertLess(bottom_right[1], 0)

    def test_wider_fov_produces_wider_ray_spread(self):
        narrow = Camera(1280, 720, 40, (0, 0, 0), (0, 0, -1))
        wide = Camera(1280, 720, 90, (0, 0, 0), (0, 0, -1))
        narrow_ray = narrow.generate_pixel_ray(1279, 359).direction
        wide_ray = wide.generate_pixel_ray(1279, 359).direction
        narrow_angle = np.arccos(np.clip(np.dot(narrow.forward, narrow_ray), -1, 1))
        wide_angle = np.arccos(np.clip(np.dot(wide.forward, wide_ray), -1, 1))
        self.assertGreater(wide_angle, narrow_angle)

    def test_rotated_camera_centre_ray_follows_target(self):
        camera = Camera(800, 600, 60, (1, 2, 3), (2, 2, 3))
        ray = camera.generate_ray(400, 300)
        np.testing.assert_allclose(ray.direction, (1, 0, 0), atol=1e-12)

    def test_invalid_dimensions_and_fov_are_rejected(self):
        with self.assertRaises(ValueError):
            Camera(0, 720, 60, (0, 0, 0), (0, 0, -1))
        with self.assertRaises(ValueError):
            Camera(640, 480, 180, (0, 0, 0), (0, 0, -1))

    def test_out_of_range_pixel_is_rejected(self):
        with self.assertRaises(IndexError):
            self.camera.generate_pixel_ray(1280, 0)


if __name__ == "__main__":
    unittest.main()
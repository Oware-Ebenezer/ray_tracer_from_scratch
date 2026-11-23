"""Quick manual check of camera settings and generated pixel rays."""
import numpy as np
from ray_tracer.camera import Camera

camera = Camera(
    width=1280,
    height=720,
    vertical_fov_degrees=60,
    position=(0, 0, 0),
    target=(0, 0, -1),
)

print("Resolution:", f"{camera.width} x {camera.height}")
print("Aspect ratio:", camera.aspect_ratio)
print("FOV:", camera.vertical_fov_degrees, "degrees")
print("Forward:", camera.forward)
print("Right:", camera.right)
print("Up:", camera.up)
print()

samples = {
    "top-left": (0, 0),
    "centre pixel": (639, 359),
    "bottom-right": (1279, 719),
}

for label, (x, y) in samples.items():
    ray = camera.generate_pixel_ray(x, y)
    print(f"{label:15} pixel=({x:4}, {y:3})  direction={np.round(ray.direction, 5)}")

# A full render loop should call generate_pixel_ray once per pixel.
total_primary_rays = camera.width * camera.height
print()
print("Primary rays required for one full image:", total_primary_rays)
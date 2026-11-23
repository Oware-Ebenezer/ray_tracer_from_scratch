# Camera Integration Contract

This component owns primary-ray generation only. It does not perform intersections, shading, shadows, reflections, refraction, sampling, acceleration, or image output.

## Public API

```python
from raytracer.camera import Camera

camera = Camera(
    width=1280,
    height=720,
    vertical_fov_degrees=60,
    position=(0.0, 0.0, 0.0),
    target=(0.0, 0.0, -1.0),
)
ray = camera.generate_pixel_ray(x, y)
```

`generate_pixel_ray(x, y)` accepts integer pixel coordinates, with `(0, 0)` at the top-left. It returns a `Ray` with:
- `ray.origin`: NumPy 3-vector
- `ray.direction`: normalized NumPy 3-vector
- `ray.at(t)`: point at distance/parameter `t`

## Renderer hand-off

The renderer should request one primary ray per pixel and pass that ray to the team's existing tracing function. Keep the tracing function and its return type owned by the renderer team.

```python
for y in range(camera.height):
    for x in range(camera.width):
        primary_ray = camera.generate_pixel_ray(x, y)
        pixel_colour = trace(primary_ray)  # team's existing tracing function
        image[y, x] = pixel_colour
```

Replace `trace` and `image` with the names and interfaces already used in the shared renderer. Do not add a second Ray class if the team already has one; during integration, either use this `Ray` type or adapt the camera to the team's existing ray type.

## Coordinate and projection conventions

- Camera looks from `position` toward `target`.
- `world_up` defaults to `(0, 1, 0)`.
- Image coordinates start at the top-left; x increases right and y increases down.
- `vertical_fov_degrees` is the vertical field of view.
- Aspect ratio is calculated as `width / height`.
- Rays pass through pixel centres for integer pixel coordinates.
- The ray direction is normalized.

## Assignment render target

For 1280 × 720, the renderer requests 921,600 primary rays per frame. The camera does not itself render or save a frame.

## Verification

From the project root:

```bash
python -m unittest -v
python camera_check.py
```

Install NumPy first if it is not already installed:

```bash
python -m pip install numpy
```
import numpy as np
from .ray import Ray
from .vectors import normalize, vec

class Camera:
    def __init__(self, width: int, height: int, vertical_fov_degrees: float, position, target, world_up=(0.0, 1.0, 0.0)):
        if width <= 0 or height <= 0:
            raise ValueError("Camera width and height must be positive.")
        if not 0.0 < vertical_fov_degrees < 180.0:
            raise ValueError("Vertical field of view must be between 0 and 180 degrees.")
        self.width =  int(width)
        self.height = int(height)
        self.position = np.asarray(position, dtype=float)
        
        if self.position.shape != (3,):
            raise ValueError("Camera position must contain exactly three coordinates (x, y, z).")
        
        target = np.asarray(target, dtype=float)
        if target.shape != (3,):
            raise ValueError("Camera target must contain exactly three coordinates (x, y, z).")
        
        self.forward = normalize(target - self.position)
        self.right = normalize(np.cross(self.forward, world_up))
        self.up = normalize(np.cross(self.right, self.forward))
        
        # Avoid a zero cross product by checking if forward and world_up are parallel
        if abs(np.dot(self.forward, world_up)) > 0.999:
            world_up = vec(0.0, 0.0, 1.0)  # Use a different up vector if they are parallel
        self.right = normalize(np.cross(self.forward, world_up))
        self.up = normalize(np.cross(self.right, self.forward))
        
        #Image proportions and perspective projection dimensions
        self.aspect_ratio = self.width / self.height
        self.vertical_fov_degrees = float(vertical_fov_degrees)
        
        half_fov_radians = np.radians(self.vertical_fov_degrees ) / 2.0
        self.half_height = np.tan(half_fov_radians) 
        self.half_width = self.aspect_ratio * self.half_height
        
    def generate_ray(self, pixel_x: float, pixel_y: float) -> Ray:
            """ Generate a camera ray through a continous image-plane coordinate """
            
            #Map pixel coordinates to the range [-1, 1] on the image plane
            screen_x = (2.0 * pixel_x / self.width) - 1.0
            screen_y = 1.0 - (2.0 * pixel_y / self.height)  # Invert y-axis for image coordinates
            
            # Scale by the perspective image plane dimensions.
            screen_x *= self.half_width
            screen_y *= self.half_height
            
            direction = (self.forward + screen_x * self.right + screen_y * self.up)
            
            return Ray(origin=self.position.copy(), direction=normalize(direction))
        
    def generate_pixel_ray(self, pixel_x: int, pixel_y: int) -> Ray:
            """ Generate one ray through the center of an integer pixel """
            if not (0 <= pixel_x < self.width and 0 <= pixel_y < self.height):
                raise IndexError("Pixel coordinates must be within the image dimensions.")
            return self.generate_ray(float(pixel_x) + 0.5, float(pixel_y) + 0.5)
        
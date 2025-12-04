import math
from operator import sub


class Circle():
    """Class that describes circle figure."""

    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

    def distance_to(self, circle):
        """Returns distance to passed circle."""
        diff = tuple(map(sub, self.pos, circle.pos))
        return math.hypot(*diff)

    def sqr_distance_to(self, circle):
        """Returns square distance to passed circle."""
        return (self.pos[0] - circle.pos[0])**2 + (self.pos[1] - circle.pos[1])**2

    def is_intersects(self, circle):
        """Returns True if circles intersects, otherwise False."""
        threshold = self.radius + circle.radius
        if self.sqr_distance_to(circle) < threshold * threshold:
            return True
        return False

    def area(self):
        """Return circle area."""
        return math.pi * self.radius**2
    
    def perimeter(self):
        """Return circle perimeter."""
        return 2 * math.pi * self.radius
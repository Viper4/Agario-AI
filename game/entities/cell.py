import random
from operator import add, sub

from .. import gameutils as gu
from .interfaces import Victim
from .circle import Circle


class Cell(Circle, Victim):
    """Represents cell(food) state."""

    BORDER_WIDTH = 0
    FRICTION = 0.05
    SIZES = (6, 7, 8)
    SIZES_CUM = (20, 70, 10)

    def __init__(self, fps, pos, radius, color, angle=0, speed=0):
        super().__init__(pos, radius)
        # cell color [r, g, b]
        self.color = color
        # angle of speed in rad
        self.angle = angle
        # speed coeff from 0.0 to 1.0
        self.speed = speed
        self.num_food_eaten = 0
        self.fps = fps
        self.time_scale = 60 / fps  # 60 FPS is the default running speed for a normal game

    def move(self):
        """Move accroding to stored velocity."""
        self.speed -= self.FRICTION
        if self.speed < 0:
            self.speed = 0
        # Max speed affected by size with this formula: speed = (mass / mass^1.44) * 10
        MAX_SPEED = (self.mass() / self.mass()**1.44) * 10
        # get cartesian vector
        diff_xy = gu.polar_to_cartesian(self.angle, self.speed*MAX_SPEED)
        # change position
        self.pos = list(map(add, self.pos, diff_xy))

    def update_velocity(self, angle, speed):
        """Add self velocity vector with passed velocity vector."""
        before_speed = self.speed

        v1 = gu.polar_to_cartesian(angle, speed)
        v2 = gu.polar_to_cartesian(self.angle, self.speed)
        # adding vectors
        v3 = list(map(add, v1, v2))
        # convert to polar
        self.angle, self.speed = gu.cartesian_to_polar(*v3)

        #self.angle = angle
        #self.speed = speed
        # normilize speed coeff
        if before_speed <= 1 and self.speed > 1:
            self.speed = 1
        elif before_speed > 1 and self.speed > before_speed:
            self.speed = before_speed

    def try_to_kill_by(self, killer):
        """Check is killer cell could eat current cell."""
        # Must be ~82% mass of killer to be eaten
        if self.mass() < killer.mass() * 0.82 and self.distance_to(killer) <= killer.radius - self.radius:
            return self
        return None

    def mass(self):
        """Returns the mass of this cell."""
        # agar.io stats from https://gamefaqs.gamespot.com/webonly/163063-agario/faqs/73510
        # mass = size^2 / 100
        return (self.radius * self.radius) / 100

    @classmethod
    def make_random(cls, fps, bounds):
        """Creates random cell."""
        pos = gu.random_pos(bounds)
        radius = random.choices(cls.SIZES, cls.SIZES_CUM)[0]
        color = gu.random_safe_color()
        return cls(fps, pos, radius, color)

    def __repr__(self):
        return '<{} pos={} radius={}>'.format(
            self.__class__.__name__,
            list(map(int, self.pos)),
            int(self.radius))
import random
import math

from .cell import Cell
from .. import gameutils as gu


class Virus(Cell):
    """Agar.io Virus entity.
    - Acts like a cell on the field.
    - When fully engulfed by a player cell, it is "eaten" and causes the player to split into many parts.
    """

    BORDER_WIDTH = 5
    FRICTION = 0.05
    MAX_SPEED = 0
    # Typical virus size range
    SIZES = (50, 51, 52)
    SIZES_CUM = (30, 50, 20)

    DEFAULT_COLOR = (80, 200, 120)

    SPLIT_RADIUS_THRESHOLD = 65  # Radius threshold for virus to split

    def __init__(self, fps, pos, radius, color=None, angle=0, speed=0):
        color = self.DEFAULT_COLOR if color is None else color
        super().__init__(fps, pos, radius, color, angle, speed)

    def try_to_kill_by(self, killer):
        """If a player cell fully engulfs the virus, the virus is considered 'killed'.
        Returning self allows the model to remove the virus and then apply splitting logic.
        """
        # Must be ~82% mass of killer to be eaten
        if self.mass() < killer.mass() * 0.82 and self.distance_to(killer) <= killer.radius - self.radius:
            return self
        return None

    def __add_area(self, area):
        """Increase current cell area with passed area."""
        self.radius = math.sqrt((super().area() + area) / math.pi)

    def eat(self, cell):
        """Eat the given cell"""
        self.__add_area(cell.area())

    def try_split(self, angle, speed):
        """Try to split the virus and return a new virus if one was created otherwise return None."""
        # Check if virus is big enough to split
        if self.radius < self.SPLIT_RADIUS_THRESHOLD:
            return None
        # Update current virus radius
        self.radius = random.choices(self.SIZES, self.SIZES_CUM)[0]

        # Create a new virus
        radius = random.choices(self.SIZES, self.SIZES_CUM)[0]
        # find diff_xy to move spawn virus on current circle border
        diff_xy = gu.polar_to_cartesian(angle, self.radius + radius)
        pos = [self.pos[0] + diff_xy[0], self.pos[1] + diff_xy[1]]
        return Virus(self.fps, pos, radius, self.color, angle, speed)

    @classmethod
    def make_random(cls, fps, bounds):
        pos = gu.random_pos(bounds)
        radius = random.choices(cls.SIZES, cls.SIZES_CUM)[0]
        color = cls.DEFAULT_COLOR
        return cls(fps, pos, radius, color)

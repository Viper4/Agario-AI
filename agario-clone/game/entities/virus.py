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
    FRICTION = 0.0  # Viruses don't move by themselves once placed
    MAX_SPEED = 0
    # Typical virus size range
    SIZES = (40, 41, 42)
    SIZES_CUM = (30, 50, 20)

    DEFAULT_COLOR = (80, 200, 120)

    def __init__(self, pos, radius, color=None, angle=0, speed=0):
        color = self.DEFAULT_COLOR if color is None else color
        super().__init__(pos, radius, color, angle, speed)

    def move(self):
        """Viruses are static by default."""
        return

    def try_to_kill_by(self, killer):
        """If a player cell fully engulfs the virus, the virus is considered 'killed'.
        Returning self allows the model to remove the virus and then apply splitting logic.
        """
        # Use engulfing condition similar to Cell, but without area ratio constraint
        if self.distance_to(killer) <= killer.radius - self.radius:
            return self
        return None

    @classmethod
    def make_random(cls, bounds):
        pos = gu.random_pos(bounds)
        radius = random.choices(cls.SIZES, cls.SIZES_CUM)[0]
        color = cls.DEFAULT_COLOR
        return cls(pos, radius, color)

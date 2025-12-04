import math
import random
from operator import add, sub

from .. import gameutils as gu
from . import interfaces
from .cell import Cell
from .virus import Virus


class PlayerCell(Cell, interfaces.Killer):
    """Represents player cell(part of player) state."""

    BORDER_WIDTH = 5
    # size of player when created
    SIZES = (5,)
    SIZES_CUM = (1,)

    # min ratius of cell to be able shoot
    SHOOTCELL_COND_RADIUS = 48
    SHOOTCELL_RADIUS = 24
    SHOOTCELL_SPEED = 2

    # min ratius of cell to be able split
    SPLITCELL_COND_RADIUS = 40
    SPLITCELL_SPEED = 3
    # the time before a сell can merge with another cell in seconds
    SPLIT_TIMEOUT = 30

    DECAY_TIME = 5
    MIN_RADIUS = 20

    def __init__(self, pos, radius, color, angle=0, speed=0):
        super().__init__(pos, radius, color, angle, speed)
        # merge_time = 30 + cell mass * 2.33%
        self.split_timeout = self.SPLIT_TIMEOUT + int(self.mass() * 0.0233)
        self.split_timeout *= 60  # Convert to number of frames in normal 60 FPS
        # food storage, to make the radius change smooth
        self.area_pool = 0
        self.decay_timer = self.DECAY_TIME * 60  # Lose 1 radius every x*60 frames

    def move(self):
        """Update cell state and move by stored velocity."""
        self.__tick()
        self.__add_area(self.__area_pool_give_out())
        super().move()

    def eat(self, cell, num_parts, max_parts):
        """Increase current cell area with passed cell area,
        by changing cell area.
        """
        if isinstance(cell, Virus) and num_parts < max_parts:
            # Add less of virus area to pool
            self.area_pool += cell.area() * 0.5
        else:
            self.area_pool += cell.area()
        self.__add_area(self.__area_pool_give_out())

    def __tick(self):
        """Make updates over time."""
        # Count down split timer
        if self.split_timeout > 0:
            self.split_timeout -= 1

        # Decay mass
        self.decay_timer -= 1
        if self.radius > self.MIN_RADIUS and self.decay_timer <= 0:
            self.radius -= 1
            self.decay_timer = self.DECAY_TIME * 60

    def __add_area(self, area):
        """Increase current cell area with passed area."""
        self.radius = math.sqrt((super().area() + area) / math.pi)

    def __area_pool_give_out(self, part=0.05):
        """Returns some part of food from area pool."""
        if self.area_pool > 0:
            area = self.area_pool * part
            self.area_pool *= 1 - part
        else:
            area = 0
        return area

    def spit_out(self, cell):
        """Decrease current cell area with passed cell area,
        by changing cell area.
        """
        self.radius = math.sqrt((super().area() - cell.area()) / math.pi)

    def able_to_emit(self, cond_radius):
        """Checks if cell able to emmit."""
        return self.radius >= cond_radius

    def emit(self, angle, speed, radius, ObjClass):
        """Emit cell with given angle in degrees and emit type.
        Returns emmited object.
        """
        # create emmited object at pos [0, 0]
        obj = ObjClass(
            [0, 0], radius, 
            self.color, 
            angle, speed)
        # change current cell radius
        self.spit_out(obj)
        # find diff_xy to move spawn cell on current circle border
        diff_xy = gu.polar_to_cartesian(angle, self.radius + radius)
        # move created object
        obj.pos = list(map(add, self.pos, diff_xy))
        return obj

    def attempt_murder(self, victim):
        """Try to kill passed victim cell by self cell."""
        return victim.try_to_kill_by(self)

    def shoot(self, angle):
        """Shoot in the given angle.
        Returns the fired cell.
        """
        return self.emit(
            angle, 
            self.SHOOTCELL_SPEED,
            self.SHOOTCELL_RADIUS,
            Cell)

    def able_to_shoot(self):
        """Checks is cell able to shoot."""
        return self.able_to_emit(self.SHOOTCELL_COND_RADIUS)

    def split(self, angle):
        """Spit cell in the given angle in degrees.
        Returns the splitted part.
        """
        return self.emit(
            angle, 
            self.SPLITCELL_SPEED,
            self.radius/math.sqrt(2),
            PlayerCell)

    def able_to_split(self):
        """Checks is cell able to split."""
        return self.able_to_emit(self.SPLITCELL_COND_RADIUS)

    def regurgitate_from(self, cell):
        """Pushing current cell to edge of the passed cell.
        It is necessary to get rid of the collision beetwen them.
        """
        # vector between centers
        dx = self.pos[0] - cell.pos[0]
        dy = self.pos[1] - cell.pos[1]

        dist2 = dx * dx + dy * dy
        dist = math.sqrt(dist2)

        # intersection amount
        delta = self.radius + cell.radius - dist
        if delta <= 0:
            return  # no overlap

        # normalize (dx,dy)
        if dist == 0:
            # if cells are on the same point, push cell in random direction
            self.pos[0] += random.uniform(-1, 1)
            self.pos[1] += random.uniform(-1, 1)
            return
        nx = dx / dist
        ny = dy / dist

        # push out by delta
        self.pos[0] += nx * delta
        self.pos[1] += ny * delta

    def area(self):
        """Returns full PlayerCell area, including area stored in pool."""
        return super().area() + self.area_pool

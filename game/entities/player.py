import functools
import operator
import math
import time

from .. import gameutils as gu
from . import interfaces
from .playercell import PlayerCell
from .virus import Virus


class Player(interfaces.Victim, interfaces.Killer):
    """Class that represents player game state."""

    START_SIZE = 20
    BORDER_WIDTH = 5

    MAX_PARTS = 16

    def __init__(self, nick, player_cell):
        self.nick = nick
        # cells of which player consists
        self.parts = [player_cell]
        # self.parts = [PlayerCell(pos, radius, color, border_color)]
        self.num_food_eaten = 0
        self.num_players_eaten = 0
        self.highest_score = 0.0
        self.ticks_alive = 0
        self.alive = True

    def move(self):
        """Move each part of player and check parts for collision."""
        if not self.alive:
            return
        self.ticks_alive += 1
        for i, cell in enumerate(self.parts):
            cell.move()
            for part in self.parts[i + 1:]:
                # cells shoud intersects and not be the same
                if cell == part or not cell.is_intersects(part):
                    continue

                # merge cells if their timeout is zero
                # otherwise get rid off colission between them
                if cell.split_timeout <= 0 and part.split_timeout <= 0:
                    cell.eat(part, len(self.parts), self.MAX_PARTS)
                    self.parts.remove(part)
                else:
                    cell.regurgitate_from(part)

    def update_velocity(self, target_pos):
        """Update velocity of each part."""
        for cell in self.parts:
            direction = (target_pos[0] - cell.pos[0], target_pos[1] - cell.pos[1])
            angle = math.atan2(direction[1], direction[0])

            # Calculate speed as square distance to target pos from this cell
            speed = min(1.0, (direction[0] * direction[0] + direction[1] * direction[1]) / 10000)
            cell.update_velocity(angle, speed)

    def shoot(self, target_pos):
        """Shoots with cells towards target_pos."""
        emmited = list()
        for cell in self.parts:
            direction = (target_pos[0] - cell.pos[0], target_pos[1] - cell.pos[1])
            angle = math.atan2(direction[1], direction[0])
            if cell.able_to_shoot():
                emmited.append(cell.shoot(angle))

        return emmited

    def split(self, target_pos):
        new_parts = list()
        for cell in self.parts:
            direction = (target_pos[0] - cell.pos[0], target_pos[1] - cell.pos[1])
            angle = math.atan2(direction[1], direction[0])
            if len(self.parts) + len(new_parts) >= self.MAX_PARTS:
                break
            if cell.able_to_split():
                new_parts.append(cell.split(angle))

        self.parts.extend(new_parts)
        return new_parts

    def explode(self, hit_cell):
        """
        Explode the hit_cell into 1 big split and 15 small splits (from eating a virus)
        :param hit_cell:
        :return:
        """
        max_new_parts = self.MAX_PARTS - len(self.parts)

        if max_new_parts <= 0:
            return

        # Make the big split first
        self.parts.append(hit_cell.split(0))
        max_new_parts -= 1

        # Make the remaining smaller splits
        for i in range(max_new_parts):
            if not hit_cell.able_to_split():
                break
            angle = 2 * 3.1415926535 * i / self.MAX_PARTS

            new_cell = hit_cell.emit(angle, hit_cell.SPLITCELL_SPEED, hit_cell.SPLITCELL_COND_RADIUS, PlayerCell)
            new_cell.split_timeout /= 2
            self.parts.append(new_cell)

    def center(self):
        """Returns median position of all player cells."""
        xsum = sum((cell.pos[0] for cell in self.parts))
        ysum = sum((cell.pos[1] for cell in self.parts))
        center = [
            xsum/len(self.parts),
            ysum/len(self.parts)]
        return center

    def score(self):
        """Returns player score and updates highest score.
        Score is radius of circle that consists of all parts area sum.
        """
        radius_sqr = functools.reduce(
            operator.add,
            (cell.radius**2 for cell in self.parts))
        score = math.sqrt(radius_sqr)
        if score > self.highest_score:
            self.highest_score = score
        return score

    def attempt_murder(self, victim):
        """Try to kill passed victim by player parts. 
        Returns killed Cell if can and the cell part that killed it, otherwise return None.
        """
        for part in self.parts:
            killed_cell = victim.try_to_kill_by(part)
            if killed_cell:
                # feed player cell with killed cell
                part.eat(killed_cell, len(self.parts), self.MAX_PARTS)
                if isinstance(killed_cell, PlayerCell):
                    self.num_players_eaten += 1
                elif not isinstance(killed_cell, Virus):
                    self.num_food_eaten += 1
                return killed_cell, part
        return None, None

    def try_to_kill_by(self, killer):
        """Check is killer cell could eat some of player parts.
        Returns killed player part or None.
        """
        for cell in self.parts:
            killed_cell = killer.attempt_murder(cell)
            if killed_cell:
                return killed_cell
        return None

    def remove_part(self, cell):
        """Removes passed player cell from player parts list."""
        try:
            self.parts.remove(cell)
        except ValueError:
            pass

    def reset(self):
        self.parts = self.parts[:1]
        self.parts[0].area_pool = 0
        self.parts[0].radius = self.START_SIZE

    @classmethod
    def make_random(cls, nick, bounds):
        """Returns random player with given nick."""
        player_cell = PlayerCell.make_random(bounds)
        player_cell.radius = cls.START_SIZE
        return cls(nick, player_cell)

    def __repr__(self):
        return '<{} nick={} score={}>'.format(
            self.__class__.__name__,
            self.nick,
            int(self.score()))
import functools
import operator
import math

from .. import gameutils as gu
from .playercell import PlayerCell


class Player:
    """Class that represents player game state."""

    START_SIZE = 20
    BORDER_WIDTH = 5

    MAX_PARTS = 16

    def __init__(self, nick, player_cell):
        self.nick = nick
        # cells of which player consists
        self.parts = {player_cell}
        player_cell.parent = self
        # self.parts = [PlayerCell(pos, radius, color, border_color)]
        self.num_food_eaten = 0
        self.num_players_eaten = 0
        self.highest_score = 0.0
        self.alive = True
        self.ticks_alive = 0

    def check_merge(self, playercell):
        """
        Check if playercell can be merged with any of player parts.
        :param playercell:
        :return: The merged cell if merged, otherwise None
        """
        for cell in self.parts:
            if cell is playercell or not cell.is_intersects(playercell):
                continue
            if cell.split_timeout <= 0 and playercell.split_timeout <= 0:
                if playercell not in self.parts:
                    return None  # Another part of this player already merged this cell
                cell.eat(playercell, len(self.parts), self.MAX_PARTS)
                self.parts.remove(playercell)
                return playercell
            else:
                cell.regurgitate_from(playercell)  # Push playercell away
        return None

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

        for part in new_parts:
            self.parts.add(part)
        return new_parts

    def explode(self, hit_cell):
        """
        Explode the hit_cell into 1 big split and 15 small splits (from eating a virus)
        :param hit_cell: cell to explode
        :return: list of new parts
        """
        max_new_parts = self.MAX_PARTS - len(self.parts)

        if max_new_parts <= 0:
            return []

        new_parts = []
        # Make the big split first
        big_split = hit_cell.split(0)
        new_parts.append(big_split)
        self.parts.add(big_split)
        max_new_parts -= 1

        # Make the remaining smaller splits
        for i in range(max_new_parts):
            if not hit_cell.able_to_split():
                break
            angle = 2 * 3.1415926535 * i / self.MAX_PARTS

            new_cell = hit_cell.emit(angle, hit_cell.SPLITCELL_SPEED, hit_cell.SPLITCELL_COND_RADIUS, PlayerCell)
            new_cell.split_timeout /= 2
            new_cell.parent = self
            new_parts.append(new_cell)
            self.parts.add(new_cell)
        return new_parts

    def center(self):
        """Returns median position of all player cells."""
        if len(self.parts) == 0:
            return [0, 0]
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
        if len(self.parts) == 0:
            return 0
        radius_sqr = functools.reduce(
            operator.add,
            (cell.radius**2 for cell in self.parts))
        score = math.sqrt(radius_sqr)
        if score > self.highest_score:
            self.highest_score = score
        return score

    def remove_part(self, cell):
        """Removes passed player cell from player parts list."""
        try:
            self.parts.remove(cell)
        except KeyError:
            pass
        if len(self.parts) == 0:
            self.alive = False

    def reset(self):
        self.parts = self.parts[:1]
        self.parts[0].area_pool = 0
        self.parts[0].radius = self.START_SIZE

    @classmethod
    def make_random(cls, nick, bounds, start_size=START_SIZE):
        """Returns random player with given nick."""
        player_cell = PlayerCell.make_random(bounds)
        player_cell.radius = start_size
        return cls(nick, player_cell)

    def __repr__(self):
        return '<{} nick={} score={}>'.format(
            self.__class__.__name__,
            self.nick,
            int(self.score()))
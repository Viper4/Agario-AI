import itertools
import time

from loguru import logger

from .entities import Cell, Virus


class Model():
    """Class that represents game state."""

    class Chunk():
        def __init__(self, players=None, cells=None):
            players = list() if players is None else players
            cells = list() if cells is None else cells
            self.players = players
            self.cells = cells

    # duration of round in seconds
    ROUND_DURATION = 240

    def __init__(self, players=None, cells=None, bounds=(1000, 1000), chunk_size=1000):
        players = list() if players is None else players
        cells = list() if cells is None else cells
        # means that size of world is [-world_size, world_size]
        self.bounds = bounds
        self.chunk_size = chunk_size
        self.chunks = list()
        for i in range((self.bounds[0] * 2) // chunk_size + 1):
            self.chunks.append(list())
            for j in range((self.bounds[1] * 2) // chunk_size + 1):
                self.chunks[-1].append(self.Chunk())

        self.num_players = 0
        self.num_cells = 0
        self.num_viruses = 0  # Just including viruses within the cells lists since its more convenient
        for player in players:
            self.add_player(player)
            self.num_players += 1
        for cell in cells:
            self.add_cell(cell)
            self.num_cells += 1

        self.round_start = time.time()

    def update_velocity(self, player, target_pos):
        """Update passed player velocity."""
        player.update_velocity(target_pos)

    def shoot(self, player, target_pos):
        """Shoots into given direction."""
        emitted_cells = player.shoot(target_pos)
        for cell in emitted_cells:
            self.add_cell(cell)

        #if emitted_cells:
        #    logger.debug(f'{player} shot')
        #else:
        #    logger.debug(f'{player} tried to shoot, but he can\'t')

    def split(self, player, target_pos):
        """Splits player."""
        self.remove_player(player)
        new_parts = player.split(target_pos)
        self.add_player(player)

        #if new_parts:
        #    logger.debug(f'{player} splitted')
        #else:
        #    logger.debug(f'{player} tried to split, but he can\'t')

    def update(self):
        """Updates game state."""
        '''if time.time() - self.round_start >= self.ROUND_DURATION:
            #logger.debug('New round was started.')
            self.__reset_players()
            self.round_start = time.time()'''

        # update cells
        for cell in self.cells:
            self.remove_cell(cell)
            cell.move()
            self.bound_cell(cell)
            self.add_cell(cell)

        # update players
        observable_players = self.players
        for player in observable_players:
            self.remove_player(player)
            player.move()
            self.bound_player(player)
            self.add_player(player)

            # get chuncks around player
            chunks = self.__nearby_chunks(player.center())
            # get objects that stored in chunks
            players = list()
            cells = list()
            for chunk in chunks:
                players.extend(chunk.players)
                cells.extend(chunk.cells)
            
            # check is player killed some cells (including viruses)
            for cell in cells:
                killed_cell, killer_cell = player.attempt_murder(cell)
                if killed_cell:
                    if isinstance(killed_cell, Virus):
                        player.explode(killer_cell)
                        self.remove_virus(killed_cell)
                    else:
                        self.remove_cell(killed_cell)

            # check is player killed other players or their parts
            for another_player in players:
                if player == another_player:
                    continue
                killed_cell, killer_cell = player.attempt_murder(another_player)
                if killed_cell:
                    if len(another_player.parts) == 1:
                        logger.debug(f'{player} ate {another_player}')
                        self.remove_player(another_player)
                        observable_players.remove(another_player)
                        another_player.remove_part(killed_cell)
                    else:
                        logger.debug(f'{player} ate {another_player} part {killed_cell}')

    def spawn_cells(self, amount):
        """Spawn passed amount of cells on the field."""
        for _ in range(amount):
            self.add_cell(Cell.make_random(self.bounds))

    def spawn_viruses(self, amount):
        """Spawn a number of viruses on the field."""
        for _ in range(amount):
            self.add_virus(Virus.make_random(self.bounds))

    def bound_cell(self, cell):
        cell.pos[0] = self.bounds[0] if cell.pos[0] > self.bounds[0] else cell.pos[0]
        cell.pos[0] = -self.bounds[0] if cell.pos[0] < -self.bounds[0] else cell.pos[0]

        cell.pos[1] = self.bounds[1] if cell.pos[1] > self.bounds[1] else cell.pos[1]
        cell.pos[1] = -self.bounds[1] if cell.pos[1] < -self.bounds[1] else cell.pos[1]

    def bound_player(self, player):
        for cell in player.parts:
            self.bound_cell(cell)

    def add_player(self, player):
        self.__pos_to_chunk(player.center()).players.append(player)

    def add_cell(self, cell):
        self.__pos_to_chunk(cell.pos).cells.append(cell)
        self.num_cells += 1

    def add_virus(self, virus):
        self.__pos_to_chunk(virus.pos).cells.append(virus)
        self.num_viruses += 1

    def remove_player(self, player):
        try:
            self.__pos_to_chunk(player.center()).players.remove(player)
            self.num_players -= 1
        except ValueError:
            pass

    def remove_cell(self, cell):
        try:
            self.__pos_to_chunk(cell.pos).cells.remove(cell)
            self.num_cells -= 1
        except ValueError:
            pass

    def remove_virus(self, virus):
        try:
            self.__pos_to_chunk(virus.pos).cells.remove(virus)
            self.num_viruses -= 1
        except ValueError:
            pass

    def copy_for_client(self, pos):
        chunks = self.__nearby_chunks(pos)
        players = list()
        cells = list()
        for chunk in chunks:
            players.extend(chunk.players)
            cells.extend(chunk.cells)

        model = Model(players, cells, self.bounds, self.chunk_size)
        model.round_start = self.round_start
        return model

    def __reset_players(self):
        for player in self.players:
            player.reset()

    def __pos_to_chunk(self, pos):
        chunk_pos = self.__chunk_pos(pos)
        return self.chunks[chunk_pos[0]][chunk_pos[1]]

    def __chunk_pos(self, pos):
        return [
            int((pos[0] + self.bounds[0]) // self.chunk_size),
            int((pos[1] + self.bounds[1]) // self.chunk_size)]

    def __nearby_chunks(self, pos):
        chunks = list()
        chunk_pos = self.__chunk_pos(pos)

        def is_valid_chunk_pos(pos):
            if pos[0] >= 0 and pos[0] < len(self.chunks) and \
                    pos[1] >= 0 and pos[1] < len(self.chunks[0]):
                return True
            return False

        for diff in itertools.product([-1, 0, 1], repeat=2):
            pos = [chunk_pos[0] + diff[0], chunk_pos[1] + diff[1]]
            if is_valid_chunk_pos(pos):
                chunks.append(self.chunks[pos[0]][pos[1]])
        
        return chunks

    @property
    def cells(self):
        cells = list()
        for chunks_line in self.chunks:
            for chunk in chunks_line:
                cells.extend(chunk.cells)
        return cells

    @property
    def players(self):
        players = list()
        for chunks_line in self.chunks:
            for chunk in chunks_line:
                players.extend(chunk.players)
        return players
    
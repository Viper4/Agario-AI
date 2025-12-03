import time
import itertools
from dataclasses import dataclass, field

from loguru import logger
from .entities import Cell, Virus, PlayerCell


@dataclass
class Chunk:
    players: list = field(default_factory=list)
    cells: list = field(default_factory=list)


class Model:
    """Efficient game-state model for Agar.io simulation."""

    ROUND_DURATION = 240  # seconds

    def __init__(self, players=None, cells=None, viruses=None,
                 bounds=(1000, 1000), chunk_size=1000, fps=60, sim_speed=1.0):

        self.bounds = bounds
        self.chunk_size = chunk_size
        self.fps = fps
        self.sim_speed = sim_speed

        # Each list indexed by object.id
        self.players = []
        self.cells = []
        self.viruses = []

        # Lists to indicate which indices are empty in each respective list
        self.empty_player_slots = []
        self.empty_cell_slots = []
        self.empty_virus_slots = []

        # Build chunk grid
        self.chunk_count_x = (bounds[0] * 2) // chunk_size + 1
        self.chunk_count_y = (bounds[1] * 2) // chunk_size + 1

        self.chunks = [
            [Chunk() for _ in range(self.chunk_count_y)]
            for _ in range(self.chunk_count_x)
        ]

        if players is not None:
            for p in players:
                self.add_player(p)
        if cells is not None:
            for c in cells:
                self.add_cell(c)
        if viruses is not None:
            for v in viruses:
                self.add_virus(v)

        self.round_start = time.time()

    # ---------------------------------------------------------------------
    # Chunking helpers
    # ---------------------------------------------------------------------

    def _chunk_coords(self, pos):
        """Convert a world position → chunk grid coords."""
        x = int((pos[0] + self.bounds[0]) // self.chunk_size)
        y = int((pos[1] + self.bounds[1]) // self.chunk_size)

        x = max(0, min(self.chunk_count_x - 1, x))
        y = max(0, min(self.chunk_count_y - 1, y))
        return x, y

    def _chunk(self, pos):
        x, y = self._chunk_coords(pos)
        return self.chunks[x][y]

    def _neighbor_chunks(self, pos):
        """Yield all 3×3 neighboring chunks around a position."""
        cx, cy = self._chunk_coords(pos)

        for dx, dy in itertools.product([-1, 0, 1], repeat=2):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < self.chunk_count_x and 0 <= ny < self.chunk_count_y:
                yield self.chunks[nx][ny]

    # ---------------------------------------------------------------------
    # Bounding helpers
    # ---------------------------------------------------------------------

    def _clamp_cell(self, cell):
        """Clamp a cell to map bounds."""
        bx, by = self.bounds
        x, y = cell.pos

        if x > bx:
            cell.pos[0] = bx
        elif x < -bx:
            cell.pos[0] = -bx

        if y > by:
            cell.pos[1] = by
        elif y < -by:
            cell.pos[1] = -by

    def _clamp_player(self, player):
        for part in player.parts:
            self._clamp_cell(part)

    # ---------------------------------------------------------------------
    # Add/remove objects
    # ---------------------------------------------------------------------

    def add_player(self, player):
        if len(self.empty_player_slots) == 0:
            # Append to end of list
            new_id = len(self.players)
            self.players.append(player)
        else:
            # Insert at first empty slot
            new_id = self.empty_player_slots.pop()
            self.players[new_id] = player
        player.id = new_id
        self._chunk(player.center()).players.append(player)

    def remove_player(self, player):
        if self.players[player.id] is None:
            return  # Player is already removed
        self.players[player.id] = None
        self.empty_player_slots.append(player.id)
        chunk = self._chunk(player.center())
        if player in chunk.players:
            chunk.players.remove(player)

    def add_cell(self, cell):
        if len(self.empty_cell_slots) == 0:
            # Append to end of list
            new_id = len(self.cells)
            self.cells.append(cell)
        else:
            # Insert at first empty slot
            new_id = self.empty_cell_slots.pop()
            self.cells[new_id] = cell
        cell.id = new_id
        self._chunk(cell.pos).cells.append(cell)

    def remove_cell(self, cell):
        if self.cells[cell.id] is None:
            return  # Cell is already removed
        self.cells[cell.id] = None
        self.empty_cell_slots.append(cell.id)
        chunk = self._chunk(cell.pos)
        if cell in chunk.cells:
            chunk.cells.remove(cell)

    def add_virus(self, virus):
        if len(self.empty_virus_slots) == 0:
            # Append to end of list
            new_id = len(self.viruses)
            self.viruses.append(virus)
        else:
            # Insert at first empty slot
            new_id = self.empty_virus_slots.pop()
            self.viruses[new_id] = virus
        virus.id = new_id
        self._chunk(virus.pos).cells.append(virus)

    def remove_virus(self, virus):
        if self.viruses[virus.id] is None:
            return  # Virus is already removed
        self.viruses[virus.id] = None
        self.empty_virus_slots.append(virus.id)
        chunk = self._chunk(virus.pos)
        if virus in chunk.cells:
            chunk.cells.remove(virus)

    # ---------------------------------------------------------------------
    # Game actions
    # ---------------------------------------------------------------------

    def update_velocity(self, player, target_pos):
        player.update_velocity(target_pos)

    def shoot(self, player, target_pos):
        emitted = player.shoot(target_pos)
        for c in emitted:
            self.add_cell(c)

    def split(self, player, target_pos):
        self.remove_player(player)
        new_parts = player.split(target_pos)
        self.add_player(player)

    # ---------------------------------------------------------------------
    # Simulation update
    # ---------------------------------------------------------------------

    def update(self):
        """Advance game world by one tick."""

        # ---------- Update cells ----------
        for cell in self.cells:
            if cell is None:
                continue
            old_chunk = self._chunk(cell.pos)

            cell.move(self.sim_speed)
            self._clamp_cell(cell)

            new_chunk = self._chunk(cell.pos)
            if new_chunk is not old_chunk:
                old_chunk.cells.remove(cell)
                new_chunk.cells.append(cell)

        # ---------- Update viruses ----------
        for virus in self.viruses:
            if virus is None:
                continue
            old_chunk = self._chunk(virus.pos)

            virus.move(self.sim_speed)
            self._clamp_cell(virus)

            new_chunk = self._chunk(virus.pos)
            if new_chunk is not old_chunk:
                old_chunk.cells.remove(virus)
                new_chunk.cells.append(virus)

            # Split virus when it eats ejected mass
            nearby_cells = []
            for chunk in self._neighbor_chunks(virus.pos):
                nearby_cells.extend(chunk.cells)

            # Eat ejected cells
            for cell in nearby_cells:
                if cell.radius != PlayerCell.SHOOTCELL_RADIUS or isinstance(cell, PlayerCell) or isinstance(cell, Virus):
                    continue
                if cell.try_to_kill_by(virus):
                    virus.eat(cell)
                    new_virus = virus.try_split(cell.angle, cell.speed)
                    self.remove_cell(cell)
                    if new_virus:
                        self.add_virus(new_virus)

        # ---------- Update players ----------
        for player in self.players:
            if player is None:
                continue
            old_chunk = self._chunk(player.center())

            player.move(self.sim_speed)
            self._clamp_player(player)

            new_chunk = self._chunk(player.center())
            if new_chunk is not old_chunk:
                try:
                    old_chunk.players.remove(player)
                except ValueError:
                    pass
                new_chunk.players.append(player)

            # Nearby objects for collisions
            nearby_players = []
            nearby_cells = []
            for chunk in self._neighbor_chunks(player.center()):
                nearby_players.extend(chunk.players)
                nearby_cells.extend(chunk.cells)

            # Eat cells & viruses
            for cell in nearby_cells:
                killed, killer = player.attempt_murder(cell)
                if killed:
                    if isinstance(killed, Virus):
                        player.explode(killer)
                        self.remove_virus(killed)
                    else:
                        self.remove_cell(killed)

            # Eat other players / parts
            for other in nearby_players:
                if other is player:
                    continue
                killed, killer = player.attempt_murder(other)
                if killed:
                    if len(other.parts) == 1:
                        #logger.debug(f"{player} ate {other}")
                        self.remove_player(other)
                    else:
                        #logger.debug(f"{player} ate a part of {other}")
                        other.remove_part(killed)

    # ---------------------------------------------------------------------
    # Spawning
    # ---------------------------------------------------------------------

    def spawn_cells(self, n):
        for _ in range(n):
            c = Cell.make_random(self.fps, self.bounds)
            self.add_cell(c)

    def spawn_viruses(self, n):
        for _ in range(n):
            v = Virus.make_random(self.fps, self.bounds)
            self.add_virus(v)

    # ---------------------------------------------------------------------
    # Client copy (view around a position)
    # ---------------------------------------------------------------------

    def copy_for_client(self, pos):
        """Return a small model containing only objects near pos."""
        chunks = list(self._neighbor_chunks(pos))

        players = list(itertools.chain.from_iterable(c.players for c in chunks))
        cells = list(itertools.chain.from_iterable(c.cells for c in chunks))

        m = Model(players, cells, self.bounds, self.chunk_size)
        m.round_start = self.round_start
        return m

    def player_count(self):
        return len(self.players) - len(self.empty_player_slots)

    def cell_count(self):
        return len(self.cells) - len(self.empty_cell_slots)

    def virus_count(self):
        return len(self.viruses) - len(self.empty_virus_slots)

import time
import itertools
from dataclasses import dataclass, field

from .entities import Cell, Virus, PlayerCell


@dataclass
class Chunk:
    players: set = field(default_factory=set)
    cells: set = field(default_factory=set)
    viruses: set = field(default_factory=set)


class Model:
    """Efficient game-state model for Agar.io simulation."""

    ROUND_DURATION = 240  # seconds

    def __init__(self, players=None, cells=None, viruses=None,
                 bounds=(1000, 1000), chunk_size=1000):

        self.bounds = bounds
        self.chunk_size = chunk_size

        # Each list indexed by object.id
        self.num_players = 0
        self.num_cells = 0
        self.num_viruses = 0

        # Build chunk grid
        self.chunk_count_x = (bounds[0] * 2) // chunk_size + 1
        self.chunk_count_y = (bounds[1] * 2) // chunk_size + 1

        self.chunks = list()
        for _ in range(self.chunk_count_x):
            self.chunks.append([])
            for _ in range(self.chunk_count_y):
                self.chunks[-1].append(Chunk())

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
        """Convert a world position to chunk grid coords."""
        x = int((pos[0] + self.bounds[0]) // self.chunk_size)
        y = int((pos[1] + self.bounds[1]) // self.chunk_size)

        x = max(0, min(self.chunk_count_x - 1, x))
        y = max(0, min(self.chunk_count_y - 1, y))
        return x, y

    def _chunk(self, pos):
        """Get the chunk at a world position."""
        x, y = self._chunk_coords(pos)
        return self.chunks[x][y]

    def _overlap_chunks(self, cells: list):
        """Yield all chunks whose bounding boxes intersect the given cells."""
        min_x = 10000000
        max_x = -10000000
        min_y = 10000000
        max_y = -10000000
        for cell in cells:
            min_x = min(min_x, cell.pos[0] - cell.radius)
            max_x = max(max_x, cell.pos[0] + cell.radius)
            min_y = min(min_y, cell.pos[1] - cell.radius)
            max_y = max(max_y, cell.pos[1] + cell.radius)

        # Convert world bounds to chunk coordinates
        start_cx, start_cy = self._chunk_coords((min_x, min_y))
        end_cx, end_cy = self._chunk_coords((max_x, max_y))

        # Iterate only the chunks inside that region
        for cx in range(start_cx, end_cx + 1):
            for cy in range(start_cy, end_cy + 1):
                yield self.chunks[cx][cy]

    def get_overlap_chunks(self, cells):
        return list(self._overlap_chunks(cells))

    def get_chunks(self, bounds: tuple[tuple[int, int], tuple[int, int]]):
        """
        Get all chunks that overlap the given bounds.
        :param bounds: (top_left, bottom_right)
        :return:
        """
        min_x = bounds[0][0]
        max_x = bounds[1][0]
        min_y = bounds[1][1]
        max_y = bounds[0][1]

        # Convert world bounds to chunk coordinates
        start_cx, start_cy = self._chunk_coords((min_x, min_y))
        end_cx, end_cy = self._chunk_coords((max_x, max_y))

        # Iterate only the chunks inside that region
        for cx in range(start_cx, end_cx + 1):
            for cy in range(start_cy, end_cy + 1):
                yield self.chunks[cx][cy]

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
        self._chunk(player.center()).players.add(player)
        self.num_players += 1

    def remove_player(self, player):
        chunk = self._chunk(player.center())
        if player in chunk.players:
            player.alive = False
            chunk.players.remove(player)
            self.num_players -= 1

    def add_cell(self, cell):
        self._chunk(cell.pos).cells.add(cell)
        self.num_cells += 1

    def remove_cell(self, cell):
        chunk = self._chunk(cell.pos)
        if cell in chunk.cells:
            chunk.cells.remove(cell)
            self.num_cells -= 1
        else:
            print("Cell not found in chunk")

    def add_virus(self, virus):
        self._chunk(virus.pos).viruses.add(virus)
        self.num_viruses += 1

    def remove_virus(self, virus):
        chunk = self._chunk(virus.pos)
        if virus in chunk.viruses:
            chunk.viruses.remove(virus)
            self.num_viruses -= 1

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
            chunk = self._chunk(cell.pos)

            cell.move()
            self._clamp_cell(cell)

            new_chunk = self._chunk(cell.pos)
            if new_chunk is not chunk:
                chunk.cells.remove(cell)
                new_chunk.cells.add(cell)

        # ---------- Update viruses ----------
        for virus in self.viruses:
            chunk = self._chunk(virus.pos)

            virus.move()
            self._clamp_cell(virus)

            new_chunk = self._chunk(virus.pos)
            if new_chunk is not chunk:
                chunk.viruses.remove(virus)
                new_chunk.viruses.add(virus)

            nearby_cells = []
            for nc in self._overlap_chunks([virus]):
                nearby_cells.extend(nc.cells)

            # Eat ejected cells
            for cell in nearby_cells:
                if cell.radius != PlayerCell.SHOOTCELL_RADIUS or isinstance(cell, PlayerCell):
                    continue
                if cell.try_to_kill_by(virus):
                    virus.eat(cell)
                    new_virus = virus.try_split(cell.angle, cell.speed)
                    self.remove_cell(cell)
                    if new_virus:
                        self.add_virus(new_virus)

        # ---------- Update players ----------
        for player in self.players:
            chunk = self._chunk(player.center())

            player.move()
            self._clamp_player(player)

            new_chunk = self._chunk(player.center())
            if new_chunk is not chunk:
                try:
                    chunk.players.remove(player)
                except KeyError:
                    pass
                new_chunk.players.add(player)

            nearby_cells = []
            nearby_players = []
            nearby_viruses = []
            # Nearby objects for collisions
            for nc in self._overlap_chunks(player.parts):
                nearby_cells.extend(nc.cells)
                nearby_players.extend(nc.players)
                nearby_viruses.extend(nc.viruses)

            # Eat cells
            for cell in nearby_cells:
                killed, killer = player.attempt_murder(cell)
                if killed:
                    self.remove_cell(killed)

            # Eat viruses
            for virus in nearby_viruses:
                killed, killer = player.attempt_murder(virus)
                if killed:
                    player.explode(killer)
                    self.remove_virus(killed)

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
            c = Cell.make_random(self.bounds)
            self.add_cell(c)

    def spawn_viruses(self, n):
        for _ in range(n):
            v = Virus.make_random(self.bounds)
            self.add_virus(v)

    # ---------------------------------------------------------------------
    # Client copy (view around a position)
    # ---------------------------------------------------------------------

    def copy_for_client(self, cell):
        """Return a small model containing only objects near pos."""
        chunks = list(self._overlap_chunks([cell]))

        players = list(itertools.chain.from_iterable(c.players for c in chunks))
        cells = list(itertools.chain.from_iterable(c.cells for c in chunks))

        m = Model(players, cells, self.bounds, self.chunk_size)
        m.round_start = self.round_start
        return m

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

    @property
    def viruses(self):
        viruses = list()
        for chunks_line in self.chunks:
            for chunk in chunks_line:
                viruses.extend(chunk.viruses)
        return viruses

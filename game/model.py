import time
from dataclasses import dataclass, field

from .entities import Cell, Virus, PlayerCell


@dataclass
class Chunk:
    playercells: set = field(default_factory=set)
    cells: set = field(default_factory=set)
    viruses: set = field(default_factory=set)


class Model:
    """
    Efficient game-state model for Agar.io simulation.
    """
    def __init__(self, players=None, cells=None, viruses=None,
                 bounds=(1000, 1000), chunk_size=1000):

        self.bounds = bounds
        self.chunk_size = chunk_size

        # Each list indexed by object.id
        self.num_player_cells = 0
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
                for pc in p.parts:
                    self.add_playercell(pc)
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

    def _chunk_coords(self, pos: tuple[float, float]):
        """
        Convert a world position to chunk grid coords.
        :param pos:
        :return:
        """
        x = int((pos[0] + self.bounds[0]) // self.chunk_size)
        y = int((pos[1] + self.bounds[1]) // self.chunk_size)

        x = max(0, min(self.chunk_count_x - 1, x))
        y = max(0, min(self.chunk_count_y - 1, y))
        return x, y

    def _chunk(self, pos: tuple[float, float]):
        """
        Get the chunk at a world position.
        :param pos:
        :return:
        """
        x, y = self._chunk_coords(pos)
        return self.chunks[x][y]

    def _overlap_chunks(self, cell):
        """
        Yield all chunks whose bounding boxes intersect the given cell.
        :param cell:
        :return:
        """
        min_x = cell.pos[0] - cell.radius
        max_x = cell.pos[0] + cell.radius
        min_y = cell.pos[1] - cell.radius
        max_y = cell.pos[1] + cell.radius

        # Convert world bounds to chunk coordinates
        start_cx, start_cy = self._chunk_coords((min_x, min_y))
        end_cx, end_cy = self._chunk_coords((max_x, max_y))

        # Iterate only the chunks inside that region
        for cx in range(start_cx, end_cx + 1):
            for cy in range(start_cy, end_cy + 1):
                yield self.chunks[cx][cy]

    def get_overlap_chunks(self, cell):
        """
        Return a list of chunks whose bounding boxes intersect the given cell.
        :param cell:
        :return:
        """
        return list(self._overlap_chunks(cell))

    def get_chunks(self, bounds: tuple[tuple[float, float], tuple[float, float]]):
        """
        Yield all chunks that overlap the given bounds.
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
        """
        Clamp a cell to map bounds.
        :param cell:
        :return:
        """
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

    # ---------------------------------------------------------------------
    # Add/remove objects
    # ---------------------------------------------------------------------

    def add_playercell(self, playercell: PlayerCell):
        """
        Adds the given playercell to the chunk it's in, updates its cx, cy, marks its player alive, and increments the player cell count.
        :param playercell:
        :return:
        """
        cx, cy = self._chunk_coords(playercell.pos)
        playercell.cx = cx
        playercell.cy = cy
        chunk = self.chunks[cx][cy]
        chunk.playercells.add(playercell)
        playercell.parent.alive = True
        self.num_player_cells += 1

    def remove_playercell(self, playercell: PlayerCell):
        """
        Removes the given playercell from its cx, cy chunk and decrements player cell count.
        :param playercell:
        :return:
        """
        chunk = self.chunks[playercell.cx][playercell.cy]
        if playercell in chunk.playercells:
            chunk.playercells.remove(playercell)
            self.num_player_cells -= 1

    def add_cell(self, cell):
        """
        Adds the given cell to the chunk it's centered in, and increments the cell count.
        :param cell:
        :return:
        """
        cx, cy = self._chunk_coords(cell.pos)
        cell.cx = cx
        cell.cy = cy
        chunk = self.chunks[cx][cy]
        chunk.cells.add(cell)
        self.num_cells += 1

    def remove_cell(self, cell):
        """
        Removes the given cell from the chunk it's centered in, and decrements the cell count.
        :param cell:
        :return:
        """
        chunk = self.chunks[cell.cx][cell.cy]
        if cell in chunk.cells:
            chunk.cells.remove(cell)
            self.num_cells -= 1

    def add_virus(self, virus):
        """
        Adds the given virus to the chunk it's centered in, and increments the virus count.
        :param virus:
        :return:
        """
        cx, cy = self._chunk_coords(virus.pos)
        virus.cx = cx
        virus.cy = cy
        chunk = self.chunks[cx][cy]
        chunk.viruses.add(virus)
        self.num_viruses += 1

    def remove_virus(self, virus):
        """
        Removes the given virus from the chunk it's centered in, and decrements the virus count.
        :param virus:
        :return:
        """
        chunk = self.chunks[virus.cx][virus.cy]
        if virus in chunk.viruses:
            chunk.viruses.remove(virus)
            self.num_viruses -= 1

    # ---------------------------------------------------------------------
    # Game actions
    # ---------------------------------------------------------------------

    def update_velocity(self, player, target_pos):
        """
        Updates the velocity of the player given the target position to move to.
        :param player:
        :param target_pos:
        :return:
        """
        player.update_velocity(target_pos)

    def shoot(self, player, target_pos):
        """
        Shoots a ejected cell from the player towards the target position.
        :param player:
        :param target_pos:
        :return:
        """
        emitted = player.shoot(target_pos)
        for c in emitted:
            self.add_cell(c)

    def split(self, player, target_pos):
        """
        Splits the player towards target position.
        :param player:
        :param target_pos:
        :return:
        """
        for pc in player.parts:
            self.remove_playercell(pc)  # Remove parts from old chunks
        player.split(target_pos)
        for p in player.parts:
            self.add_playercell(p)  # Add all parts back to chunks

    # ---------------------------------------------------------------------
    # Simulation update
    # ---------------------------------------------------------------------

    def move_cell(self, cell):
        """
        Moves the cell and updates its old chunk and new chunk.
        :param cell:
        :return:
        """
        old_cx = cell.cx
        old_cy = cell.cy

        cell.move()
        self._clamp_cell(cell)

        new_cx, new_cy = self._chunk_coords(cell.pos)
        if old_cx != new_cx or old_cy != new_cy:
            if isinstance(cell, PlayerCell):
                self.remove_playercell(cell)  # Remove from old chunk
                self.add_playercell(cell)  # Add to new chunk and update cx, cy
            elif isinstance(cell, Virus):
                self.remove_virus(cell)  # Remove from old chunk
                self.add_virus(cell)  # Add to new chunk and update cx, cy
            else:
                self.remove_cell(cell)  # Remove from old chunk
                self.add_cell(cell)  # Add to new chunk and update cx, cy

    def update(self):
        """
        Advances the game world by one tick.
        :return:
        """
        # ---------- Update cells ----------
        for cell in self.cells:
            self.move_cell(cell)

        # ---------- Update viruses ----------
        for virus in self.viruses:
            self.move_cell(virus)

            nearby_cells = []
            for nc in self._overlap_chunks(virus):
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
        for playercell in self.playercells:
            if not playercell.parent.alive:
                self.remove_playercell(playercell)
                continue

            self.move_cell(playercell)

            merged_cell = playercell.check_merge()
            if merged_cell:
                if merged_cell is playercell:
                    self.remove_playercell(playercell)
                    continue  # Skip rest of the update for this playercell

            nearby_cells = []
            nearby_playercells = []
            nearby_viruses = []
            # Nearby objects for collisions
            for nc in self._overlap_chunks(playercell):
                nearby_cells.extend(nc.cells)
                nearby_playercells.extend(nc.playercells)
                nearby_viruses.extend(nc.viruses)

            # Eat cells
            for cell in nearby_cells:
                killed = playercell.attempt_murder(cell)
                if killed:
                    self.remove_cell(killed)

            # Eat viruses
            for virus in nearby_viruses:
                killed = playercell.attempt_murder(virus)
                if killed:
                    new_parts = playercell.parent.explode(playercell)
                    for p in new_parts:
                        self.add_playercell(p)
                    self.remove_virus(killed)

            # Eat other players / parts
            for other_pc in nearby_playercells:
                if other_pc.parent is playercell.parent:
                    continue  # Cant eat itself
                killed = playercell.attempt_murder(other_pc)
                if killed:
                    other_pc.parent.remove_part(killed)
                    self.remove_playercell(killed)

    # ---------------------------------------------------------------------
    # Spawning
    # ---------------------------------------------------------------------

    def spawn_cells(self, n):
        """
        Spawns n cells randomly distributed within the map bounds.
        :param n:
        :return:
        """
        for _ in range(n):
            c = Cell.make_random(self.bounds)
            self.add_cell(c)

    def spawn_viruses(self, n):
        """
        Spawns n viruses randomly distributed within the map bounds.
        :param n:
        :return:
        """
        for _ in range(n):
            v = Virus.make_random(self.bounds)
            self.add_virus(v)

    @property
    def cells(self):
        """
        Returns a list of all cells in the model.
        :return:
        """
        cells = list()
        for chunks_line in self.chunks:
            for chunk in chunks_line:
                cells.extend(chunk.cells)
        return cells

    @property
    def playercells(self):
        """
        Returns a list of all playercells in the model.
        :return:
        """
        playercells = list()
        for chunks_line in self.chunks:
            for chunk in chunks_line:
                playercells.extend(chunk.playercells)
        return playercells

    @property
    def viruses(self):
        """
        Returns a list of all viruses in the model.
        :return:
        """
        viruses = list()
        for chunks_line in self.chunks:
            for chunk in chunks_line:
                viruses.extend(chunk.viruses)
        return viruses

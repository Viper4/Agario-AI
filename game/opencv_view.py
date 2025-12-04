import cv2
import numpy as np
import time
import math


class OCCamera:
    """Camera that converts world positions to screen pixel positions."""
    def __init__(self, width, height, scale=1.0):
        self.width = width
        self.height = height
        self.scale = scale
        self.x = 0
        self.y = 0

    def set_center(self, pos):
        # same logic as pygame version
        self.x = pos[0] - self.width * 0.5 / self.scale
        self.y = pos[1] + self.height * 0.5 / self.scale

    def world_to_screen(self, pos):
        """World → pixel coordinates (x, y)."""
        sx = int((pos[0] - self.x) * self.scale)
        sy = int((self.y - pos[1]) * self.scale)
        return (sx, sy)


class OCView:
    """High-performance OpenCV renderer for training visualization."""

    BG_COLOR = (242, 251, 255)     # RGB originally
    GRID_COLOR = (226, 234, 238)
    DEBUG_COLOR = (0, 0, 255)      # red
    GRID_STEP = 25

    def __init__(self, width, height, model, target_player, debug=False, scale=1.0):
        self.width = width
        self.height = height
        self.model = model
        self.player = target_player
        self.debug = debug

        self.camera = OCCamera(width, height, scale)

        # OpenCV expects BGR not RGB, convert once
        self.bg_bgr = self._rgb_to_bgr(self.BG_COLOR)
        self.grid_bgr = self._rgb_to_bgr(self.GRID_COLOR)

    # ---------------- Utility -----------------

    def _rgb_to_bgr(self, color):
        return (color[2], color[1], color[0])

    def _cell_color_bgr(self, color_tuple):
        return (color_tuple[2], color_tuple[1], color_tuple[0])

    # ---------------- Rendering -----------------

    def draw_grid(self, frame):
        (world_w, world_h) = self.model.bounds
        cam = self.camera

        for i in range(-world_w, world_w + self.GRID_STEP, self.GRID_STEP):
            # horizontal line
            p1 = cam.world_to_screen(( -world_w,  i ))
            p2 = cam.world_to_screen((  world_w,  i ))
            cv2.line(frame, p1, p2, self.grid_bgr, 1)

            # vertical line
            p3 = cam.world_to_screen(( i, -world_h ))
            p4 = cam.world_to_screen(( i,  world_h ))
            cv2.line(frame, p3, p4, self.grid_bgr, 1)

    def draw_cell(self, frame, cell):
        pos = self.camera.world_to_screen(cell.pos)
        color = self._cell_color_bgr(cell.color)

        cv2.circle(frame, pos, int(cell.radius * self.camera.scale), color, -1)

    def draw_player(self, frame, player):
        for part in player.parts:
            self.draw_cell(frame, part)

    def draw_debug(self, frame):
        # Draw center point
        center = self.camera.world_to_screen(self.player.center())
        cv2.circle(frame, center, 5, self.DEBUG_COLOR, -1)

        # Draw velocity vectors
        for cell in self.player.parts:
            dx, dy = gu.polar_to_cartesian(cell.angle, cell.speed * 100)
            start = self.camera.world_to_screen(cell.pos)
            end = self.camera.world_to_screen((cell.pos[0] + dx, cell.pos[1] + dy))

            cv2.line(frame, start, end, self.DEBUG_COLOR, 2)
            cv2.circle(frame, end, 3, self.DEBUG_COLOR, -1)

    # ---------------- Main redraw -----------------

    def redraw(self):
        if not self.player.alive:
            return

        # Update camera position
        self.camera.set_center(self.player.center())

        # Create blank frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = self.bg_bgr

        # Grid
        self.draw_grid(frame)

        # Cells
        for cell in self.model.cells:
            if cell:
                self.draw_cell(frame, cell)

        # Viruses
        for virus in self.model.viruses:
            if virus:
                self.draw_cell(frame, virus)

        # Players
        for pl in self.model.players:
            if pl:
                self.draw_player(frame, pl)

        # Debug
        if self.debug:
            self.draw_debug(frame)

        cv2.imshow("Training View", frame)
        cv2.waitKey(1)  # non-blocking

import cv2
import numpy as np
import pyautogui

from game import gameutils as gu


class OCCamera:
    """
    Camera that converts world positions to screen pixel positions.
    """
    def __init__(self, width, height, scale=1.0):
        self.width = width
        self.height = height
        self.scale = scale
        self.x = 0
        self.y = 0

    def set_center(self, pos):
        """
        Changes position of the camera to be centered at given pos
        :param pos: tuple(x, y)
        :return:
        """
        # same logic as pygame version
        self.x = pos[0] - self.width * 0.5 / self.scale
        self.y = pos[1] + self.height * 0.5 / self.scale

    def world_to_screen(self, pos):
        """
        Converts world position to pixel coordinates
        :param pos: World position tuple (x, y)
        :return: Screen position tuple (sx, sy)
        """
        sx = int((pos[0] - self.x) * self.scale)
        sy = int((self.y - pos[1]) * self.scale)
        return sx, sy

    @staticmethod
    def get_inverse_scale(score):
        """
        Calculates the inverse of camera scale given score.
        :param score:
        :return:
        """
        return 0.75 + score * 0.004


class OCView:
    """
    High-performance OpenCV renderer for training visualization.
    """

    BG_COLOR = (242, 251, 255)  # RGB originally
    GRID_COLOR = (226, 234, 238)
    DEBUG_COLOR = (0, 0, 255)  # red
    GRID_STEP = 25

    def __init__(self, width, height, model, target_player, debug=False, scale=1.0):
        self.base_width = self.width = width
        self.base_height = self.height = height
        self.model = model
        self.player = target_player
        self.player_index = 0
        self.debug = debug

        self.camera = OCCamera(width, height, scale)

        # OpenCV expects BGR not RGB, convert once
        self.bg_bgr = self._rgb_to_bgr(self.BG_COLOR)
        self.grid_bgr = self._rgb_to_bgr(self.GRID_COLOR)

    # ---------------- Utility -----------------

    def _rgb_to_bgr(self, color):
        """
        Converts RGB color tuple to BGR color tuple
        :param color: tuple(R, G, B)
        :return: tuple(B, G, R)
        """
        return color[2], color[1], color[0]

    # ---------------- Rendering -----------------

    def draw_grid(self, frame):
        """
        Draws the grid lines in the background like the real Agar.io
        :param frame: Frame to draw on
        :return:
        """
        (world_w, world_h) = self.model.bounds
        cam = self.camera

        for i in range(-world_w, world_w + self.GRID_STEP, self.GRID_STEP):
            # horizontal line
            p1 = cam.world_to_screen((-world_w, i))
            p2 = cam.world_to_screen((world_w, i))
            cv2.line(frame, p1, p2, self.grid_bgr, 1)

            # vertical line
            p3 = cam.world_to_screen((i, -world_h))
            p4 = cam.world_to_screen((i, world_h))
            cv2.line(frame, p3, p4, self.grid_bgr, 1)

    def draw_cell(self, frame, cell):
        """
        Draws a single cell to the frame
        :param frame: Frame to draw on
        :param cell: Cell object to draw
        :return:
        """
        pos = self.camera.world_to_screen(cell.pos)
        color = self._rgb_to_bgr(cell.color)

        cv2.circle(frame, pos, int(cell.radius * self.camera.scale), color, -1)

    def draw_debug(self, frame):
        """
        Draw debug information on the screen.
        :param frame: Frame to draw on
        :return:
        """
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

    # ---------------- User events -----------------

    def game_mouse_pos(self):
        """
        Returns the mouse position in game coordinates.
        :return: tuple(x, y)
        """
        x, y = pyautogui.position()

        # convert to local position relative to game window
        win_x, win_y = cv2.getWindowImageRect("Game")[:2]  # top-left corner
        local_x = x - win_x
        local_y = win_y - y  # window y is inverted
        # Camera x, y is in the top left corner of the view at scale=1.0
        return self.camera.x + local_x / self.camera.scale, self.camera.y + local_y / self.camera.scale

    def poll_spectate_keys(self, playercells):
        """
        Check for key events in spectate mode.
        :param playercells: List of playercells to switch between
        :return:
        """
        # Poll key press (non-blocking)
        key = cv2.waitKeyEx(1)

        if key == 2424832:  # Left arrow
            self.player_index -= 1
            if self.player_index < 0:
                self.player_index = len(playercells) - 1
            self.player = playercells[self.player_index].parent
        elif key == 2555904:  # Right arrow
            self.player_index += 1
            if self.player_index >= len(playercells):
                self.player_index = 0
            self.player = playercells[self.player_index].parent

    def poll_human_keys(self):
        """
        Check for key events in human mode.
        :return:
        """
        # Poll key press (non-blocking)
        key = cv2.waitKeyEx(1)

        mouse_pos = self.game_mouse_pos()

        if key == ord('w'):
            self.model.shoot(
                self.player,
                mouse_pos)
        elif key == ord(' '):
            self.model.split(
                self.player,
                mouse_pos)

        # Move player
        self.model.update_velocity(self.player, mouse_pos)

    # ---------------- Main redraw -----------------

    def redraw(self, spectate_mode: bool):
        """
        Draws all cells, players, and viruses to the frame from the model.
        :param spectate_mode: Whether user is spectating or playing
        :return:
        """
        if self.player is None:
            return

        # Expand view based on score
        score = self.player.score()
        # Scale shrinks from 1.0 to maybe 0.25 as score grows
        self.camera.scale = 1.0 / self.camera.get_inverse_scale(score)

        # Clamp so you can’t zoom infinitely far
        self.camera.scale = max(self.camera.scale, 0.25)

        # Update camera position
        self.camera.set_center(self.player.center())

        # Create blank frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = self.bg_bgr

        # Grid
        self.draw_grid(frame)

        # Cells
        for cell in self.model.cells:
            self.draw_cell(frame, cell)

        # Viruses
        for virus in self.model.viruses:
            self.draw_cell(frame, virus)

        # Players
        playercells = self.model.playercells
        for pc in playercells:
            self.draw_cell(frame, pc)

        # Debug
        if self.debug:
            self.draw_debug(frame)

        if spectate_mode:
            cv2.imshow("Spectating", frame)
            self.poll_spectate_keys(playercells)
        else:
            cv2.imshow("Game", frame)
            self.poll_human_keys()


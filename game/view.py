import math
import time

import pygame
import pygame.gfxdraw

from . import gameutils as gu
from .model import Model
from .entities import Player, Virus, PlayerCell
from geometry_utils import Vector, GameObject
from image_processing import ImageProcessing
import agent


class Camera(object):
    """Class that converts cartesian pos to pixel pos on the screen."""

    def __init__(self, x, y, width, height, scale=1.0):
        # top left point of camera box
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.scale = 1.0

    def set_center(self, pos):
        """Change camera postion according to passed center."""
        self.x = pos[0] - self.width*0.5
        self.y = pos[1] + self.height*0.5

    def adjust(self, pos):
        """Convert cartesian pos to pos relative to the camera."""
        return pos[0]*self.scale - self.x, self.y - pos[1]*self.scale


class View():
    """"Class that displays model state and shows HUD"""

    TEXT_COLOR = (50, 50, 50)
    HUD_BACGROUND_COLOR = (50,50,50,80)
    BACKGROUND_COLOR = (242, 251, 255)
    MESSAGE_COLOR = (255, 0, 0)
    GRID_COLOR = (226, 234, 238)
    HUD_PADDING = (3, 3)
    FONT_SIZE = 18
    MESSAGE_EXPIRE_TIME = 5

    DEBUG_COLOR = (255, 0, 0)

    def __init__(self, screen, model, player, debug=False):
        self.screen = screen
        self.width, self.height = self.screen.get_size()
        self.model = model
        self.target_player = player
        self.debug = debug
        self.camera = Camera(0, 0, self.width, self.height)
        self.fps = 60
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.hud_surface = pygame.Surface((1, 1), pygame.SRCALPHA)
        self.hud_surface.fill(View.HUD_BACGROUND_COLOR)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 18)

    def expand_camera(self):
        """Expand camera view bounds as player's cell grows or splits."""
        pass

    def redraw(self):
        """Redraw screen according to model of game."""
        if not self.target_player.alive:
            return
        self.camera.set_center(self.target_player.center())
        self.screen.fill(View.BACKGROUND_COLOR)
        self.draw_grid()
        for cell in self.model.cells:
            self.draw_cell(cell)
        for player in self.model.players:
            self.draw_player(player)
        # self.draw_object(self.model.player)
        self.draw_hud((8, 5))
        if self.debug:
            self.draw_debug_info()
        self.draw_messages()
        pygame.display.flip()

    def draw_messages(self):
        if time.time() - self.model.round_start <= self.MESSAGE_EXPIRE_TIME:
            self.draw_text(
                self.screen,
                "New round started!",
                [self.width // 2, self.height // 2 * 0.1],
                self.MESSAGE_COLOR,
                align_center=True)  

    def draw_grid(self, step=25):
        """Draw grid on screen with passed step."""
        world_size = self.model.bounds[0]
        for i in range(-world_size, world_size+step, step):
            start_coord = (-world_size, i)
            end_coord = (world_size, i)
            pygame.draw.line(
                self.screen, 
                View.GRID_COLOR, 
                self.camera.adjust(start_coord), 
                self.camera.adjust(end_coord), 
                2)
            pygame.draw.line(
                self.screen, 
                View.GRID_COLOR, 
                self.camera.adjust(start_coord[::-1]), 
                self.camera.adjust(end_coord[::-1]), 
                2)

    def draw_cell(self, cell):
        """Draw passed cell on the screen"""
        # draw filled circle
        pygame.draw.circle(
            self.screen,
            cell.color,
            self.camera.adjust(cell.pos),
            cell.radius)
        
        # draw circle border
        if cell.BORDER_WIDTH != 0:
            pygame.draw.circle(
                self.screen,
                gu.make_border_color(cell.color),
                self.camera.adjust(cell.pos),
                cell.radius,
                cell.BORDER_WIDTH)

    def draw_player(self, player):
        """Draw passed player on the screen."""
        for cell in player.parts:
            # draw player part
            self.draw_cell(cell)
            # draw nickname on top of the part
            self.draw_text(
                self.screen,
                player.nick,
                self.camera.adjust(cell.pos),
                align_center=True)

    def draw_text(self, surface, text, pos, color=TEXT_COLOR, align_center=False):
        """Draw passed text on passed surface."""
        text_surface = self.font.render(text, True, color)
        pos = list(pos)
        if align_center:
            # offset pos if was passed center
            pos[0] -= text_surface.get_width() // 2
            pos[1] -= text_surface.get_height() // 2
        surface.blit(text_surface, pos)

    def draw_hud(self, padding):
        """Draw score and top players HUDs."""
        # draw score HUD item
        score_text = 'Score: {:6}'.format(int(self.target_player.score()))
        self.draw_hud_item(
             (15, self.height - 30 - 2*padding[1]),
             (score_text,),
             10,
             padding)
        # draw leaderboard HUD item
        lines = list()
        lines.append('Leaderboard')
        top10 = sorted(
            self.model.players,
            key=lambda pl: pl.score(),
            reverse=True)[:10]
        for i, player in enumerate(top10):
            lines.append('{}. {} ({})'.format(i + 1, player.nick, round(player.score(), 2)))
        self.draw_hud_item(
             (self.width - 150, 15),
             lines,
             10,
             padding)

    def draw_hud_item(self, pos, lines, maxchars, padding):
        """Draw HUD item with passed string lines."""
        # seacrh max line width
        max_width = max(map(lambda line: self.font.size(line)[0], lines))
        font_height = self.font.get_height()
        # size of HUD item background
        item_size = (
            max_width + 2*padding[0], 
            font_height*len(lines) + 2*padding[1])
        # scaling transparent HUD background
        item_surface = pygame.transform.scale(self.hud_surface, item_size)
        # draw each line
        for i, line in enumerate(lines):
            self.draw_text(
                item_surface,
                line,
                (padding[0], padding[1] + font_height*i))
        # bilt on main surface
        self.screen.blit(item_surface, pos)
    
    def start_human_game(self):
        """Start game loop for human players."""
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.model.shoot(
                            self.target_player,
                            self.game_mouse_pos())
                    elif event.key == pygame.K_SPACE:
                        self.model.split(
                            self.target_player,
                            self.game_mouse_pos())

            # Expand camera view bounds as player's cell grows or splits
            self.expand_camera()

            self.model.update_velocity(self.target_player, self.game_mouse_pos())
            self.model.update()
            self.redraw()
            self.clock.tick(self.fps)

    def start_ai_game(self, food_count: int, virus_count: int, agents: list[agent.RNNAgent], cluster_settings: dict):
        """
        Starts the game loop for an AI game with RNN agents.
        :param food_count: Number of food cells to spawn
        :param virus_count: Number of viruses to spawn
        :param agents: List of RNN agents to use
        :param cluster_settings: Settings for cell clustering
        :return:
        """
        # Clustering for each cell type to reduce input size to our RNNs
        # The way the JSON settings are set up is stupid but it's what works best with object recognition
        cluster_by = {
            "virus": cluster_settings["virus"]["cluster_by"],
            "player": cluster_settings["unknown"]["variants"]["player"]["cluster_by"],
            "food": cluster_settings["unknown"]["variants"]["food"]["cluster_by"]
        }
        max_cluster_counts = {
            "virus": cluster_settings["virus"]["max_count"],
            "player": cluster_settings["unknown"]["variants"]["player"]["max_count"],
            "food": cluster_settings["unknown"]["variants"]["food"]["max_count"]
        }
        cluster_distances = {
            "virus": cluster_settings["virus"]["cluster_distance"],
            "player": cluster_settings["unknown"]["variants"]["player"]["cluster_distance"],
            "food": cluster_settings["unknown"]["variants"]["food"]["cluster_distance"]
        }

        agent_view_index = 0
        human_playing = self.target_player is not None

        while True:
            cells = self.model.cells

            virus_cells = []
            players = []
            player_cells = []
            food_cells = []

            # Lists of objects we'll use as input to the RNNs
            virus_objs = []
            player_objs = []
            food_objs = []

            for cell in cells:
                pos = Vector(cell.pos[0], cell.pos[1])
                top_left = Vector(pos.x - cell.radius, pos.y + cell.radius)
                bottom_right = Vector(pos.x + cell.radius, pos.y - cell.radius)
                bounds = (top_left, bottom_right)
                if isinstance(cell, Virus):
                    virus_cells.append(cell)
                    virus_obj = GameObject("virus", pos, cell.area(), cell.perimeter(), 0.9, 1, bounds)
                    ImageProcessing.cluster_or_add(virus_obj, virus_objs, "virus", cluster_by["virus"], max_cluster_counts["virus"], cluster_distances["virus"])
                elif isinstance(cell, Player):
                    players.append(cell)
                elif isinstance(cell, PlayerCell):
                    player_cells.append(cell)
                    player_obj = GameObject("player", pos, cell.area(), cell.perimeter(), 1.0, 1, bounds)
                    ImageProcessing.cluster_or_add(player_obj, virus_objs, "player", cluster_by["player"], max_cluster_counts["player"], cluster_distances["player"])
                else:
                    food_cells.append(cell)
                    food_obj = GameObject("food", pos, cell.area(), cell.perimeter(), 1.0, 1, bounds)
                    ImageProcessing.cluster_or_add(food_obj, virus_objs, "food", cluster_by["food"], max_cluster_counts["food"], cluster_distances["food"])

            # Maintain virus count
            if len(virus_cells) < virus_count:
                self.model.spawn_viruses(virus_count - len(virus_cells))

            # Maintain food count
            if len(food_cells) < food_count:
                self.model.spawn_cells(food_count - len(food_cells))

            # Handle user events
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if human_playing:
                        if event.key == pygame.K_w:
                            self.model.shoot(self.target_player, self.game_mouse_pos())
                        elif event.key == pygame.K_SPACE:
                            self.model.split(self.target_player, self.game_mouse_pos())
                    else:
                        # Spectating the AI agents
                        if event.key == pygame.K_LEFT:
                            # Spectate the previous player
                            agent_view_index -= 1
                            if agent_view_index < 0:
                                agent_view_index = len(self.model.players) - 1
                            self.target_player = self.model.players[agent_view_index]
                        elif event.key == pygame.K_RIGHT:
                            # Spectate the next player
                            agent_view_index += 1
                            if agent_view_index >= len(self.model.players):
                                agent_view_index = 0
                            self.target_player = self.model.players[agent_view_index]

            # Player mouse movement if human player exists
            if human_playing:
                self.model.update_velocity(self.target_player, self.game_mouse_pos())

            # Execute AI actions
            for i in range(len(agents)):
                # Generate inputs given the viewing bounds of the agent's player center
                center = players[i].center()
                center_pos = Vector(center[0], center[1])
                objects_in_view = []

                half_view_width = self.camera.width / 2
                half_view_height = self.camera.height / 2
                # Bounds is top left corner and bottom right corner
                view_bounds = (Vector(center_pos.x - half_view_width, center_pos.y + half_view_height),
                               Vector(center_pos.x + half_view_width, center_pos.y - half_view_height))

                for virus_obj in virus_objs:
                    if virus_obj.check_visible(view_bounds):
                        # Update object position to relative position
                        visible_obj = virus_obj.copy()
                        visible_obj.pos -= center_pos
                        objects_in_view.append(visible_obj)

                for player_obj in player_objs:
                    if player_obj.check_visible(view_bounds):
                        visible_obj = player_obj.copy()
                        visible_obj.pos -= center_pos
                        objects_in_view.append(visible_obj)

                for food_obj in food_objs:
                    if food_obj.check_visible(view_bounds):
                        visible_obj = food_obj.copy()
                        visible_obj.pos -= center_pos
                        objects_in_view.append(visible_obj)

                move_x, move_y, split, eject = agents[i].get_action(objects_in_view)

                # Normalize move vector
                move_length = math.sqrt(move_x**2 + move_y**2)
                if move_length > 1:
                    move_x /= move_length
                    move_y /= move_length

                target_pos = (move_x * agents[i].hyperparameters.move_sensitivity,
                              move_y * agents[i].hyperparameters.move_sensitivity)

                # Execute actions
                if split > 0.5:
                    players[i].split(target_pos)
                if eject > 0.5:
                    players[i].shoot(target_pos)
                self.model.update_velocity(players[i], target_pos)

            self.model.update()
            self.redraw()
            self.clock.tick(self.fps)

    def draw_debug_info(self):
        """Draw debug information on the screen."""
        # draw player center
        pygame.draw.circle(
            self.screen, 
            self.DEBUG_COLOR,
            self.camera.adjust(self.target_player.center()), 
            5)
        # draw velocity vectors of player parts
        for cell in self.target_player.parts:
            dx, dy = gu.polar_to_cartesian(cell.angle, cell.speed*100)
            x, y = cell.pos
            self.draw_vector(x, y, dx, dy, self.DEBUG_COLOR)

    def draw_vector(self, x, y, dx, dy, color):
        """Draw passed vector on the screen."""
        pygame.draw.line(
            self.screen,
            color,
            self.camera.adjust([x, y]),
            self.camera.adjust([x+dx, y+dy]))
        pygame.draw.circle(
            self.screen,
            color,
            self.camera.adjust([x+dx, y+dy]),
            3)

    def game_mouse_pos(self):
        """Get mouse position in the game bounds."""
        x, y = self.mouse_pos()
        # Due to camera starting from top left corner
        actual_camera_pos = (self.camera.x + self.width/2, self.camera.y - self.height/2)
        return x + actual_camera_pos[0], y + actual_camera_pos[1]

    def mouse_pos(self):
        """Get mouse position."""
        x, y = pygame.mouse.get_pos()
        # center offset
        x -= self.width / 2
        y = self.height / 2 - y
        return x, y

    def mouse_pos_to_polar(self):
        """Convert mouse position to polar vector."""
        x, y = pygame.mouse.get_pos()
        # center offset 
        x -= self.width/2
        y = self.height/2 - y
        # get angle and length(speed) of vector
        angle = math.atan2(y, x)
        speed = math.sqrt(x**2 + y**2)
        # setting radius of speed change zone
        speed_bound = 0.8*min(self.width/2, self.height/2)
        # normalize speed
        speed = 1 if speed >= speed_bound else speed/speed_bound
        return angle, speed

if __name__ == '__main__':
    bounds = [1000, 1000]
    cell_num = 100
    
    p = Player.make_random("Jetraid", bounds)
    p.parts[0].radius = 100
    players = [
        Player.make_random("Sobaka", bounds),
        Player.make_random("Kit", bounds),
        Player.make_random("elohssa", bounds),
        p,
    ]
    m = Model(players, bounds=bounds)
    m.spawn_cells(cell_num)

    pygame.init()
    screen = pygame.display.set_mode((900, 600))

    v = View(screen, m, p, debug=True)
    v.start_human_game()

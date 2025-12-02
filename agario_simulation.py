import argparse
import pygame
import torch
import json
import agent
import time
import math
from game.view import View
from game.model import Model
from game.entities import Player, Virus, PlayerCell
from geometry_utils import Vector, GameObject
from image_processing import ImageProcessing


class Simulation:
    def __init__(self, width: int, height: int, bounds: int, food_count: int, virus_count: int, agents: list[agent.RNNAgent]):
        self.width = width
        self.height = height
        self.bounds = bounds
        self.food_count = food_count
        self.virus_count = virus_count
        self.agents = agents

    def run_parallel(self):
        """
        Runs simulation, assumed to be in parallel with other simulations.
        Cannot draw or include human player in the simulation since it's running in parallel.
        :return:
        """
        bounds = [self.bounds, self.bounds]

        players = []

        for i in range(len(self.agents)):
            ai_player = Player.make_random(f"Agent {i}", bounds)
            players.append(ai_player)
        model = Model(players, bounds=bounds)
        model.spawn_cells(self.food_count)
        model.spawn_viruses(self.virus_count)

        with open("cluster_settings.json") as f:
            cluster_settings = json.load(f)

        start_time = time.time()
        while True:
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

            while True:
                cells = model.cells

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
                        ImageProcessing.cluster_or_add(virus_obj, virus_objs, "virus", cluster_by["virus"],
                                                       max_cluster_counts["virus"], cluster_distances["virus"])
                    elif isinstance(cell, Player):
                        players.append(cell)
                    elif isinstance(cell, PlayerCell):
                        player_cells.append(cell)
                        player_obj = GameObject("player", pos, cell.area(), cell.perimeter(), 1.0, 1, bounds)
                        ImageProcessing.cluster_or_add(player_obj, virus_objs, "player", cluster_by["player"],
                                                       max_cluster_counts["player"], cluster_distances["player"])
                    else:
                        food_cells.append(cell)
                        food_obj = GameObject("food", pos, cell.area(), cell.perimeter(), 1.0, 1, bounds)
                        ImageProcessing.cluster_or_add(food_obj, virus_objs, "food", cluster_by["food"],
                                                       max_cluster_counts["food"], cluster_distances["food"])

                # Maintain virus count
                if len(virus_cells) < self.virus_count:
                    model.spawn_viruses(self.virus_count - len(virus_cells))

                # Maintain food count
                if len(food_cells) < self.food_count:
                    model.spawn_cells(self.food_count - len(food_cells))

                # Execute AI actions
                for i in range(len(self.agents)):
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

                    move_x, move_y, split, eject = self.agents[i].get_action(objects_in_view)
                    move_angle = math.atan2(move_y, move_x)

                    # Normalize move vector
                    move_length = math.sqrt(move_x ** 2 + move_y ** 2)
                    if move_length > 1:
                        move_x /= move_length
                        move_y /= move_length

                    # Execute actions
                    model.update_velocity(players[i], move_angle, self.agents[i].hyperparameters.move_sensitivity)
                    target_pos = (move_x * self.agents[i].hyperparameters.move_sensitivity,
                                  move_y * self.agents[i].hyperparameters.move_sensitivity)
                    if split > 0.5:
                        players[i].split(target_pos)
                    if eject > 0.5:
                        players[i].shoot(target_pos)

                model.update()
                time.sleep(0.1)

    def run_single(self, draw_game: bool, human_playing: bool):
        """
        Runs the simulation, assumed to be the sole simulation.
        :param: draw_game: Whether this simulation should be drawn
        :param human_playing: Whether there will be a human player for this simulation
        :return:
        """
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Agar.io AI Offline')

        bounds = [self.bounds, self.bounds]

        players = []
        human_player = None
        if human_playing:
            human_player = Player.make_random("Human", bounds)
            players.append(human_player)

        for i in range(len(self.agents)):
            ai_player = Player.make_random(f"Agent {i}", bounds)
            players.append(ai_player)
        model = Model(players, bounds=bounds)
        model.spawn_cells(self.food_count)
        model.spawn_viruses(self.virus_count)

        # Start view loop (handles input: mouse to move, W to shoot, SPACE to split)
        view = View(screen, model, human_player, debug=False)

        with open("cluster_settings.json") as f:
            cluster_settings = json.load(f)
        view.start_ai_game(self.food_count, self.virus_count, self.agents, cluster_settings, draw_game)


def main():
    parser = argparse.ArgumentParser(description="Offline AI Agar.io Game")
    parser.add_argument('-wt', '--width', dest='width', type=int, default=900, help='screen width')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=600, help='screen height')
    parser.add_argument('-b', '--bounds', dest='bounds', type=int, default=1000, help='half-size of world bounds (world is [-b,b] x [-b,b])')
    parser.add_argument('-f', '--food', dest='food', type=int, default=500, help='initial food cell count')
    parser.add_argument('-v', '--viruses', dest='viruses', type=int, default=15, help='initial virus count')
    parser.add_argument('-a', '--agents', dest='agents', type=int, default=500, help='number of AI agents')
    parser.add_argument('-n', '--nick', dest='nick', type=str, default='Player', help='your nickname')
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption('Agar.io AI Offline')

    bounds = [args.bounds, args.bounds]

    players = []
    player = Player.make_random(args.nick, bounds)
    players.append(player)
    for i in range(args.agents):
        ai_player = Player.make_random(f"Agent {i}", bounds)
        players.append(ai_player)
    model = Model(players, bounds=bounds)
    model.spawn_cells(args.cells)
    model.spawn_viruses(args.viruses)

    # Start view loop (handles input: mouse to move, W to shoot, SPACE to split)
    view = View(screen, model, player, debug=False)
    ai_agent = agent.RNNAgent(None, None, False, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with open("cluster_settings.json") as f:
        cluster_settings = json.load(f)
    view.start_ai_game(args.food, args.viruses, [ai_agent] * args.agents, cluster_settings, True)


if __name__ == '__main__':
    main()

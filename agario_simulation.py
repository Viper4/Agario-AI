import argparse
import pygame
import torch
import json
import agent
import time
import math
from game.view import View
from game.new_model import Model
from game.entities import Player
from geometry_utils import Vector, GameObject
from image_processing import ImageProcessing


class AgarioSimulation:
    def __init__(self, base_view_width: int, base_view_height: int, bounds: int, food_count: int, virus_count: int, agents: list[agent.RNNAgent]):
        self.base_view_width = base_view_width
        self.base_view_height = base_view_height
        self.bounds = bounds
        self.food_count = food_count
        self.virus_count = virus_count
        self.agents = agents

    def run_headless(self, cluster_settings: dict, fps: int, simulation_speed: float, duration: float):
        """
        Runs simulation with only AI agents without drawing the game's visuals.
        Stops when the duration is reached or when only 1 agent is alive.
        :param cluster_settings: Settings for clustering objects in the game
        :param fps: Frames per second or number of model updates per second
        :param simulation_speed: Number of times faster the simulation runs
        :param duration: Duration of the simulation until termination in seconds
        :return: List of fitness values for each agent
        """
        spf = 1 / fps
        time_scale = 1 / simulation_speed
        scaled_duration = duration * time_scale

        bounds = [self.bounds, self.bounds]

        players = []
        prev_target_pos = []

        for i in range(len(self.agents)):
            ai_player = Player.make_random(fps, f"Agent {i}", bounds)
            players.append(ai_player)

            prev_target_pos.append(players[i].center())

        model = Model(players, bounds=bounds, fps=fps, sim_speed=simulation_speed)
        model.spawn_cells(self.food_count)
        model.spawn_viruses(self.virus_count)

        #pygame.init()
        #screen = pygame.display.set_mode((self.base_view_width, self.base_view_height))
        #pygame.display.set_caption('Agar.io AI Simulation')
        #view = View(screen, model, players[0], debug=False)

        # Clustering for each cell type to reduce input size to our RNNs
        # The way the JSON settings are set up is stupid but it's what works best with object recognition
        max_cluster_counts = {
            "virus": cluster_settings["virus"]["max_count"],
            "player": cluster_settings["unknown"]["variants"]["player"]["max_count"],
            "food": cluster_settings["unknown"]["variants"]["food"]["max_count"]
        }
        cluster_by = {
            "virus": cluster_settings["virus"]["cluster_by"],
            "player": cluster_settings["unknown"]["variants"]["player"]["cluster_by"],
            "food": cluster_settings["unknown"]["variants"]["food"]["cluster_by"]
        }
        cluster_distances = {
            "virus": cluster_settings["virus"]["cluster_distance"],
            "player": cluster_settings["unknown"]["variants"]["player"]["cluster_distance"],
            "food": cluster_settings["unknown"]["variants"]["food"]["cluster_distance"]
        }

        start_time = time.time()
        last_agent_run_time = -100000
        while True:
            if time.time() - start_time >= scaled_duration:
                break

            num_viruses = model.virus_count()
            num_cells = model.cell_count()
            # Maintain virus and food counts
            if num_viruses < self.virus_count:
                model.spawn_viruses(self.virus_count - num_viruses)
            if num_cells < self.food_count:
                model.spawn_cells(self.food_count - num_cells)

            run_agent = time.time() - last_agent_run_time >= self.agents[0].run_interval * time_scale

            if run_agent:
                # Lists of objects we'll use as input to the RNNs
                virus_objs = []
                player_objs = []
                food_objs = []

                for cell in model.cells:
                    if cell is None:
                        continue
                    pos = Vector(cell.pos[0], cell.pos[1])
                    top_left = Vector(pos.x - cell.radius, pos.y + cell.radius)
                    bottom_right = Vector(pos.x + cell.radius, pos.y - cell.radius)
                    bounds = (top_left, bottom_right)
                    # Cluster food cells for input to RNN
                    food_obj = GameObject("food", pos, cell.area(), cell.perimeter(), 1.0, 1, bounds)
                    ImageProcessing.cluster_or_add(food_obj, food_objs, "food", max_cluster_counts["food"], cluster_by["food"], cluster_distances["food"])

                for virus in model.viruses:
                    if virus is None:
                        continue
                    pos = Vector(virus.pos[0], virus.pos[1])
                    top_left = Vector(pos.x - virus.radius, pos.y + virus.radius)
                    bottom_right = Vector(pos.x + virus.radius, pos.y - virus.radius)
                    bounds = (top_left, bottom_right)
                    # Cluster viruses to be input to RNN
                    virus_obj = GameObject("virus", pos, virus.area(), virus.perimeter(), 0.9, 1, bounds)
                    ImageProcessing.cluster_or_add(virus_obj, virus_objs, "virus", max_cluster_counts["virus"], cluster_by["virus"], cluster_distances["virus"])

                for player in players:
                    if not player.alive:
                        continue
                    for part in player.parts:
                        pos = Vector(part.pos[0], part.pos[1])
                        top_left = Vector(pos.x - part.radius, pos.y + part.radius)
                        bottom_right = Vector(pos.x + part.radius, pos.y - part.radius)
                        bounds = (top_left, bottom_right)
                        # Cluster players to be input to RNN
                        player_obj = GameObject("player", pos, part.area(), part.perimeter(), 1.0, 1, bounds)
                        ImageProcessing.cluster_or_add(player_obj, player_objs, "player", max_cluster_counts["player"], cluster_by["player"], cluster_distances["player"])

                # Execute AI actions
                agents_alive = 0
                for i in range(len(self.agents)):
                    if not players[i].alive:
                        continue
                    agents_alive += 1
                    # Generate inputs given the viewing bounds of the agent's player center
                    center = players[i].center()
                    score = players[i].score()  # View bounds is expanded based on score
                    center_pos = Vector(center[0], center[1])
                    objects_in_view = []

                    half_view_width = (self.base_view_width + score) / 2
                    half_view_height = (self.base_view_height + score) / 2
                    # Bounds is top left corner and bottom right corner
                    view_bounds = (Vector(center_pos.x - half_view_width, center_pos.y + half_view_height),
                                   Vector(center_pos.x + half_view_width, center_pos.y - half_view_height))

                    # Update object position to relative position
                    # Also normalize all inputs to be passed into the RNN
                    for virus_obj in virus_objs:
                        if virus_obj.check_visible(view_bounds):
                            visible_obj = virus_obj.copy()

                            # Normalize position to relative pos from [-1, 1] within view bounds
                            visible_obj.pos -= center_pos  # relative position
                            visible_obj.pos.x /= half_view_width
                            visible_obj.pos.y /= half_view_height

                            # Normalize area and perimeter
                            visible_obj.area /= 100000  # Good enough max area estimate
                            visible_obj.perimeter /= 1000  # Good enough max perimeter estimate

                            # Normalize count
                            visible_obj.count /= max_cluster_counts["virus"]

                            objects_in_view.append(visible_obj)

                    for player_obj in player_objs:
                        if player_obj.check_visible(view_bounds):
                            visible_obj = player_obj.copy()

                            # Normalize position to relative pos from [-1, 1] within view bounds
                            visible_obj.pos -= center_pos  # relative position
                            visible_obj.pos.x /= half_view_width
                            visible_obj.pos.y /= half_view_height

                            # Normalize area and perimeter
                            visible_obj.area /= 100000  # Good enough max area estimate
                            visible_obj.perimeter /= 1000  # Good enough max perimeter estimate

                            # Normalize count
                            visible_obj.count /= max_cluster_counts["player"]

                            objects_in_view.append(visible_obj)

                    for food_obj in food_objs:
                        if food_obj.check_visible(view_bounds):
                            visible_obj = food_obj.copy()

                            # Normalize position to relative pos from [-1, 1] within view bounds
                            visible_obj.pos -= center_pos  # relative position
                            visible_obj.pos.x /= half_view_width
                            visible_obj.pos.y /= half_view_height

                            # Normalize area and perimeter
                            visible_obj.area /= 10000  # Good enough max area estimate
                            visible_obj.perimeter /= 1000  # Good enough max perimeter estimate

                            # Normalize count
                            visible_obj.count /= max_cluster_counts["food"]

                            objects_in_view.append(visible_obj)

                    move_x, move_y, split, eject = self.agents[i].get_action(objects_in_view)
                    agent_pos = players[i].center()

                    # Normalize move vector
                    move_length = math.sqrt(move_x ** 2 + move_y ** 2)
                    if move_length > 1:
                        move_x /= move_length
                        move_y /= move_length

                    # Execute actions
                    target_pos = (agent_pos[0] + move_x * self.agents[i].hyperparameters.move_sensitivity,
                                  agent_pos[1] + move_y * self.agents[i].hyperparameters.move_sensitivity)

                    if split > 0:
                        players[i].split(target_pos)
                    if eject > 0:
                        players[i].shoot(target_pos)
                    model.update_velocity(players[i], target_pos)
                    prev_target_pos[i] = target_pos
                    players[i].score()

                    #if not view.target_player.alive:
                    #    view.target_player = players[i]

                last_agent_run_time = time.time()

                if agents_alive <= 1:
                    break

                #view.redraw()

            model.update()
            time.sleep(spf)

        fitnesses = []
        for i in range(len(self.agents)):
            fitnesses.append(self.agents[i].calculate_fitness(players[i].num_food_eaten,
                                                              players[i].time_alive,
                                                              players[i].num_players_eaten,
                                                              players[i].highest_score))
        print(f"Simulation complete after {time.time() - start_time:.2f} seconds")
        return fitnesses


def main():
    parser = argparse.ArgumentParser(description="Offline AI Agar.io Game")
    parser.add_argument('-wt', '--width', dest='width', type=int, default=900, help='screen width')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=600, help='screen height')
    parser.add_argument('-b', '--bounds', dest='bounds', type=int, default=1500, help='half-size of world bounds (world is [-b,b] x [-b,b])')
    parser.add_argument('-f', '--food', dest='food', type=int, default=750, help='initial food cell count')
    parser.add_argument('-v', '--viruses', dest='viruses', type=int, default=20, help='initial virus count')
    parser.add_argument('-a', '--agents', dest='agents', type=int, default=500, help='number of AI agents')
    parser.add_argument('-n', '--nick', dest='nick', type=str, default='Player', help='your nickname')
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption('Agar.io AI Offline')

    bounds = [args.bounds, args.bounds]

    players = []
    for i in range(args.agents):
        ai_player = Player.make_random(60, f"Agent {i}", bounds)
        players.append(ai_player)

    player = Player.make_random(60, args.nick, bounds)
    players.append(player)

    model = Model(players, bounds=bounds)
    model.spawn_cells(args.food)
    model.spawn_viruses(args.viruses)

    # Start view loop (handles input: mouse to move, W to shoot, SPACE to split)
    view = View(screen, model, player, debug=False)
    ai_agent = agent.RNNAgent(None, None, False, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with open("cluster_settings.json") as f:
        cluster_settings = json.load(f)


if __name__ == '__main__':
    main()

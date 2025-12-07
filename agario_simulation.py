import time
import torch
import argparse
import pygame
import pickle
from tqdm import tqdm
from game.opencv_view import OCView, OCCamera
from game.view import View
from game.model import Model
from game.entities import Player
from geometry_utils import Vector
from agent import RNNAgent, ModelBasedReflexAgent, Hyperparameters, FitnessWeights


class AgarioSimulation:
    def __init__(self, view_width: int, view_height: int, bounds: int, food_count: int, virus_count: int):
        self.view_width = view_width
        self.view_height = view_height
        self.bounds = bounds
        self.food_count = food_count
        self.virus_count = virus_count

    @staticmethod
    def rnn_tick(view_width: int, view_height: int, agent: RNNAgent, player: Player, model: Model):
        """
        Runs a single tick of the simulation for the given RNN agent.
        :param view_width: Width of the view
        :param view_height: Height of the view
        :param agent: RNNAgent to make actions with during simulation
        :param player: Player to make actions for
        :param model: Model to use for simulation
        :return: The agent's (target_pos, split, eject)
        """
        player.ticks_alive += 1

        # Expand view bounds based on player's score
        center_pos = player.center()
        score = player.score()
        inv_scale = OCCamera.get_inverse_scale(score)

        scaled_width = view_width * inv_scale
        scaled_height = view_height * inv_scale
        half_view_width = scaled_width * 0.5
        half_view_height = scaled_height * 0.5
        # Bounds is top left corner and bottom right corner
        view_bounds = ((center_pos[0] - half_view_width, center_pos[1] + half_view_height),
                       (center_pos[0] + half_view_width, center_pos[1] - half_view_height))

        vision_grid = agent.init_grid()
        max_food = 0
        max_virus = 0
        max_player = 0
        max_area = 0.0

        for chunk in model.get_chunks(view_bounds):
            for cell in chunk.cells:
                if cell.within_bounds(view_bounds):
                    # Normalize global position to relative position in the view bounds from [-1, 1]
                    normalized_pos = Vector((cell.pos[0] - center_pos[0]) / half_view_width,
                                            (cell.pos[1] - center_pos[1]) / half_view_height)

                    gx, gy = agent.get_grid_index(normalized_pos)
                    grid_cell = vision_grid[gx, gy]
                    grid_cell[0] += 1  # Food count
                    grid_cell[3] += cell.area()

                    if grid_cell[0] > max_food:
                        max_food = grid_cell[0]

                    if grid_cell[3] > max_area:
                        max_area = grid_cell[3]
            for virus in chunk.viruses:
                if virus.within_bounds(view_bounds):
                    # Normalize global position to relative position in the view bounds from [-1, 1]
                    normalized_pos = Vector((virus.pos[0] - center_pos[0]) / half_view_width,
                                            (virus.pos[1] - center_pos[1]) / half_view_height)

                    gx, gy = agent.get_grid_index(normalized_pos)
                    grid_cell = vision_grid[gx, gy]
                    grid_cell[1] += 1  # Virus count
                    grid_cell[3] += virus.area()

                    if grid_cell[1] > max_virus:
                        max_virus = grid_cell[1]

                    if grid_cell[3] > max_area:
                        max_area = grid_cell[3]
            for playercell in chunk.playercells:
                if playercell.within_bounds(view_bounds):
                    # Normalize global position to relative position in the view bounds from [-1, 1]
                    normalized_pos = Vector((playercell.pos[0] - center_pos[0]) / half_view_width,
                                            (playercell.pos[1] - center_pos[1]) / half_view_height)

                    gx, gy = agent.get_grid_index(normalized_pos)
                    grid_cell = vision_grid[gx, gy]
                    grid_cell[2] += 1  # PlayerCell count
                    grid_cell[3] += playercell.area()

                    if grid_cell[2] > max_player:
                        max_player = grid_cell[2]

                    if grid_cell[3] > max_area:
                        max_area = grid_cell[3]

        # Normalize nodes in grid using vectorized division
        vision_grid /= torch.tensor(
            [max_food, max_virus, max_player, max_area],
            device=vision_grid.device,
            dtype=vision_grid.dtype
        )
        vision_grid.nan_to_num_(nan=0.0)  # NaN values come from divisions by 0 (when max values are 0)

        move_x, move_y, split, eject = agent.get_action(vision_grid)

        target_pos = (center_pos[0] + move_x * agent.hyperparameters.move_sensitivity,
                      center_pos[1] + move_y * agent.hyperparameters.move_sensitivity)

        return target_pos, split, eject

    def run_rnn(self, agents: list[RNNAgent], base_fps: int, simulation_duration: float, headless: bool):
        """
        Run a simulation with only RNN agents.
        Stops when the duration is reached or when only 1 agent is alive.
        :param agents: List of RNN agents
        :param base_fps: Frames per second for a game at normal speed.
        :param simulation_duration: How long the simulation runs in seconds. Doesn't mean termination will occur at this time.
        :param headless: Whether to draw the game's visuals.
        :return: List of fitness values for each agent
        """
        num_frames = int(simulation_duration * base_fps)  # Number of iterations to run the simulation for

        bounds = [self.bounds, self.bounds]

        players = []
        prev_target_pos = []

        for i in range(len(agents)):
            ai_player = Player.make_random(f"Agent {i}", bounds)
            players.append(ai_player)

            prev_target_pos.append(players[i].center())

        model = Model(players, bounds=bounds, chunk_size=self.bounds // 10)  # 21 chunks per side
        model.spawn_cells(self.food_count)
        model.spawn_viruses(self.virus_count)
        print(f"Simulation model initialized with {model.chunk_count_x}x{model.chunk_count_y}={model.chunk_count_x*model.chunk_count_y} chunks.")

        view = OCView(self.view_width, self.view_height, model, players[0], debug=False)

        start_time = time.time()
        last_agent_run_frame = -100
        for frame in tqdm(range(num_frames), desc="Running simulation", unit="frames"):
            # Maintain virus and food counts
            if model.num_viruses < self.virus_count:
                model.spawn_viruses(self.virus_count - model.num_viruses)
            if model.num_cells < self.food_count:
                model.spawn_cells(self.food_count - model.num_cells)

            run_agent = frame - last_agent_run_frame >= agents[0].run_interval * base_fps

            if not run_agent:
                # Maintain agent's previous target position
                for i in range(len(players)):
                    if players[i].alive:
                        players[i].ticks_alive += 1
                        model.update_velocity(players[i], prev_target_pos[i])
            else:
                # Run agent's actions
                agents_alive = 0
                for i in range(len(agents)):
                    if not players[i].alive:
                        continue

                    agents_alive += 1

                    target_pos, split, eject = self.rnn_tick(self.view_width, self.view_height, agents[i], players[i], model)

                    # Execute actions
                    if split > 0:
                        model.split(players[i], target_pos)
                    if eject > 0:
                        model.shoot(players[i], target_pos)
                    model.update_velocity(players[i], target_pos)
                    prev_target_pos[i] = target_pos

                last_agent_run_frame = frame

                if agents_alive <= 1:
                    break

            model.update()
            if not headless:
                target_score = view.player.score()
                # Expand view while maintaining aspect ratio
                inv_scale = OCCamera.get_inverse_scale(target_score)
                view.camera.scale = 1.0 / inv_scale
                view.redraw(spectate_mode=True)

        fitnesses = []
        agents_alive = 0
        for i in range(len(agents)):
            if players[i].alive:
                agents_alive += 1
            fitnesses.append(agents[i].calculate_fitness(players[i].num_food_eaten,
                                                         players[i].ticks_alive / num_frames,
                                                         players[i].num_players_eaten,
                                                         players[i].score(),
                                                         int(not players[i].alive)))
        print(f"Simulation complete after {time.time() - start_time:.2f} seconds with {agents_alive} agents alive")
        return fitnesses
    
    def run_mbra(self, agent: ModelBasedReflexAgent, base_fps: int, simulation_duration: float, headless: bool):
        """
        Run a simulation with only ModelBasedReflexAgent.
        Stops when the duration is reached or when only 1 agent is alive.
        :param agent: Single ModelBasedReflexAgent to make actions with during simulation
        :param base_fps: Frames per second for a game at normal speed.
        :param simulation_duration: How long the simulation runs in seconds. Doesn't mean termination will occur at this time.
        :param headless: Whether to draw the game's visuals.
        :return: List of fitness values for each agent
        :return:
        """
        pass


def main():
    parser = argparse.ArgumentParser(description="Offline AI Agar.io Game")
    parser.add_argument('-wt', '--width', dest='width', type=int, default=900, help='screen width')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=600, help='screen height')
    parser.add_argument('-b', '--bounds', dest='bounds', type=int, default=1500, help='half-size of world bounds (world is [-b,b] x [-b,b])')
    parser.add_argument('-f', '--food', dest='food', type=int, default=700, help='initial food cell count')
    parser.add_argument('-v', '--viruses', dest='viruses', type=int, default=20, help='initial virus count')
    parser.add_argument('-a', '--agents', dest='agents', type=int, default=5, help='number of AI agents')
    parser.add_argument('-n', '--nick', dest='nick', type=str, default='Player', help='your nickname')
    parser.add_argument('-oc', '--opencv', dest='opencv', type=bool, default=True, help='Whether to use OpenCV view')
    args = parser.parse_args()

    bounds = [args.bounds, args.bounds]

    # Load best RNNAgent from agent_snapshots.pkl
    hyperparameters = Hyperparameters(hidden_layers=[64, 16],
                                      output_size=4,
                                      run_interval=0.1,
                                      param_mutations={"weight": 2.0, "bias": 0.5},
                                      move_sensitivity=50.0,
                                      grid_width=9,
                                      grid_height=6,
                                      nodes_per_cell=4)
    fitness_weights = FitnessWeights(food=0.5, time_alive=100.0, cells_eaten=50.0, score=0.75, death=500.0)
    base_agent = RNNAgent(hyperparameters=hyperparameters.copy(), fitness_weights=fitness_weights,
                          randomize_params=False, device=torch.device("cpu"))
    with open("agent_snapshots.pkl", "rb") as f:
        agent_snapshots = pickle.load(f)
        # Snapshots are saved in order of fitness
        base_agent.rnn.load_state_dict(agent_snapshots[0])

    # Create model and players
    human_player = Player.make_random(args.nick, bounds)
    players = []
    agents = []
    for i in range(args.agents):
        ai_player = Player.make_random(f"Agent {i}", bounds)
        players.append(ai_player)
        agents.append(base_agent.copy())
    players.append(human_player)

    model = Model(players, bounds=bounds, chunk_size=args.bounds // 10)
    model.spawn_cells(args.food)
    model.spawn_viruses(args.viruses)

    if args.opencv:
        view = OCView(args.width, args.height, model, human_player)

        while True:
            # Maintain virus count
            if model.num_viruses < args.viruses:
                model.spawn_viruses(args.viruses - model.num_viruses)

            # Maintain food count
            if model.num_cells < args.food:
                model.spawn_cells(args.food - model.num_cells)

            # Update AI agents
            for i in range(len(players) - 1):  # Don't include human player
                if not players[i].alive:
                    continue
                target_pos, split, eject = AgarioSimulation.rnn_tick(args.width, args.height, agents[i], players[i], model)
                if split > 0:
                    model.split(players[i], target_pos)
                if eject > 0:
                    model.shoot(players[i], target_pos)
                model.update_velocity(players[i], target_pos)

            view.redraw(spectate_mode=False)
            model.update()
            time.sleep(0.008333)  # 1/60 = 60 FPS
    else:
        # Start view loop (handles input: mouse to move, W to shoot, SPACE to split)
        pygame.init()
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption('Agar.io Offline')
        view = View(screen, model, players[0], debug=False)
        view.start_human_game(args.food, args.viruses)


if __name__ == '__main__':
    main()

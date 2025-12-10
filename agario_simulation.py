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
from agent import RNNAgent, LSTMAgent, GRUAgent, SimpleReflexAgent, ModelBasedReflexAgent, Hyperparameters, FitnessWeights


class AgarioSimulation:
    def __init__(self, view_width: int, view_height: int, bounds: int, food_count: int, virus_count: int):
        self.view_width = view_width
        self.view_height = view_height
        self.bounds = bounds
        self.food_count = food_count
        self.virus_count = virus_count

    @staticmethod
    def sra_tick(view_width: int, view_height: int, agent: SimpleReflexAgent, player: Player, model: Model):
        """
        Runs a single tick of the simulation for the given SRA.
        :param view_width: Width of the view
        :param view_height: Height of the view
        :param agent: ModelBasedReflexAgent to make actions with during simulation
        :param player: Player to make actions for
        :param model: Model to use for simulation
        :return: The agent's (target_pos, split, eject)
        """
        center_pos = player.center()
        min_area = 10000000
        max_area = 0.0
        max_radius = 0.0
        for part in player.parts:
            area = part.area()
            min_area = min(min_area, area)
            max_area = max(max_area, area)
            max_radius = max(max_radius, part.radius)
        score = player.score()
        inv_scale = OCCamera.get_inverse_scale(score)

        scaled_width = view_width * inv_scale
        scaled_height = view_height * inv_scale
        half_view_width = scaled_width * 0.5
        half_view_height = scaled_height * 0.5
        view_bounds = ((center_pos[0] - half_view_width, center_pos[1] + half_view_height),
                       (center_pos[0] + half_view_width, center_pos[1] - half_view_height))

        threats = []
        prey = []
        foods = []
        viruses = []
        # Prepare objects in view to be passed to SRA
        for chunk in model.get_chunks(view_bounds):
            for cell in chunk.cells:
                if cell.within_bounds(view_bounds):
                    foods.append((cell.pos[0], cell.pos[1]))
            for virus in chunk.viruses:
                if virus.within_bounds(view_bounds):
                    viruses.append((virus.pos[0], virus.pos[1]))
            for playercell in chunk.playercells:
                if playercell.parent is player:
                    continue  # Skip own cells
                if playercell.within_bounds(view_bounds):
                    player_area = playercell.area()
                    if player_area > max_area * agent.THREAT_SIZE_RATIO:
                        threats.append((playercell.pos[0], playercell.pos[1]))
                    elif player_area < max_area * agent.PREY_SIZE_RATIO:
                        prey.append((playercell.pos[0], playercell.pos[1], player_area))

        return agent.get_action(threats, prey, foods, viruses, center_pos, min_area, max_area, max_radius)

    @staticmethod
    def mbra_tick(view_width: int, view_height: int, agent: ModelBasedReflexAgent, player: Player, model: Model, simulation_tick: int):
        """
        Runs a single tick of the simulation for the given MBRA agent.
        :param view_width: Width of the view
        :param view_height: Height of the view
        :param agent: ModelBasedReflexAgent to make actions with during simulation
        :param player: Player to make actions for
        :param model: Model to use for simulation
        :param simulation_tick: Current simulation tick for memory buffer decay
        :return: The agent's (target_pos, split, eject)
        """
        center_pos = player.center()
        min_area = 10000000
        max_area = 0.0
        max_radius = 0.0
        for part in player.parts:
            area = part.area()
            min_area = min(min_area, area)
            max_area = max(max_area, area)
            max_radius = max(max_radius, part.radius)
        score = player.score()
        inv_scale = OCCamera.get_inverse_scale(score)

        scaled_width = view_width * inv_scale
        scaled_height = view_height * inv_scale
        half_view_width = scaled_width * 0.5
        half_view_height = scaled_height * 0.5
        view_bounds = ((center_pos[0] - half_view_width, center_pos[1] + half_view_height),
                       (center_pos[0] + half_view_width, center_pos[1] - half_view_height))

        threats = []
        prey = []
        foods = []
        viruses = []
        # Prepare objects in view to be passed to SRA
        for chunk in model.get_chunks(view_bounds):
            for cell in chunk.cells:
                if cell.within_bounds(view_bounds):
                    foods.append((cell.pos[0], cell.pos[1]))
            for virus in chunk.viruses:
                if virus.within_bounds(view_bounds):
                    viruses.append((virus.pos[0], virus.pos[1]))
            for playercell in chunk.playercells:
                if playercell.parent is player:
                    continue  # Skip own cells
                if playercell.within_bounds(view_bounds):
                    player_area = playercell.area()
                    if player_area > max_area * agent.THREAT_SIZE_RATIO:
                        threats.append((playercell.pos[0], playercell.pos[1]))
                    elif player_area < max_area * agent.PREY_SIZE_RATIO:
                        prey.append((playercell.pos[0], playercell.pos[1], player_area))

        return agent.get_action(threats, prey, foods, viruses, center_pos, min_area, max_area, max_radius, simulation_tick)

    @staticmethod
    def generate_vision_grid_v1(agent: RNNAgent | LSTMAgent | GRUAgent, center_pos: tuple[float, float], model: Model, view_bounds: tuple[tuple[float, float], tuple[float, float]]):
        """
        Generates a vision grid where each grid cell is (food_count, virus_count, playercell_count, area).
        Each node value is normalized by the maximum value of that type in the entire grid.
        :param agent: RNNAgent to generate the vision grid for
        :param center_pos: Center position of the view
        :param model: Model to use for simulation
        :param view_bounds: Bounds of the view
        :return: torch.Tensor of shape (grid_width, grid_height, nodes_per_cell)
        """
        vision_grid = agent.init_grid()
        max_food = 0
        max_virus = 0
        max_player = 0
        max_area = 0.0

        # view_bounds = (top left, bottom right)
        view_width = view_bounds[1][0] - view_bounds[0][0]
        view_height = view_bounds[0][1] - view_bounds[1][1]
        half_view_width = view_width * 0.5
        half_view_height = view_height * 0.5

        for chunk in model.get_chunks(view_bounds):
            for cell in chunk.cells:
                if cell.within_bounds(view_bounds):
                    # Normalize global position to relative position in the view bounds from [-1, 1]
                    normalized_pos = ((cell.pos[0] - center_pos[0]) / half_view_width,
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
                    normalized_pos = ((virus.pos[0] - center_pos[0]) / half_view_width,
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
                    normalized_pos = ((playercell.pos[0] - center_pos[0]) / half_view_width,
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
        return vision_grid

    @staticmethod
    def generate_vision_grid_v2(agent: RNNAgent | LSTMAgent | GRUAgent, center_pos: tuple[float, float], model: Model,
                                view_bounds: tuple[tuple[float, float], tuple[float, float]]):
        """
        Generates a vision grid where each grid cell is (food_perimeter, virus_perimeter, player_perimeter).
        Each node value is normalized by the maximum value of that type in the entire grid.
        :param agent: RNNAgent to generate the vision grid for
        :param center_pos: Center position of the view
        :param model: Model to use for simulation
        :param view_bounds: Bounds of the view
        :return: torch.Tensor of shape (grid_width, grid_height, nodes_per_cell)
        """
        vision_grid = agent.init_grid()
        max_food_perimeter = 0.0
        max_virus_perimeter = 0.0
        max_player_perimeter = 0.0

        # view_bounds = (top left, bottom right)
        view_width = view_bounds[1][0] - view_bounds[0][0]
        view_height = view_bounds[0][1] - view_bounds[1][1]
        half_view_width = view_width * 0.5
        half_view_height = view_height * 0.5

        for chunk in model.get_chunks(view_bounds):
            for cell in chunk.cells:
                if cell.within_bounds(view_bounds):
                    # Normalize global position to relative position in the view bounds from [-1, 1]
                    normalized_pos = ((cell.pos[0] - center_pos[0]) / half_view_width,
                                      (cell.pos[1] - center_pos[1]) / half_view_height)

                    gx, gy = agent.get_grid_index(normalized_pos)
                    grid_cell = vision_grid[gx, gy]
                    grid_cell[0] += cell.perimeter()

                    if grid_cell[0] > max_food_perimeter:
                        max_food_perimeter = grid_cell[0]
            for virus in chunk.viruses:
                if virus.within_bounds(view_bounds):
                    # Normalize global position to relative position in the view bounds from [-1, 1]
                    normalized_pos = ((virus.pos[0] - center_pos[0]) / half_view_width,
                                      (virus.pos[1] - center_pos[1]) / half_view_height)

                    gx, gy = agent.get_grid_index(normalized_pos)
                    grid_cell = vision_grid[gx, gy]
                    grid_cell[1] += virus.perimeter()

                    if grid_cell[1] > max_virus_perimeter:
                        max_virus_perimeter = grid_cell[1]
            for playercell in chunk.playercells:
                if playercell.within_bounds(view_bounds):
                    # Normalize global position to relative position in the view bounds from [-1, 1]
                    normalized_pos = ((playercell.pos[0] - center_pos[0]) / half_view_width,
                                      (playercell.pos[1] - center_pos[1]) / half_view_height)

                    gx, gy = agent.get_grid_index(normalized_pos)
                    grid_cell = vision_grid[gx, gy]
                    grid_cell[2] += playercell.perimeter()

                    if grid_cell[2] > max_player_perimeter:
                        max_player_perimeter = grid_cell[2]

        # Normalize nodes in grid using vectorized division
        #print(max_food_perimeter, max_virus_perimeter, max_player_perimeter)
        vision_grid /= torch.tensor(
            [100.0, 330.0, 8000.0],  # Absolute max values determined manually
            device=vision_grid.device,
            dtype=vision_grid.dtype
        )
        vision_grid.nan_to_num_(nan=0.0)  # NaN values come from divisions by 0 (when max values are 0)
        return vision_grid

    @staticmethod
    def rnn_tick(view_width: int, view_height: int, agent: RNNAgent | LSTMAgent | GRUAgent, player: Player, model: Model):
        """
        Runs a single tick of the simulation for the given RNN agent.
        :param view_width: Width of the view
        :param view_height: Height of the view
        :param agent: agent to make actions with during simulation
        :param player: Player to make actions for
        :param model: Model to use for simulation
        :return: The agent's (target_pos, split, eject)
        """
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

        vision_grid = AgarioSimulation.generate_vision_grid_v2(agent, center_pos, model, view_bounds)

        move_x, move_y, split, eject = agent.get_action(vision_grid)

        target_pos = (center_pos[0] + move_x * agent.hyperparameters.move_sensitivity,
                      center_pos[1] + move_y * agent.hyperparameters.move_sensitivity)

        return target_pos, split, eject

    def run_training(self, agents: list[RNNAgent | LSTMAgent | GRUAgent], num_sra: int, base_fps: int, simulation_duration: float, headless: bool):
        """
        Run a simulation meant for training with the RNN agents and a number of SRAs.
        Stops when the duration is reached or when only 1 agent is alive.
        :param agents: List of RNN agents to spawn
        :param num_sra: Number of SRAs to spawn
        :param base_fps: Frames per second for a game at normal speed.
        :param simulation_duration: How long the simulation runs in seconds. Doesn't mean termination will occur at this time.
        :param headless: Whether to draw the game's visuals.
        :return: List of fitness values for each RNN agent
        """
        num_frames = int(simulation_duration * base_fps)

        bounds = [self.bounds, self.bounds]

        rnn_players = []
        prev_target_pos = []

        for i in range(len(agents)):
            ai_player = Player.make_random(f"Agent {i}", bounds, color=(0, 255, 0))
            rnn_players.append(ai_player)
            prev_target_pos.append(rnn_players[i].center())

        sras = []
        sra_players = []
        for i in range(num_sra):
            # sra color is red
            ai_player = Player.make_random(f"sra {i}", bounds, color=(255, 0, 0))
            sra_players.append(ai_player)
            prev_target_pos.append(sra_players[i].center())
            sras.append(SimpleReflexAgent(agents[0].run_interval, agents[0].fitness_weights, 50.0))

        model = Model(rnn_players + sra_players, bounds=bounds, chunk_size=self.bounds // 10)  # 21 chunks per side
        model.spawn_cells(self.food_count)
        model.spawn_viruses(self.virus_count)
        print(f"Simulation model initialized with {model.chunk_count_x}x{model.chunk_count_y}={model.chunk_count_x*model.chunk_count_y} chunks.")

        view = OCView(self.view_width, self.view_height, model, rnn_players[0], debug=False)

        start_time = time.time()
        last_agent_run_frame = -100
        for frame in tqdm(range(num_frames), desc="Running simulation", unit="frames"):
            if model.num_viruses < self.virus_count:
                model.spawn_viruses(self.virus_count - model.num_viruses)
            if model.num_cells < self.food_count:
                model.spawn_cells(self.food_count - model.num_cells)

            run_agent = frame - last_agent_run_frame >= agents[0].run_interval * base_fps

            if not run_agent:
                for i in range(len(rnn_players)):
                    if rnn_players[i].alive:
                        rnn_players[i].ticks_alive += 1
                        model.update_velocity(rnn_players[i], prev_target_pos[i])
            else:
                # Run sra actions
                for i in range(len(sras)):
                    if not sra_players[i].alive:
                        continue

                    target_pos, split, eject = AgarioSimulation.sra_tick(self.view_width, self.view_height, sras[i], sra_players[i], model)

                    if split > 0:
                        model.split(sra_players[i], target_pos)
                    if eject > 0:
                        model.shoot(sra_players[i], target_pos)
                    model.update_velocity(sra_players[i], target_pos)
                    prev_target_pos[i] = target_pos

                rnn_agents_alive = 0
                for i in range(len(agents)):
                    if not rnn_players[i].alive:
                        continue
                    rnn_players[i].ticks_alive += 1

                    rnn_agents_alive += 1

                    target_pos, split, eject = self.rnn_tick(self.view_width, self.view_height, agents[i], rnn_players[i], model)

                    if split > 0:
                        model.split(rnn_players[i], target_pos)
                    if eject > 0:
                        model.shoot(rnn_players[i], target_pos)
                    model.update_velocity(rnn_players[i], target_pos)
                    prev_target_pos[i] = target_pos

                last_agent_run_frame = frame

                if rnn_agents_alive <= 0:
                    break

            model.update()
            if not headless:
                target_score = view.player.score()
                inv_scale = OCCamera.get_inverse_scale(target_score)
                view.camera.scale = 1.0 / inv_scale
                view.redraw(spectate_mode=True)

        fitnesses = []
        rnn_agents_alive = 0
        sras_alive = 0
        for i in range(len(agents)):
            if rnn_players[i].alive:
                rnn_agents_alive += 1
            fitnesses.append(agents[i].calculate_fitness(rnn_players[i].num_food_eaten,
                                                         rnn_players[i].ticks_alive / num_frames,
                                                         rnn_players[i].num_players_eaten,
                                                         rnn_players[i].score() + rnn_players[i].highest_score,
                                                         int(not rnn_players[i].alive)))
        for i in range(len(sras)):
            if sra_players[i].alive:
                sras_alive += 1

        print(f"Simulation complete after {time.time() - start_time:.2f} seconds with {rnn_agents_alive} RNN agents alive and {sras_alive} SRAs alive")
        return fitnesses
    
    def run_evaluation(self, agents: list[RNNAgent | LSTMAgent | GRUAgent], num_sra: int, num_mbra: int, base_fps: int, simulation_duration: float, headless: bool):
        pass


# Function needs to be picklable so keep it out of classes
def run_simulation_worker(fps: int, simulation_duration: float, state_dicts: list[dict], agent_classes: list[str],
                          pickled_data: bytes, headless: bool, generation: int):
    """
    Function intended for parallel processing which reconstructs agents and runs a simulation with them.
    Returns final fitness of all the agents in this simulation.
    :param fps: "Normal" frames per second for simulation. Basically, defines how fast a normal game runs for humans.
    :param simulation_duration: Duration of simulation in seconds
    :param state_dicts: List of parameters
    :param agent_classes: List of agent classes in string format
    :param pickled_data: hyperparameters and fitness weights
    :param headless: Whether to visualize the simulation or not
    :param generation:
    :return:
    """
    hyperparameters, fitness_weights = pickle.loads(pickled_data)
    agents = []
    # Reconstruct agents from snapshot
    for i in range(len(state_dicts)):
        agent_class_str = agent_classes[i]
        # Overhead of moving to GPU is too high so only use CPU
        if agent_class_str == "RNN":
            agent = RNNAgent(hyperparameters, fitness_weights, False, torch.device("cpu"))
        elif agent_class_str == "LSTM":
            agent = LSTMAgent(hyperparameters, fitness_weights, False, torch.device("cpu"))
        elif agent_class_str == "GRU":
            agent = GRUAgent(hyperparameters, fitness_weights, False, torch.device("cpu"))
        else:
            raise ValueError(f"Unknown agent class: {agent_class_str}")
        agent.net.load_state_dict(state_dicts[i])  # Load agent parameters
        agents.append(agent)

    sim = AgarioSimulation(view_width=900, view_height=600,
                                             bounds=2000,
                                             food_count=700,
                                             virus_count=20)
    if generation <= 25:
        # Run without predators
        return sim.run_training(agents, 0, fps, simulation_duration, headless)
    else:
        # Run with predators
        return sim.run_training(agents, len(agents) // 2, fps, simulation_duration, headless)


def main():
    parser = argparse.ArgumentParser(description="Offline AI Agar.io Game")
    parser.add_argument('-wt', '--width', dest='width', type=int, default=900, help='screen width')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=600, help='screen height')
    parser.add_argument('-b', '--bounds', dest='bounds', type=int, default=1500, help='half-size of world bounds (world is [-b,b] x [-b,b])')
    parser.add_argument('-f', '--food', dest='food', type=int, default=700, help='initial food cell count')
    parser.add_argument('-v', '--viruses', dest='viruses', type=int, default=20, help='initial virus count')
    parser.add_argument('-sr', '--sras', dest='sras', type=int, default=5, help='number of SRAs')
    parser.add_argument('-mbr', '--mbras', dest='mbras', type=int, default=5, help='number of MBRA agents')
    parser.add_argument('-rnn', '--rnns', dest='rnns', type=int, default=5, help='number of RNN agents')
    parser.add_argument('-n', '--nick', dest='nick', type=str, default='Player', help='your nickname')
    parser.add_argument('-oc', '--opencv', dest='opencv', type=bool, default=True, help='Whether to use OpenCV view')
    args = parser.parse_args()

    bounds = [args.bounds, args.bounds]

    # Load best agent from snapshots file
    fitness_weights = FitnessWeights(food=0.1, time_alive=100.0, cells_eaten=50.0, score=0.9, death=500.0)
    '''
    # RNN agent
    hyperparameters = Hyperparameters(hidden_layers=[64, 16],
                                      output_size=4,
                                      run_interval=0.1,
                                      param_mutations={"weight": {"strength": 1.0, "chance": 0.05}, "bias": {"strength": 0.25, "chance": 0.025}},
                                      move_sensitivity=50.0,
                                      grid_width=9,
                                      grid_height=6,
                                      nodes_per_cell=4)
    base_rnn = RNNAgent.load_best_agent("rnn_agent_snapshots_216i_64h_16h_4o.pkl", hyperparameters, fitness_weights)'''

    # GRU agent
    hyperparameters = Hyperparameters(hidden_layers=[72],
                                      output_size=4,
                                      run_interval=0.1,
                                      param_mutations={"weight": {"strength": 1.0, "chance": 0.05}, "bias": {"strength": 0.25, "chance": 0.025}},
                                      move_sensitivity=50.0,
                                      grid_width=12,
                                      grid_height=8,
                                      nodes_per_cell=3)
    base_rnn = RNNAgent.load_best_agent("gru_agent_snapshots_288i_72h_4o.pkl", hyperparameters, fitness_weights)

    # Create model and players
    human_player = Player.make_random(args.nick, bounds)
    sra_players = []
    sra_agents = []
    mbra_players = []
    mbra_agents = []
    rnn_players = []
    rnn_agents = []
    for i in range(args.sras):
        sra_agents.append(SimpleReflexAgent(0.1, fitness_weights, 50.0))
        sra_player = Player.make_random(f"SRA {i}", bounds, color=(255, 0, 0))
        sra_players.append(sra_player)
    for i in range(args.mbras):
        mbra_agents.append(ModelBasedReflexAgent(0.1, fitness_weights, 50.0))
        mbra_player = Player.make_random(f"SRA {i}", bounds, color=(255, 50, 120))
        mbra_players.append(mbra_player)
    for i in range(args.rnns):
        rnn_agents.append(base_rnn.copy())
        rnn_player = Player.make_random(f"RNN {i}", bounds, color=(0, 255, 0))
        rnn_players.append(rnn_player)

    players = [human_player] + sra_players + mbra_players + rnn_players
    model = Model(players, bounds=bounds, chunk_size=args.bounds // 10)
    model.spawn_cells(args.food)
    model.spawn_viruses(args.viruses)

    print("In spectate mode, use the arrow keys to cycle through the player cells")
    print("In playing mode, use WASD to move, space to split, and w to eject")

    if args.opencv:
        view = OCView(args.width, args.height, model, human_player)
        frame = 0

        while True:
            if model.num_viruses < args.viruses:
                model.spawn_viruses(args.viruses - model.num_viruses)

            if model.num_cells < args.food:
                model.spawn_cells(args.food - model.num_cells)

            # Run SRAs
            for i in range(len(sra_players)):
                if not sra_players[i].alive:
                    continue

                target_pos, split, eject = AgarioSimulation.sra_tick(args.width, args.height, sra_agents[i], sra_players[i], model)

                if split > 0:
                    model.split(sra_players[i], target_pos)
                if eject > 0:
                    model.shoot(sra_players[i], target_pos)
                model.update_velocity(sra_players[i], target_pos)

            # Run MBRA agents
            for i in range(len(mbra_players)):
                if not mbra_players[i].alive:
                    continue

                target_pos, split, eject = AgarioSimulation.mbra_tick(args.width, args.height, mbra_agents[i], mbra_players[i], model, frame)

                if split > 0:
                    model.split(mbra_players[i], target_pos)
                if eject > 0:
                    model.shoot(mbra_players[i], target_pos)
                model.update_velocity(mbra_players[i], target_pos)

            # Run RNN agents
            for i in range(len(rnn_players)):
                if not rnn_players[i].alive:
                    continue
                rnn_players[i].ticks_alive += 1
                target_pos, split, eject = AgarioSimulation.rnn_tick(args.width, args.height, rnn_agents[i], rnn_players[i], model)
                if split > 0:
                    model.split(rnn_players[i], target_pos)
                if eject > 0:
                    model.shoot(rnn_players[i], target_pos)
                model.update_velocity(rnn_players[i], target_pos)

            view.redraw(spectate_mode=True)
            model.update()
            frame += 1
            time.sleep(0.01)
    else:
        # Need to implement pygame viewing still
        pygame.init()
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption('Agar.io Offline')
        view = View(screen, model, players[0], debug=False)
        view.start_human_game(args.food, args.viruses)


if __name__ == '__main__':
    main()

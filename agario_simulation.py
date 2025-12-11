import time
import torch
import argparse
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from game.opencv_view import OCView, OCCamera
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
    def mbra_tick(view_width: int, view_height: int, agent: ModelBasedReflexAgent, player: Player, model: Model,
                  simulation_tick: int):
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

        return agent.get_action(threats, prey, foods, viruses, center_pos, min_area, max_area, max_radius,
                                simulation_tick)

    @staticmethod
    def generate_vision_grid_v1(agent: RNNAgent | LSTMAgent | GRUAgent, center_pos: tuple[float, float], model: Model,
                                view_bounds: tuple[tuple[float, float], tuple[float, float]]):
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
        # print(max_food_perimeter, max_virus_perimeter, max_player_perimeter)
        vision_grid /= torch.tensor(
            [100.0, 330.0, 8000.0],  # Absolute max values determined manually
            device=vision_grid.device,
            dtype=vision_grid.dtype
        )
        vision_grid.nan_to_num_(nan=0.0)  # NaN values come from divisions by 0 (when max values are 0)
        return vision_grid

    @staticmethod
    def rnn_tick(view_width: int, view_height: int, agent: RNNAgent | LSTMAgent | GRUAgent, player: Player,
                 model: Model):
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

    def run(self, agents: list[RNNAgent | LSTMAgent | GRUAgent], num_sra: int, num_mbra: int, base_fps: int,
            simulation_duration: float, headless: bool, spectating: bool = True):
        """
        Run a simulation meant for training with the RNN agents and a number of SRAs.
        Stops when the duration is reached or when only 1 agent is alive.
        :param agents: List of RNN agents to spawn
        :param num_sra: Number of SRAs to spawn
        :param num_mbra: Numbere of MBRAs to spawn
        :param base_fps: Frames per second for a game at normal speed.
        :param simulation_duration: How long the simulation runs in seconds. Doesn't mean termination will occur at this time.
        :param headless: Whether to draw the game's visuals.
        :param spectating: Whether to spectate the simulation or have the user be a player in the simulation.
        :return: List of fitness values for each RNN agent
        """
        num_frames = int(simulation_duration * base_fps)

        bounds = [self.bounds, self.bounds]

        rnn_players = []
        prev_target_pos = []

        for i in range(len(agents)):
            # RNN color is green
            ai_player = Player.make_random(f"RNN {i}", bounds, color=(0, 255, 0))
            rnn_players.append(ai_player)
            prev_target_pos.append(rnn_players[i].center())

        sras = []
        sra_players = []
        for i in range(num_sra):
            # SRA color is red
            ai_player = Player.make_random(f"SRA {i}", bounds, color=(255, 0, 0))
            sra_players.append(ai_player)
            prev_target_pos.append(sra_players[i].center())
            sras.append(SimpleReflexAgent(agents[0].run_interval, agents[0].fitness_weights, 50.0))

        mbras = []
        mbra_players = []
        for i in range(num_mbra):
            # MBRA color is pink
            ai_player = Player.make_random(f"MBRA {i}", bounds, color=(255, 50, 120))
            mbra_players.append(ai_player)
            prev_target_pos.append(mbra_players[i].center())
            mbras.append(ModelBasedReflexAgent(agents[0].run_interval, agents[0].fitness_weights, 50.0))

        model = Model(rnn_players + sra_players, bounds=bounds, chunk_size=self.bounds // 10)  # 21 chunks per side
        model.spawn_cells(self.food_count)
        model.spawn_viruses(self.virus_count)
        print(f"Simulation model initialized with {model.chunk_count_x}x{model.chunk_count_y}={model.chunk_count_x * model.chunk_count_y} chunks.")

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
                for i in range(len(sra_players)):
                    if sra_players[i].alive:
                        sra_players[i].ticks_alive += 1
                for i in range(len(mbra_players)):
                    if mbra_players[i].alive:
                        mbra_players[i].ticks_alive += 1
                for i in range(len(rnn_players)):
                    if rnn_players[i].alive:
                        rnn_players[i].ticks_alive += 1
                        model.update_velocity(rnn_players[i], prev_target_pos[i])
            else:
                # Run SRA actions
                for i in range(len(sras)):
                    if not sra_players[i].alive:
                        continue
                    sra_players[i].ticks_alive += 1

                    target_pos, split, eject = AgarioSimulation.sra_tick(self.view_width, self.view_height, sras[i],
                                                                         sra_players[i], model)

                    if split > 0:
                        model.split(sra_players[i], target_pos)
                    if eject > 0:
                        model.shoot(sra_players[i], target_pos)
                    model.update_velocity(sra_players[i], target_pos)

                # Run MBRA actions
                for i in range(len(mbras)):
                    if not mbra_players[i].alive:
                        continue
                    mbra_players[i].ticks_alive += 1

                    target_pos, split, eject = AgarioSimulation.mbra_tick(self.view_width, self.view_height, mbras[i],
                                                                          mbra_players[i], model, frame)

                    if split > 0:
                        model.split(mbra_players[i], target_pos)
                    if eject > 0:
                        model.shoot(mbra_players[i], target_pos)
                    model.update_velocity(mbra_players[i], target_pos)

                # Run RNN actions
                rnn_agents_alive = 0
                for i in range(len(agents)):
                    if not rnn_players[i].alive:
                        continue
                    rnn_players[i].ticks_alive += 1

                    rnn_agents_alive += 1

                    target_pos, split, eject = self.rnn_tick(self.view_width, self.view_height, agents[i],
                                                             rnn_players[i], model)

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

        rnn_fitnesses = []
        rnn_agents_alive = 0
        for i in range(len(agents)):
            if rnn_players[i].alive:
                rnn_agents_alive += 1
            rnn_fitnesses.append(agents[i].calculate_fitness(rnn_players[i].num_food_eaten,
                                                             rnn_players[i].ticks_alive / num_frames,
                                                             rnn_players[i].num_players_eaten,
                                                             rnn_players[i].score() + rnn_players[i].highest_score,
                                                             int(not rnn_players[i].alive)))
        sra_fitnesses = []
        sras_alive = 0
        for i in range(len(sras)):
            if sra_players[i].alive:
                sras_alive += 1
            sra_fitnesses.append(sras[i].calculate_fitness(sra_players[i].num_food_eaten,
                                                           sra_players[i].ticks_alive / num_frames,
                                                           sra_players[i].num_players_eaten,
                                                           sra_players[i].score() + sra_players[i].highest_score,
                                                           int(not sra_players[i].alive)))

        mbra_fitnesses = []
        mbras_alive = 0
        for i in range(len(mbras)):
            if mbra_players[i].alive:
                mbras_alive += 1
            mbra_fitnesses.append(sras[i].calculate_fitness(mbra_players[i].num_food_eaten,
                                                            mbra_players[i].ticks_alive / num_frames,
                                                            mbra_players[i].num_players_eaten,
                                                            mbra_players[i].score() + mbra_players[i].highest_score,
                                                            int(not mbra_players[i].alive)))

        print(f"Simulation complete after {time.time() - start_time:.2f} seconds with {rnn_agents_alive} RNN agents alive, {sras_alive} SRAs alive, and {mbras_alive} MBRAs alive")
        return rnn_fitnesses, sra_fitnesses, mbra_fitnesses


# Function needs to be picklable for parallel processing so keep it out of classes
def run_sim_worker(fps: int, simulation_duration: float, state_dicts: list[dict], agent_classes: list[str],
                   pickled_data: bytes, num_sra: int, num_mbra: int, headless: bool):
    """
    Function intended to run a simulation in parallel which reconstructs agents and runs a simulation with them.
    Returns final fitness of all the RNN agents in this simulation.
    :param fps: "Normal" frames per second for simulation. Basically, defines how fast a normal game runs for humans.
    :param simulation_duration: Duration of simulation in seconds
    :param state_dicts: List of parameters
    :param agent_classes: List of agent classes in string format
    :param pickled_data: hyperparameters and fitness weights
    :param num_sra: Number of SRA agents
    :param num_mbra: Number of MBRA agents
    :param headless: Whether to visualize the simulation or not
    :return: (rnn_fitnesses, sra_fitnesses, mbra_fitnesses)
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

    return sim.run(agents, num_sra, num_mbra, fps, simulation_duration, headless)


def main():
    parser = argparse.ArgumentParser(description="Offline AI Agar.io Game")
    parser.add_argument('-wt', '--width', dest='width', type=int, default=900, help='screen width')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=600, help='screen height')
    parser.add_argument('-b', '--bounds', dest='bounds', type=int, default=1500, help='half-size of world bounds (world is [-b,b] x [-b,b])')
    parser.add_argument('-f', '--food', dest='food', type=int, default=700, help='initial food cell count')
    parser.add_argument('-v', '--viruses', dest='viruses', type=int, default=20, help='initial virus count')
    parser.add_argument('-rnn', '--rnns', dest='rnns', type=int, default=5, help='number of RNN agents')
    parser.add_argument('-sr', '--sras', dest='sras', type=int, default=5, help='number of SRAs')
    parser.add_argument('-mbr', '--mbras', dest='mbras', type=int, default=5, help='number of MBRA agents')
    parser.add_argument('-n', '--num_simulations', dest='num_simulations', type=int, default=10, help='number of simulations to run')
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
                                      param_mutations={"weight": {"strength": 1.0, "chance": 0.05},
                                                       "bias": {"strength": 0.25, "chance": 0.025}},
                                      move_sensitivity=50.0,
                                      grid_width=12,
                                      grid_height=8,
                                      nodes_per_cell=3)
    base_rnn = RNNAgent.load_best_agent("gru_agent_snapshots_288i_72h_4o.pkl", hyperparameters, fitness_weights)

    # Create RNNs
    rnn_agents = []
    state_dicts = []
    agent_classes = []
    pickled_data = pickle.dumps((hyperparameters, fitness_weights))
    for i in range(args.rnns):
        rnn_agents.append(base_rnn.copy())
        state_dicts.append(base_rnn.net.state_dict())
        agent_classes.append("GRU")

    print("In spectate mode, use the left and right arrow keys to cycle through the player cells")

    sim = AgarioSimulation(args.width, args.height, args.bounds, args.food, args.viruses)
    pool = Pool(processes=6)
    jobs = []
    # Run simulations in parallel, one worker per simulation
    jobs.append(pool.apply_async(
        run_sim_worker,
        args=(60, 300, state_dicts, agent_classes, pickled_data, args.sras, args.mbras, True,)
    ))

    pool.close()  # no more tasks
    # Wait for all jobs to finish and collect fitness
    for job in tqdm(jobs, desc="Running Simulations", total=len(jobs), unit="sims"):
        sim_fitnesses, sra_fitnesses, mbra_fitnesses = job.get()

    pool.join()
    sim.run(rnn_agents, args.sras, args.mbras, 60, args.duration, args.headless)


if __name__ == '__main__':
    main()

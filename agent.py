import time
import threading
import geometry_utils
import random
import math
import torch
from image_processing import ImageProcessing
from web_scraper import WebScraper


class Hyperparameters:
    def __init__(self, hidden_layers: list[int], output_size: int, run_interval: float, param_mutations: dict, move_sensitivity: float, grid_width: int, grid_height: int, nodes_per_cell: int):
        self.hidden_layers = hidden_layers  # Defines number of hidden nodes at layer i
        self.output_size = output_size
        self.run_interval = run_interval  # Time between actions in seconds
        self.param_mutations = param_mutations  # Dict holding param mutation standard deviations Ex: {"weight": 0.5, "bias": 0.1}
        self.move_sensitivity = move_sensitivity  # Factor to multiply the move output vector by
        self.grid_width = grid_width  # How many cells wide the vision grid is
        self.grid_height = grid_height  # How many cells tall the vision grid is
        self.nodes_per_cell = nodes_per_cell  # Number of features per grid cell
        self.input_size = grid_width * grid_height * self.nodes_per_cell

    def copy(self):
        """
        Creates a copy of this hyperparameter
        :return:
        """
        copied = Hyperparameters(
            hidden_layers=self.hidden_layers.copy(),
            output_size=self.output_size,
            run_interval=self.run_interval,
            param_mutations=self.param_mutations.copy(),
            move_sensitivity=self.move_sensitivity,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            nodes_per_cell=self.nodes_per_cell
        )
        copied.input_size = self.input_size
        return copied


class FitnessWeights:
    def __init__(self, food: float, time_alive: float, cells_eaten: float, score: float, death: float):
        self.food = food
        self.time_alive = time_alive
        self.cells_eaten = cells_eaten
        self.score = score
        self.death = death


class BaseAgent(threading.Thread):
    def __init__(self, run_interval: float, fitness_weights: FitnessWeights | None):
        super().__init__()
        self.run_interval = run_interval
        self.program_running = True
        self.alive = False

        self.scraper = WebScraper()
        self.image_processor = ImageProcessing()

        self.fitness_weights = fitness_weights
        self.fitnesses = []  # List of fitnesses for each simulation
        self.avg_fitness = 0.0

    def calculate_fitness(self, food_eaten: int, time_alive: float, cells_eaten: int, score: float, died: int):
        """
        Calculates the fitness of the agent based on the given statistics.
        :param food_eaten: Number of food eaten
        :param time_alive: Proportion of time alive out of total game time (0 to 1)
        :param cells_eaten: Number of cells eaten
        :param score: Score achieved
        :param died: Binary death indicator (0 or 1)
        :return: Fitness score
        """
        return (self.fitness_weights.food * food_eaten +
                self.fitness_weights.time_alive * time_alive +
                self.fitness_weights.cells_eaten * cells_eaten +
                self.fitness_weights.score * score
                - self.fitness_weights.death * died)

    def start_web_game(self):
        """
        Generates random ID for new username and start a game on agar.io
        """
        self.scraper.connect_to_game()

        random_id = random.randint(100, 1000)  # ID for this agent, can use as name in game to differentiate it (maybe)

        self.scraper.press_continue(wait=True)

        if not self.scraper.enter_name(name=str(random_id), wait=True):
            print("Failed to enter name")
            return

        if not self.scraper.play_game(wait=True):
            print("Failed to play game")
            return

        print("Game started")

    def get_web_game_data(self, visualize: bool = False, verbose: bool = False):
        """
        Extracts data from the web game via the scraper
        """
        canvas_png = self.scraper.screenshot_canvas_image()
        img = self.image_processor.convert_to_mat(canvas_png)
        objects = self.image_processor.object_recognition(img, visualize, verbose)
        return objects

    def run_web_game(self, visualize: bool):
        """
        Runs a game on agar.io using the web scraper for this agent with no logic for testing purposes.
        Basically manual control for the agent's logic to test object recognition and fitness calculation.
        :return:
        """
        self.start_web_game()
        while self.program_running:
            if self.scraper.in_game():
                self.alive = True
                objects = self.get_web_game_data(visualize=visualize, verbose=True)
                print(f"\nObjects: {objects}")
            else:
                if self.alive:  # Just died
                    print("Agent died. Calculating fitness...")
                    stats = self.scraper.get_stats(wait=True)
                    if stats is None:
                        print("Warning: Failed to get stats for agent")
                        self.alive = False
                        return None
                    food_eaten, time_alive, cells_eaten, highest_mass = stats
                    fitness = self.calculate_fitness(food_eaten, time_alive, cells_eaten, highest_mass, 1)
                    self.alive = False
                    return fitness
            time.sleep(self.run_interval)


class CustomRNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, device: torch.device):
        """
        hidden_sizes: list like [16, 32, 20] meaning:
            layer 0: 16 hidden units
            layer 1: 32 hidden units
            layer 2: 20 hidden units
        """
        super().__init__()

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.device = device

        # Create RNNCell for each layer
        self.cells = torch.nn.ModuleList()

        for i, h in enumerate(hidden_sizes):
            inp = input_size if i == 0 else hidden_sizes[i - 1]
            self.cells.append(torch.nn.RNNCell(inp, h, device=self.device))

        # Final linear output layer
        self.fc = torch.nn.Linear(hidden_sizes[-1], output_size, device=self.device)

    def forward(self, x, h=None):
        """
        x: (batch, seq_len, input_size)
        h: list of hidden states for each layer (optional)
        """
        batch, seq_len, _ = x.size()

        # Initialize hidden states if not provided
        if h is None:
            h = [
                torch.zeros(batch, hs, device=self.device)
                for hs in self.hidden_sizes
            ]
        else:
            h = [state.to(self.device) for state in h]

        inp = x[:, 0]
        # Process sequence
        for t in range(seq_len):
            inp = x[:, t]

            # Pass through each layer manually
            for layer in range(self.num_layers):
                h[layer] = self.cells[layer](inp, h[layer])
                inp = h[layer]  # output of current layer is input to next

        # Output from final layer goes to output head
        out = self.fc(inp)
        return out, h


class RNNAgent(BaseAgent):
    def __init__(self, hyperparameters: Hyperparameters | None, fitness_weights: FitnessWeights | None, randomize_params: bool, device: torch.device):
        super().__init__(hyperparameters.run_interval, fitness_weights)
        self.hyperparameters = hyperparameters
        self.device = device

        self.rnn = CustomRNN(hyperparameters.input_size, hyperparameters.hidden_layers, hyperparameters.output_size, self.device)
        self.hidden = None  # Hidden states for each layer (memory)

        # Randomize the parameters if specified
        if randomize_params:
            for name, param in list(self.rnn.named_parameters()):
                sigma = 0
                # Find the mutation hyperparam associated with this parameter
                for key, value in hyperparameters.param_mutations.items():
                    if key in name:
                        sigma = value
                        break

                noise = torch.randn_like(param) * sigma  # Gaussian noise
                param.data.add_(noise)

    def forward(self, x):
        """
        Feeds the input through the network and updates hidden states
        :param x: input to the network
        :return: network's output
        """
        output, h = self.rnn.forward(x.to(self.device), self.hidden)
        self.hidden = h
        return output

    def reduce_sigma(self, factor: float):
        """
        Reduce the standard deviation of mutation as the agent becomes more fit.
        :param factor: factor to reduce standard deviation by
        """
        for mutation in self.hyperparameters.param_mutations:
            self.hyperparameters.param_mutations[mutation] *= factor

    def mutate(self):
        """
        Mutates this agent's model parameters with normal distribution perturbations
        """
        for name, param in list(self.rnn.named_parameters()):
            sigma = 0
            # Find the mutation hyperparam associated with this parameter
            for key, value in self.hyperparameters.param_mutations.items():
                if key in name:
                    sigma = value
                    break

            noise = torch.randn_like(param) * sigma  # Gaussian noise
            param.data.add_(noise)

    def init_grid(self):
        """
        Generates an empty grid of shape (grid_width, grid_height, nodes_per_cell)
        :return: the torch grid
        """
        return torch.zeros((self.hyperparameters.grid_width, self.hyperparameters.grid_height, self.hyperparameters.nodes_per_cell),
                           device=self.device,
                           dtype=torch.float32)

    def get_grid_index(self, pos: tuple[float, float]):
        """
        Converts given position from [-1, 1] to grid index (x, y)
        :param pos: Vector position to convert
        :return:
        """
        # Convert pos.x/y in [-1,1] to grid index 0..GRID_SIZE-1
        gx = int((pos[0] + 1) * 0.5 * (self.hyperparameters.grid_width - 1))
        gy = int((pos[1] + 1) * 0.5 * (self.hyperparameters.grid_height - 1))

        # Safety clamp (just in case)
        gx = max(0, min(self.hyperparameters.grid_width - 1, gx))
        gy = max(0, min(self.hyperparameters.grid_height - 1, gy))

        return gx, gy

    def get_action(self, vision_grid: torch.Tensor):
        """
        Returns the action of the RNN after inputting the vision grid
        :param vision_grid: torch.Tensor of shape (GRID_WIDTH, GRID_HEIGHT, nodes_per_cell)
        :return: tuple of (move_x, move_y, split, eject)
        """
        # Flatten grid to shape (1, input_size)
        x = vision_grid.reshape(1, self.hyperparameters.input_size)

        # Feed the RNN: (batch, seq_len=1, features)
        x = x.unsqueeze(1)
        output = self.forward(x)

        # Return action values
        move_x, move_y, split, eject = output[0].detach().cpu().numpy()
        return float(move_x), float(move_y), float(split), float(eject)

    def run_web_game(self, visualize: bool):
        """
        Runs a single game for this agent on agar.io using the web scraper.
        Meant for testing purposes not for training.
        :return: fitness
        """
        self.start_web_game()
        while self.program_running:
            if self.scraper.in_game():
                self.alive = True
                objects = self.get_web_game_data(visualize=visualize)

                move_x, move_y, split, eject = self.get_action(objects)

                self.scraper.move(move_x, move_y, 5)
                if split > 0:
                    self.scraper.press_space()
                if eject > 0:
                    self.scraper.press_w()
            else:
                if self.alive:  # Just died
                    stats = self.scraper.get_stats(wait=True)
                    if stats is None:
                        print("Warning: Failed to get stats for agent")
                        self.alive = False
                        return None
                    food_eaten, time_alive, cells_eaten, score = stats
                    fitness = self.calculate_fitness(food_eaten, time_alive, cells_eaten, score, int(not self.alive))
                    self.alive = False
                    return fitness
            time.sleep(self.run_interval)

    def copy(self):
        """
        Creates a copy of this agent
        :return: copy of this agent
        """
        copy = RNNAgent(self.hyperparameters.copy(), self.fitness_weights, False, self.device)
        copy.rnn.load_state_dict(self.rnn.state_dict())
        return copy


class ModelBasedReflexAgent(BaseAgent):
    """
    Model-based reflex agent using rule-based decision making instead of neural networks.
    """

    VIRUS_DANGER_SIZE = 300.0
    THREAT_SIZE_RATIO = 1.1
    PREY_SIZE_RATIO = 0.83
    SPLIT_DISTANCE_THRESHOLD = 350.0
    VIRUS_AVOID_DISTANCE = 100.0

    def __init__(self, run_interval: float, fitness_weights: FitnessWeights, move_sensitivity: float = 50.0):
        super().__init__(run_interval, fitness_weights)
        self.move_sensitivity = move_sensitivity
        self.my_area = 0.0
        self.last_action = (0.0, 0.0, 0.0, 0.0)
        # Memory buffer for observed objects. Each entry is a dict:
        # {"label": str, "pos": (x,y), "area": float, "last_seen": timestamp}
        self.memory = []
        self.memory_size = 200  # max elements to keep in memory
        # Label decay rates (lambda) as specified
        self._label_lambda = {
            "food": 0.9,
            "virus": 0.5,
            "player": 0.25,
            "other": 1.0
        }
        self._match_distance = 100.0  # distance to consider the same object across ticks

    def get_action(self, threats: list, prey: list, foods: list, viruses: list, my_pos: tuple[float, float], min_area: float, max_area: float, my_radius: float):
        """
        Rule-based decision logic.
        :param threats: List of visible threats
        :param prey: List of visible prey
        :param foods: List of visible foods
        :param viruses: List of visible viruses
        :param my_pos: Current position of the agent
        :param min_area: Min area of parts of this player
        :param max_area: Max area of parts of this player
        :return: tuple of (move_x, move_y, split, eject)
        """
        self.my_area = max_area

        target_pos = [my_pos[0], my_pos[1]]
        split, eject = 0.0, 0.0

        # Rule 1: Flee from threats (highest priority)
        running = False
        # Update memory with currently visible objects
        now = time.time()
        def add_visible(label, obj_list):
            for obj in obj_list:
                # obj can be (x,y) or (x,y,area)
                x, y = obj[0], obj[1]
                area = obj[2] if len(obj) > 2 else 1.0
                self._update_memory_entry(label, (x, y), area, now)

        add_visible("player", [(t[0], t[1], max_area) for t in threats])
        add_visible("player", [(p[0], p[1], p[2]) for p in prey])
        add_visible("food", [(f[0], f[1]) for f in foods])
        add_visible("virus", [(v[0], v[1]) for v in viruses])

        # Build effective threat list by combining visible threats and recent memory of large players
        effective_threats = []
        for m in self.memory:
            if m["label"] == "player" and m["area"] > max_area * self.THREAT_SIZE_RATIO:
                # consider recent memories only
                if now - m["last_seen"] < 5.0:
                    effective_threats.append((m["pos"][0], m["pos"][1]))
        # add visible threats too
        for t in threats:
            effective_threats.append((t[0], t[1]))

        if effective_threats:
            closest_threat = min(effective_threats, key=lambda o: geometry_utils.sqr_distance(my_pos[0], my_pos[1], o[0], o[1]))
            threat_dist = geometry_utils.sqr_distance(my_pos[0], my_pos[1], closest_threat[0], closest_threat[1])
            if threat_dist < self.SPLIT_DISTANCE_THRESHOLD * self.SPLIT_DISTANCE_THRESHOLD:
                dx = my_pos[0] - closest_threat[0]
                dy = my_pos[1] - closest_threat[1]
                urgency = self.SPLIT_DISTANCE_THRESHOLD * self.SPLIT_DISTANCE_THRESHOLD - threat_dist
                target_pos[0] = my_pos[0] + dx * urgency
                target_pos[1] = my_pos[1] + dy * urgency
                running = True

        # Rule 2: Chase prey (visible first, then memory)
        if not running:
            chosen_prey = None
            if prey:
                chosen_prey = min(prey, key=lambda o: geometry_utils.sqr_distance(my_pos[0], my_pos[1], o[0], o[1]))
            else:
                # check memory for recent prey
                recent_preys = [m for m in self.memory if m["label"] == "player" and m["area"] < max_area * self.PREY_SIZE_RATIO and (now - m["last_seen"] < 5.0)]
                if recent_preys:
                    m = min(recent_preys, key=lambda o: geometry_utils.sqr_distance(my_pos[0], my_pos[1], o["pos"][0], o["pos"][1]))
                    chosen_prey = (m["pos"][0], m["pos"][1], m["area"])

            if chosen_prey:
                closest_prey = chosen_prey
                prey_dist = geometry_utils.sqr_distance(my_pos[0], my_pos[1], closest_prey[0], closest_prey[1])
                target_pos[0] = closest_prey[0]
                target_pos[1] = closest_prey[1]

                # Split attack if close enough and can eat after split
                threshold = self.SPLIT_DISTANCE_THRESHOLD + my_radius
                if prey_dist < threshold * threshold:
                    half_my_area = min_area * 0.5
                    if half_my_area * self.PREY_SIZE_RATIO > closest_prey[2]:
                        split = 1.0
            # Rule 3: Eat food (visible then memory)
            else:
                chosen_food = None
                if foods:
                    chosen_food = min(foods, key=lambda o: geometry_utils.sqr_distance(my_pos[0], my_pos[1], o[0], o[1]))
                else:
                    recent_foods = [m for m in self.memory if m["label"] == "food" and (now - m["last_seen"] < 10.0)]
                    if recent_foods:
                        m = min(recent_foods, key=lambda o: geometry_utils.sqr_distance(my_pos[0], my_pos[1], o["pos"][0], o["pos"][1]))
                        chosen_food = (m["pos"][0], m["pos"][1])

                if chosen_food:
                    closest_food = chosen_food
                    target_pos[0] = closest_food[0]
                    target_pos[1] = closest_food[1]

        # Virus avoidance (when large enough)
        if viruses and self.my_area > self.VIRUS_DANGER_SIZE:
            for virus in viruses:
                virus_dist = geometry_utils.sqr_distance(my_pos[0], my_pos[1], virus[0], virus[1])
                avoid_dst_sqr = self.VIRUS_AVOID_DISTANCE * self.VIRUS_AVOID_DISTANCE
                if virus_dist < avoid_dst_sqr:
                    avoid_strength = (avoid_dst_sqr - virus_dist) / avoid_dst_sqr
                    dx = virus[0] - my_pos[0]
                    dy = virus[1] - my_pos[1]
                    target_pos[0] -= dx * avoid_strength
                    target_pos[1] -= dy * avoid_strength

        self.last_action = (target_pos, split, eject)
        return self.last_action

    def _update_memory_entry(self, label: str, pos: tuple[float, float], area: float, timestamp: float):
        """
        Insert or update a memory entry for an observed object.
        Matches to existing entries by proximity.
        """
        # Try to match by proximity
        best_idx = None
        best_dist = None
        for i, m in enumerate(self.memory):
            if m["label"] != label:
                continue
            d = math.hypot(m["pos"][0] - pos[0], m["pos"][1] - pos[1])
            if best_dist is None or d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None and best_dist is not None and best_dist < self._match_distance:
            # update existing
            self.memory[best_idx]["pos"] = pos
            self.memory[best_idx]["area"] = area
            self.memory[best_idx]["last_seen"] = timestamp
        else:
            # add new entry
            self.memory.append({"label": label, "pos": pos, "area": area, "last_seen": timestamp})

        # Trim memory if necessary
        self._trim_memory(timestamp)

    def _compute_priority(self, entry: dict, now: float) -> float:
        """Compute decayed priority for a memory entry."""
        area = max(1.0, entry.get("area", 1.0))
        label = entry.get("label", "other")
        lamb = self._label_lambda.get(label, 1.0)
        dt = now - entry.get("last_seen", now)
        # priority = ln(area) * (1 - lambda) ** dt
        try:
            base = math.log(area)
        except Exception:
            base = 0.0
        decay = pow(max(0.0, 1.0 - lamb), dt)
        return base * decay

    def _trim_memory(self, now: float):
        """Trim memory to at most memory_size entries by dropping lowest priority."""
        if len(self.memory) <= self.memory_size:
            return
        # Compute priorities
        priorities = [(self._compute_priority(m, now), i) for i, m in enumerate(self.memory)]
        # Sort by priority ascending and drop the lowest until size constraint satisfied
        priorities.sort(key=lambda x: x[0])
        num_to_drop = len(self.memory) - self.memory_size
        drop_indices = set(i for _, i in priorities[:num_to_drop])
        self.memory = [m for idx, m in enumerate(self.memory) if idx not in drop_indices]

    def run_web_game(self, visualize: bool):
        """
        Run the agent on the real agar.io website using web scraper.
        :param visualize: Whether to show visualization window
        :return: Final fitness score or None if stats retrieval fails
        """
        self.start_web_game()

        while self.program_running:
            if self.scraper.in_game():
                self.alive = True
                objects = self.get_web_game_data(visualize=visualize)

                # Estimate own size from nearby player objects
                estimated_area = 400.0
                for obj in objects:
                    if obj.label == "player" and abs(obj.pos.x) < 0.1 and abs(obj.pos.y) < 0.1:
                        estimated_area = obj.area
                        break

                target_pos, split, eject = self.get_action(objects, estimated_area)

                self.scraper.move(target_pos[0],
                                  target_pos[1], 5)

                if split > 0:
                    self.scraper.press_space()
                if eject > 0:
                    self.scraper.press_w()
            else:
                if self.alive:
                    print("ModelBasedReflexAgent died. Calculating fitness...")
                    stats = self.scraper.get_stats(wait=True)

                    if stats is None:
                        print("Warning: Failed to get stats for agent")
                        self.alive = False
                        return None

                    food_eaten, time_alive, cells_eaten, highest_mass = stats
                    fitness = self.calculate_fitness(food_eaten, time_alive, cells_eaten, highest_mass, 1)

                    print(f"Fitness: {fitness:.2f} (Food: {food_eaten}, Time: {time_alive}s, "
                          f"Cells: {cells_eaten}, Max Mass: {highest_mass})")

                    self.alive = False
                    return fitness

            time.sleep(self.run_interval)

        return None

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
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.run_interval = run_interval
        self.param_mutations = param_mutations
        self.move_sensitivity = move_sensitivity
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.nodes_per_cell = nodes_per_cell
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
        self.fitnesses = []
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

        random_id = random.randint(100, 1000)

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
        self.epsilon = 0.005

    def forward(self, x):
        """
        Feeds the input through the network and updates hidden states
        :param x: input to the network
        :return: network's output
        """
        output, h = self.rnn.forward(x.to(self.device), self.hidden)
        self.hidden = h
        return output

    def update_sigma(self, factor: float, base_param_mutations: dict):
        """
        Reduce the standard deviation of mutation as the agent becomes more fit.
        Resets mutation standard deviation to original if it goes below epsilon.
        :param factor: factor to reduce standard deviation by
        :param base_param_mutations: the original mutation standard deviations
        """
        for mutation in self.hyperparameters.param_mutations:
            if self.hyperparameters.param_mutations[mutation] < self.epsilon:
                self.hyperparameters.param_mutations[mutation] = base_param_mutations[mutation]
            else:
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


class MemoryItem:
    """
    Represents a single object in the memory buffer with priority and timestamp.
    """
    def __init__(self, obj_type: str, pos: tuple[float, float], priority: float, 
                 timestamp: int, area: float = 0.0, radius: float = 0.0):
        self.obj_type = obj_type
        self.pos = pos
        self.priority = priority
        self.timestamp = timestamp
        self.area = area
        self.radius = radius
        self.initial_priority = priority

    def decay(self, decay_factor: float, current_tick: int):
        """
        Apply exponential decay to priority based on time elapsed.
        :param decay_factor: Decay factor (0.9-0.95)
        :param current_tick: Current simulation tick
        """
        ticks_elapsed = current_tick - self.timestamp
        if ticks_elapsed > 0:
            self.priority = self.initial_priority * (decay_factor ** ticks_elapsed)

    def update(self, pos: tuple[float, float], priority: float, timestamp: int, 
               area: float = 0.0, radius: float = 0.0):
        """
        Update memory item with new observation (resets priority).
        :param pos: New position
        :param priority: New priority (usually initial priority)
        :param timestamp: Current timestamp
        :param area: Object area
        :param radius: Object radius
        """
        self.pos = pos
        self.priority = priority
        self.initial_priority = priority
        self.timestamp = timestamp
        self.area = area
        self.radius = radius

    def distance_to(self, other_pos: tuple[float, float]) -> float:
        """
        Calculate squared distance to another position.
        :param other_pos: (x, y) position
        :return: Squared distance
        """
        dx = self.pos[0] - other_pos[0]
        dy = self.pos[1] - other_pos[1]
        return dx * dx + dy * dy


class MemoryBuffer:
    """
    Manages memory items for different object types with priority decay.
    """
    def __init__(self, decay_factor: float = 0.92, priority_threshold: float = 0.1, 
                 max_size_per_type: int = 100, distance_weight_factor: float = 500.0):
        self.decay_factor = decay_factor
        self.priority_threshold = priority_threshold
        self.max_size_per_type = max_size_per_type
        self.distance_weight_factor = distance_weight_factor
        
        self.threats: list[MemoryItem] = []
        self.prey: list[MemoryItem] = []
        self.foods: list[MemoryItem] = []
        self.viruses: list[MemoryItem] = []
        
        self.initial_priorities = {
            "threat": 1.0,
            "prey": 0.8,
            "food": 0.6,
            "virus": 0.7
        }

    def _get_buffer(self, obj_type: str) -> list[MemoryItem]:
        """Get the appropriate buffer list for object type."""
        if obj_type == "threat":
            return self.threats
        elif obj_type == "prey":
            return self.prey
        elif obj_type == "food":
            return self.foods
        elif obj_type == "virus":
            return self.viruses
        else:
            raise ValueError(f"Unknown object type: {obj_type}")

    def _find_nearby_item(self, obj_type: str, pos: tuple[float, float], 
                         threshold_distance: float = 50.0) -> MemoryItem | None:
        """
        Find a memory item near the given position (within threshold).
        :param obj_type: Type of object
        :param pos: Position to search near
        :param threshold_distance: Maximum distance to consider as "same" object
        :return: MemoryItem if found, None otherwise
        """
        buffer = self._get_buffer(obj_type)
        threshold_sqr = threshold_distance * threshold_distance
        
        for item in buffer:
            if item.distance_to(pos) < threshold_sqr:
                return item
        return None

    def _update_object_type(self, obj_type: str, objects: list, current_tick: int):
        """Helper method to update objects of a specific type."""
        for obj_data in objects:
            if isinstance(obj_data, tuple) and len(obj_data) >= 2:
                pos = (obj_data[0], obj_data[1])
                area = obj_data[2] if len(obj_data) > 2 else 0.0
                item = self._find_nearby_item(obj_type, pos)
                priority = self.initial_priorities[obj_type]
                if item:
                    item.update(pos, priority, current_tick, area=area)
                else:
                    self.add_item(obj_type, pos, current_tick, area=area)

    def update_with_visible_objects(self, threats: list, prey: list, foods: list, 
                                    viruses: list, current_tick: int):
        """
        Update memory buffer with currently visible objects.
        Objects that are visible get their priority reset to initial value.
        :param threats: List of (x, y) tuples for threats
        :param prey: List of (x, y, area) tuples for prey
        :param foods: List of (x, y) tuples for foods
        :param viruses: List of (x, y) tuples for viruses
        :param current_tick: Current simulation tick
        """
        self._update_object_type("threat", threats, current_tick)
        self._update_object_type("prey", prey, current_tick)
        self._update_object_type("food", foods, current_tick)
        self._update_object_type("virus", viruses, current_tick)

    def add_item(self, obj_type: str, pos: tuple[float, float], timestamp: int, 
                 area: float = 0.0, radius: float = 0.0):
        """
        Add a new memory item to the buffer.
        :param obj_type: Type of object
        :param pos: Position (x, y)
        :param timestamp: Current timestamp
        :param area: Object area
        :param radius: Object radius
        """
        buffer = self._get_buffer(obj_type)
        priority = self.initial_priorities[obj_type]
        
        if len(buffer) >= self.max_size_per_type:
            buffer.pop(0)
        
        new_item = MemoryItem(obj_type, pos, priority, timestamp, area, radius)
        buffer.append(new_item)

    def decay_all(self, current_tick: int):
        """
        Apply decay to all memory items and remove low-priority ones.
        :param current_tick: Current simulation tick
        """
        for obj_type in ["threat", "prey", "food", "virus"]:
            buffer = self._get_buffer(obj_type)
            for item in buffer:
                item.decay(self.decay_factor, current_tick)
            buffer[:] = [item for item in buffer if item.priority >= self.priority_threshold]

    def _merge_memory_items(self, memory_items: list, current_objects: list, 
                            my_pos: tuple[float, float], include_area: bool = False):
        """Helper method to merge memory items with current visible objects."""
        merged = list(current_objects)
        current_positions = [(pos[0], pos[1]) if isinstance(pos, tuple) and len(pos) >= 2 
                            else pos for pos in current_objects]
        
        for item in memory_items:
            if not any(self._is_same_position(item.pos, pos, 50.0) for pos in current_positions):
                distance = math.sqrt(item.distance_to(my_pos))
                adjusted_priority = item.priority / (1.0 + distance / self.distance_weight_factor)
                if adjusted_priority >= self.priority_threshold:
                    merged.append((item.pos[0], item.pos[1], item.area) if include_area else item.pos)
        return merged

    def get_merged_objects(self, my_pos: tuple[float, float], 
                          current_threats: list, current_prey: list, 
                          current_foods: list, current_viruses: list):
        """
        Merge current visible objects with memory buffer objects.
        Returns merged lists with priority-weighted objects.
        :param my_pos: Agent's current position
        :param current_threats: Currently visible threats
        :param current_prey: Currently visible prey
        :param current_foods: Currently visible foods
        :param current_viruses: Currently visible viruses
        :return: Tuple of (merged_threats, merged_prey, merged_foods, merged_viruses)
        """
        merged_threats = self._merge_memory_items(self.threats, current_threats, my_pos)
        merged_prey = self._merge_memory_items(self.prey, current_prey, my_pos, include_area=True)
        merged_foods = self._merge_memory_items(self.foods, current_foods, my_pos)
        merged_viruses = self._merge_memory_items(self.viruses, current_viruses, my_pos)
        return merged_threats, merged_prey, merged_foods, merged_viruses

    @staticmethod
    def _is_same_position(pos1: tuple[float, float], pos2: tuple[float, float], 
                          threshold: float) -> bool:
        """Check if two positions are the same (within threshold)."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx * dx + dy * dy) < (threshold * threshold)

      
class SimpleReflexAgent(BaseAgent):
    """
    Simple reflex agent using rule-based decision-making instead of neural networks.
    """

    VIRUS_DANGER_SIZE = 300.0
    THREAT_SIZE_RATIO = 1.1
    PREY_SIZE_RATIO = 0.83
    SPLIT_DISTANCE_THRESHOLD = 350.0
    VIRUS_AVOID_DISTANCE = 100.0

    def __init__(self, run_interval: float, fitness_weights: FitnessWeights, move_sensitivity: float = 50.0,
                 decay_factor: float = 0.92, priority_threshold: float = 0.1, 
                 max_memory_size: int = 100, distance_weight_factor: float = 500.0):
        super().__init__(run_interval, fitness_weights)
        self.move_sensitivity = move_sensitivity
        self.my_area = 0.0
        self.last_action = (0.0, 0.0, 0.0, 0.0)
        self.memory_buffer = MemoryBuffer(decay_factor=decay_factor, 
                                         priority_threshold=priority_threshold,
                                         max_size_per_type=max_memory_size,
                                         distance_weight_factor=distance_weight_factor)
        self.current_tick = 0

    def get_action(self, threats: list, prey: list, foods: list, viruses: list, my_pos: tuple[float, float], min_area: float, max_area: float, my_radius: float, current_tick: int = None):
        """
        Rule-based decision logic with memory buffer integration.
        :param threats: List of visible threats
        :param prey: List of visible prey
        :param foods: List of visible foods
        :param viruses: List of visible viruses
        :param my_pos: Current position of the agent
        :param min_area: Min area of parts of this player
        :param max_area: Max area of parts of this player
        :param my_radius: Maximum radius of player parts
        :param current_tick: Current simulation tick (optional, uses internal counter if None)
        :return: tuple of (target_pos, split, eject)
        """
        self.my_area = max_area
        self.current_tick = current_tick if current_tick is not None else self.current_tick + 1
        
        self.memory_buffer.update_with_visible_objects(threats, prey, foods, viruses, self.current_tick)
        self.memory_buffer.decay_all(self.current_tick)
        merged_threats, merged_prey, merged_foods, merged_viruses = \
            self.memory_buffer.get_merged_objects(my_pos, threats, prey, foods, viruses)

        target_pos = [my_pos[0], my_pos[1]]
        split, eject = 0.0, 0.0

        split_threshold_sqr = self.SPLIT_DISTANCE_THRESHOLD * self.SPLIT_DISTANCE_THRESHOLD
        running = False
        
        if merged_threats:
            closest_threat = min(merged_threats, key=lambda o: geometry_utils.sqr_distance(my_pos[0], my_pos[1], o[0], o[1]))
            threat_dist = geometry_utils.sqr_distance(my_pos[0], my_pos[1], closest_threat[0], closest_threat[1])
            if threat_dist < split_threshold_sqr:
                dx, dy = my_pos[0] - closest_threat[0], my_pos[1] - closest_threat[1]
                urgency = split_threshold_sqr - threat_dist
                target_pos[0] = my_pos[0] + dx * urgency
                target_pos[1] = my_pos[1] + dy * urgency
                running = True

        if not running:
            if merged_prey:
                closest_prey = min(merged_prey, key=lambda o: geometry_utils.sqr_distance(my_pos[0], my_pos[1], o[0], o[1]))
                prey_dist = geometry_utils.sqr_distance(my_pos[0], my_pos[1], closest_prey[0], closest_prey[1])
                target_pos[0], target_pos[1] = closest_prey[0], closest_prey[1]
                
                attack_threshold = (self.SPLIT_DISTANCE_THRESHOLD + my_radius) ** 2
                if prey_dist < attack_threshold and len(closest_prey) >= 3:
                    if min_area * 0.5 * self.PREY_SIZE_RATIO > closest_prey[2]:
                        split = 1.0
            elif merged_foods:
                closest_food = min(merged_foods, key=lambda o: geometry_utils.sqr_distance(my_pos[0], my_pos[1], o[0], o[1]))
                target_pos[0], target_pos[1] = closest_food[0], closest_food[1]

        if merged_viruses and self.my_area > self.VIRUS_DANGER_SIZE:
            avoid_dst_sqr = self.VIRUS_AVOID_DISTANCE * self.VIRUS_AVOID_DISTANCE
            for virus in merged_viruses:
                virus_dist = geometry_utils.sqr_distance(my_pos[0], my_pos[1], virus[0], virus[1])
                if virus_dist < avoid_dst_sqr:
                    avoid_strength = (avoid_dst_sqr - virus_dist) / avoid_dst_sqr
                    dx = virus[0] - my_pos[0]
                    dy = virus[1] - my_pos[1]
                    target_pos[0] -= dx * avoid_strength
                    target_pos[1] -= dy * avoid_strength

        self.last_action = (target_pos, split, eject)
        return self.last_action

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
                    print("SimpleReflexAgent died. Calculating fitness...")
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

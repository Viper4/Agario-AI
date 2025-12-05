import time
import threading

import geometry_utils
from image_processing import ImageProcessing
from web_scraper import WebScraper
import random
import torch


class Hyperparameters:
    def __init__(self, hidden_layers: list[int], output_size: int, run_interval: float, param_mutations: dict, move_sensitivity: float, grid_width: int, grid_height: int):
        self.hidden_layers = hidden_layers  # Defines number of hidden nodes at layer i
        self.output_size = output_size
        self.run_interval = run_interval  # Time between actions in seconds
        self.param_mutations = param_mutations  # Dict holding param mutation standard deviations Ex: {"weight": 0.5, "bias": 0.1}
        self.move_sensitivity = move_sensitivity  # Factor to multiply the move output vector by
        self.grid_width = grid_width  # How many cells wide the vision grid is
        self.grid_height = grid_height  # How many cells tall the vision grid is
        self.nodes_per_cell = 4  # Number of features per grid cell
        self.input_size = grid_width * grid_height * self.nodes_per_cell


class FitnessWeights:
    def __init__(self, food: float, time_alive: float, cells_eaten: float, highest_mass: float, death: float):
        self.food = food
        self.time_alive = time_alive
        self.cells_eaten = cells_eaten
        self.highest_mass = highest_mass
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
        self.fitness = 0.0

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
                    self.fitness = (food_eaten * self.fitness_weights.food +
                                    time_alive * self.fitness_weights.time_alive +
                                    cells_eaten * self.fitness_weights.cells_eaten +
                                    highest_mass * self.fitness_weights.highest_mass)
                    self.alive = False
                    return self.fitness
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

        if hyperparameters is None:
            self.load_agent("agent.pth")
        else:
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

    def save_agent(self):
        """
        Saves this agent's hyperparameters and RNN to a file
        """
        torch.save((self.hyperparameters, self.rnn), "agent.pth")

    def load_agent(self, path: str):
        """
        Loads hyperparameters, and RNN from the given path
        :param path: str
        """
        self.hyperparameters, self.rnn = torch.load(path)

    def reduce_sigma(self, factor: float):
        """
        Reduce the standard deviation of mutation to the agent as the agent becomes more fit.
        :param factor: factor to reduce learning rate by
        """
        for mutation in self.hyperparameters.param_mutations:
            self.hyperparameters.param_mutations[mutation] *= factor

    def mutate(self):
        """
        Mutates this agent's model parameters with normal distribution perturbations
        """
        for name, param in list(self.rnn.named_parameters()):
            sigma = 0
            # Find the mutation hyperparam(s) associated with this parameter
            for key, value in self.hyperparameters.param_mutations.items():
                if key in name:
                    sigma = value
                    break

            noise = torch.randn_like(param) * sigma  # Gaussian noise
            param.data.add_(noise)

    def calculate_fitness(self, food_eaten: int, time_alive: float, cells_eaten: int, highest_mass: float, died: int):
        """
        Calculates the fitness of the agent based on the given statistics.
        :param food_score: Number of food eaten
        :param time_alive: Proportion of time alive out of total game time (0 to 1)
        :param cells_eaten: Number of cells eaten
        :param highest_mass: Highest mass achieved
        :param died: Binary death indicator (0 or 1)
        :return: Fitness score
        """
        return (self.fitness_weights.food * food_eaten +
                self.fitness_weights.time_alive * time_alive +
                self.fitness_weights.cells_eaten * cells_eaten +
                self.fitness_weights.highest_mass * highest_mass
                - self.fitness_weights.death * died)

    def init_grid(self):
        """
        Generates an empty grid of shape (grid_width, grid_height, nodes_per_cell)
        :return: the torch grid
        """
        return torch.zeros((self.hyperparameters.grid_width, self.hyperparameters.grid_height, self.hyperparameters.nodes_per_cell),
                           device=self.device,
                           dtype=torch.float32)

    def get_grid_index(self, pos: geometry_utils.Vector):
        """
        Converts given position from [-1, 1] to grid index (x, y)
        :param pos: Vector position to convert
        :return:
        """
        # Convert pos.x/y in [-1,1] to grid index 0..GRID_SIZE-1
        gx = int((pos.x + 1) * 0.5 * (self.hyperparameters.grid_width - 1))
        gy = int((pos.y + 1) * 0.5 * (self.hyperparameters.grid_height - 1))

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

        '''max_input_objects = self.hyperparameters.input_size // 8

        # Convert objects list to tensor input for the network
        x = torch.zeros((1, self.hyperparameters.input_size)).to(self.device)
        # 8 input nodes per object
        # The 8 nodes are formatted: (food, virus, player, relative pos x, relative pos y, area, perimeter, count)
        i = 0

        while i < max_input_objects and i < len(objects):
            obj = objects[i]

            # Encode label as one-hot
            nodes = [0, 0, 0,
                     obj.pos.x,
                     obj.pos.y,
                     obj.area,
                     obj.perimeter,
                     obj.count]
            if obj.label == "food":
                nodes[0] = 1
            elif obj.label == "player":
                nodes[1] = 1
            elif obj.label == "virus":
                nodes[2] = 1

            # Insert into x tensor
            start = i * 8
            end = start + 8
            x[0, start:end] = torch.tensor(nodes, dtype=torch.float32).to(self.device)
            i += 1

        # Feed the input through the network
        x = x.unsqueeze(1)  # (batch, input_size) -> (batch, seq_len=1, input_size)
        output = self.forward(x)

        # Take action with output: (move x, move y, split, eject)
        arr = output[0].detach().cpu().numpy()
        move_x = float(arr[0])
        move_y = float(arr[1])
        split = float(arr[2])
        eject = float(arr[3])
        return move_x, move_y, split, eject'''

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
                    food_eaten, time_alive, cells_eaten, highest_mass = stats
                    self.fitness = self.calculate_fitness(food_eaten, time_alive, cells_eaten, highest_mass, int(not self.alive))
                    self.alive = False
                    return self.fitness
            time.sleep(self.run_interval)


class ModelBasedReflexAgent(BaseAgent):
    def __init__(self, run_interval: float, fitness_weights: FitnessWeights):
        super().__init__(run_interval, fitness_weights)


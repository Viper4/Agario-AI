import os.path
import sys
import time
import threading
from image_processing import ImageProcessing
from web_scraper import WebScraper
import random
import torch
import numpy as np
import game


class Hyperparameters:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 run_interval: float, param_mutations: dict):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.run_interval = run_interval
        self.param_mutations = param_mutations


class FitnessWeights:
    def __init__(self, food: float, time_alive: float, cells_eaten: float, highest_mass: float):
        self.food = food
        self.time_alive = time_alive
        self.cells_eaten = cells_eaten
        self.highest_mass = highest_mass


class BaseAgent(threading.Thread):
    def __init__(self, run_interval: float, fitness_weights: FitnessWeights):
        super().__init__()
        self.run_interval = run_interval
        self.program_running = True
        self.alive = False

        self.scraper = WebScraper()
        self.image_processor = ImageProcessing()

        self.fitness_weights = fitness_weights
        self.fitness = 0.0

    def start_game(self):
        """
        Generates random ID for new username and plays the game
        """
        random_id = random.randint(100, 1000)  # ID for this agent, can use as name in game to differentiate it (maybe)

        self.scraper.press_continue(wait=True)

        if not self.scraper.enter_name(name=str(random_id), wait=True):
            print("Failed to enter name")
            return

        if not self.scraper.play_game(wait=True):
            print("Failed to play game")
            return

        print("Game started")

    def get_game_data(self, visualize: bool = False, verbose: bool = False):
        """
        Extracts data from the game
        """
        canvas_png = self.scraper.screenshot_canvas_image()
        img = self.image_processor.convert_to_mat(canvas_png)
        objects = self.image_processor.object_recognition(img, visualize, verbose)
        return objects

    def run_game(self, visualize: bool):
        """
        Runs a game for this agent with no logic for testing purposes.
        Basically manual control for the agent's logic to test object recognition and fitness calculation.
        :return:
        """
        self.start_game()
        while self.program_running:
            if self.scraper.in_game():
                self.alive = True
                objects = self.get_game_data(visualize=True, verbose=True)
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


class RNNAgent(BaseAgent):
    def __init__(self, hyperparameters: Hyperparameters, fitness_weights: FitnessWeights, randomize_params: bool, device: torch.device):
        super().__init__(hyperparameters.run_interval, fitness_weights)
        self.hyperparameters = hyperparameters
        self.device = device

        # Initialize hidden state to all 0s
        self.hidden = torch.zeros(1, hyperparameters.input_size, hyperparameters.hidden_size).to(self.device)

        # Set up the recurrent neural network
        self.rnn = torch.nn.RNN(hyperparameters.input_size, hyperparameters.hidden_size, hyperparameters.output_size, nonlinearity="tanh")
        self.fc = torch.nn.Linear(hyperparameters.hidden_size, hyperparameters.output_size)  # Fully connected layer

        # Randomize the parameters if specified
        if randomize_params:
            for name, param in (list(self.rnn.named_parameters()) + list(self.fc.named_parameters())):
                sigma = 0
                # Find the mutation hyperparam(s) associated with this parameter
                for key, value in hyperparameters.param_mutations:
                    if key in name:
                        sigma = value
                        break

                noise = torch.randn_like(param) * sigma  # Gaussian noise
                param.data.add_(noise)

    def forward(self, x):
        """
        Feeds the input through the network
        :param x: input to the network
        :return: network's output
        """
        output, h = self.rnn(x, self.hidden)
        self.hidden = h

        output = self.fc(output[:, -1, :])  # Get the last output layer
        return output

    def save_agent(self):
        """
        Saves this agent's data to a file
        """
        torch.save((self.hyperparameters, self.rnn, self.fc), "agent.pth")

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
        for name, param in (list(self.rnn.named_parameters()) + list(self.fc.named_parameters())):
            sigma = 0
            # Find the mutation hyperparam(s) associated with this parameter
            for key, value in self.hyperparameters.param_mutations:
                if key in name:
                    sigma = value
                    break

            noise = torch.randn_like(param) * sigma  # Gaussian noise
            param.data.add_(noise)

    def run_game(self, visualize: bool):
        """
        Runs a single game for this agent
        :return: fitness
        """
        max_input_objects = self.hyperparameters.input_size // 8
        self.start_game()
        while self.program_running:
            if self.scraper.in_game():
                self.alive = True
                objects = self.get_game_data(visualize=visualize)

                # Convert objects list to useable input for the network
                x = torch.zeros((1, self.hyperparameters.input_size))
                # 8 input nodes per object
                # The 8 nodes are formatted: (food, virus, player, pos x, pos y, area, perimeter, density)
                i = 0
                while i < max_input_objects and i < len(objects):
                    obj = objects[i]

                    # Encode label as one-hot
                    nodes = [0, 0, 0,
                             obj.pos.x,
                             obj.pos.y,
                             obj.area,
                             obj.perimeter,
                             obj.density]
                    if obj.label == "food":
                        nodes[0] = 1
                    elif obj.label == "player":
                        nodes[1] = 1
                    elif obj.label == "virus":
                        nodes[2] = 1

                    # Insert into x tensor
                    start = i * 8
                    end = start + 8
                    x[0, start:end] = torch.tensor(nodes, dtype=torch.float32)
                    i += 1

                output = self.forward(x)
                # Take action with output: (move x, move y, split, eject)
                move_x, move_y, split, eject = output[0]
                self.scraper.move(move_x, move_y, 1)
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
                    self.fitness = (food_eaten * self.fitness_weights.food +
                                    time_alive * self.fitness_weights.time_alive +
                                    cells_eaten * self.fitness_weights.cells_eaten +
                                    highest_mass * self.fitness_weights.highest_mass)
                    self.alive = False
                    return self.fitness
            time.sleep(self.run_interval)


class ModelBasedReflexAgent(BaseAgent):
    def __init__(self, run_interval, fitness_weights):
        super().__init__(run_interval, fitness_weights)


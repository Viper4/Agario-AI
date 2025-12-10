import os.path
import torch
import numpy as np
import random
import agario_simulation
import pickle
import sys
from agent import RNNAgent, LSTMAgent, GRUAgent, Hyperparameters, FitnessWeights
from multiprocessing import Pool
from tqdm import tqdm


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

    sim = agario_simulation.AgarioSimulation(view_width=900, view_height=600,
                                             bounds=2000,
                                             food_count=700,
                                             virus_count=20)
    # Run with predators
    return sim.run(agents, len(agents) // 2, fps, simulation_duration, headless)


class GeneticTrainer:
    def __init__(self, init_rnn_prop: float, init_lstm_prop: float, init_gru_prop: float, population_size: int,
                 hyperparameters: Hyperparameters, fitness_weights: FitnessWeights, save_file: str,
                 max_generations: int | None = None):
        if init_rnn_prop + init_lstm_prop + init_gru_prop != 1:
            raise ValueError("init_rnn_prop, init_lstm_prop, and init_gru_prop must sum to 1")
        self.init_rnn_prop = init_rnn_prop
        self.init_lstm_prop = init_lstm_prop
        self.init_gru_prop = init_gru_prop
        self.population_size = population_size
        self.hyperparameters = hyperparameters
        self.fitness_weights = fitness_weights
        self.save_file = save_file
        self.max_generations = max_generations
        self.population = []

    def init_agents(self, load_from_file: bool):
        """
        Initializes the population of agents.
        Loads agents from file and/or creates new agents with randomized parameters to fill the remaining population.
        :param load_from_file: Whether to load the population from file
        :return:
        """
        if load_from_file:
            if os.path.exists(self.save_file):
                print("Loading population from file...")
                with open(self.save_file, "rb") as f:
                    state_dicts, agent_classes = pickle.load(f)
                    num_rnns = 0
                    num_lstms = 0
                    num_grus = 0
                    for i in range(len(state_dicts)):
                        agent_class_str = agent_classes[i]
                        if agent_class_str == "RNN":
                            agent_class = RNNAgent
                            num_rnns += 1
                        elif agent_class_str == "LSTM":
                            agent_class = LSTMAgent
                            num_lstms += 1
                        elif agent_class_str == "GRU":
                            agent_class = GRUAgent
                            num_grus += 1
                        else:
                            raise ValueError(f"Unknown agent class: {agent_class_str}")
                        agent = agent_class(hyperparameters=self.hyperparameters.copy(),
                                            fitness_weights=self.fitness_weights,
                                            randomize_params=False, device=torch.device("cpu"))
                        agent.net.load_state_dict(state_dicts[i])
                        self.population.append(agent)
                    print(f"Loaded {num_rnns} RNNAgents, {num_lstms} LSTMAgents, and {num_grus} GRUAgents from file.")
            else:
                print(f"File {self.save_file} does not exist. Creating new population.")

        # Calculate difference to desired population size and determine to prune or add agents
        diff = self.population_size - len(self.population)
        if diff > 0:  # Add agents
            add_rnn = int(diff * self.init_rnn_prop)
            add_lstm = int(diff * self.init_lstm_prop)
            add_gru = diff - add_rnn - add_lstm
            for i in range(add_rnn):
                self.population.append(RNNAgent(self.hyperparameters.copy(), self.fitness_weights, randomize_params=True,
                                                device=torch.device("cpu")))
            for i in range(add_lstm):
                self.population.append(LSTMAgent(self.hyperparameters.copy(), self.fitness_weights, randomize_params=True,
                                                 device=torch.device("cpu")))
            for i in range(add_gru):
                self.population.append(GRUAgent(self.hyperparameters.copy(), self.fitness_weights, randomize_params=True,
                                                device=torch.device("cpu")))
            print(f"Added {add_rnn} RNNAgents, {add_lstm} LSTMAgents, and {add_gru} GRUAgents.")
        elif diff < 0:
            # Remove lowest fitness agents
            self.population = self.population[:self.population_size]

    def reproduce(self, parent1: RNNAgent | LSTMAgent | GRUAgent, parent2: RNNAgent | LSTMAgent | GRUAgent,
                  num_children: int):
        """
        Performs crossover between two parents and mutation to create n children.
        Each child inherits half of its parameters from each parent
        and is also mutated by Gaussian perturbation.
        :param parent1:
        :param parent2:
        :param num_children: Number of children to create
        :return: List of children
        """
        if type(parent1) != type(parent2):
            raise ValueError("Parent types must match to reproduce")
        children = []
        for i in range(num_children):
            agent_class = type(parent1)

            child = agent_class(parent1.hyperparameters.copy(), fitness_weights=parent1.fitness_weights,
                                randomize_params=False, device=torch.device("cpu"))

            '''# Crossover from both parents
            # Random crossover 50% chance for either parent at every parameter
            for (param1, param2, param_child) in zip(
                        parent1.rnn.parameters(),
                        parent2.rnn.parameters(),
                        child.rnn.parameters()):
                # Random binary mask for mixing parameters
                mask = torch.rand_like(param1) < 0.5
                param_child.data.copy_(
                    torch.where(mask, param1.data, param2.data)
                )'''

            child.mutate()
            child.update_sigma(factor=0.95, base_param_mutations=self.hyperparameters.param_mutations)
            children.append(child)
        return children

    def tournament_selection(self, subset: list[int], k: int = 3):
        """
        Returns index of best parent from random sample of k agents from the population[:num_parents]
        :param subset: List of indices of agents to select from
        :param k: Number of candidates in the tournament
        :return: Index of best parent from the tournament
        """
        candidates = random.sample(subset, min(k, len(subset)))
        best_i = candidates[0]
        best_fitness = self.population[candidates[0]].avg_fitness
        for candidate_i in candidates:
            fitness = self.population[candidate_i].avg_fitness
            if fitness > best_fitness:
                best_i = candidate_i
                best_fitness = fitness
        return best_i

    def train(self, load_from_file: bool, num_simulations: int):
        """
        Starts the training loop
        :param load_from_file: Whether to load the population from file.
        :param num_simulations: Number of simulations to run per generation.
        :return: Final population
        """
        generation = 0
        self.init_agents(load_from_file)

        while self.max_generations is None or generation < self.max_generations:
            state_dicts = []
            agent_classes = []
            for agent in self.population:
                agent.fitnesses.clear()  # Clear fitnesses for next simulations

                # Prepare data to reconstruct agents for multiprocessing
                state_dicts.append(agent.net.state_dict())
                if isinstance(agent, RNNAgent):
                    agent_classes.append("RNN")
                elif isinstance(agent, LSTMAgent):
                    agent_classes.append("LSTM")
                elif isinstance(agent, GRUAgent):
                    agent_classes.append("GRU")
                else:
                    raise ValueError(f"Unknown agent type: {type(agent)}")

            pickled_data = pickle.dumps((self.hyperparameters, self.fitness_weights))

            pool = Pool(processes=min(num_simulations, os.cpu_count() - 4))
            jobs = []
            try:
                # Run simulations in parallel, one worker per simulation
                for i in range(num_simulations):
                    jobs.append(pool.apply_async(
                        run_simulation_worker,
                        args=(60, 300, state_dicts, agent_classes, pickled_data, True, generation,)
                    ))

                pool.close()  # no more tasks
                # Wait for all jobs to finish and collect fitness
                for job in tqdm(jobs, desc=f"Generation {generation}", total=num_simulations, unit="sims"):
                    sim_fitnesses = job.get()
                    for i in range(self.population_size):
                        self.population[i].fitnesses.append(sim_fitnesses[i])

                pool.join()
            except KeyboardInterrupt:
                print("\nCtrl-C caught: terminating pool...")
                pool.terminate()  # force kill workers
                pool.join()
                sys.exit(1)

            total_final_fitness = 0.0
            for i in range(self.population_size):
                self.population[i].avg_fitness = sum(self.population[i].fitnesses) / num_simulations
                total_final_fitness += self.population[i].avg_fitness

            self.population.sort(key=lambda x: x.avg_fitness, reverse=True)
            # Split up agents into their types for reproduction pools
            rnn_agent_indices = []
            lstm_agent_indices = []
            gru_agent_indices = []

            # Save agent parameters and class types to file
            with open(self.save_file, "wb") as f:
                agent_state_dicts = []
                agent_classes = []
                for i in range(self.population_size):
                    agent = self.population[i]
                    agent_state_dicts.append(agent.net.state_dict())
                    if isinstance(agent, RNNAgent):
                        agent_classes.append("RNN")
                        if i < self.population_size // 2:
                            rnn_agent_indices.append(i)  # Add to reproduction pool
                    elif isinstance(agent, LSTMAgent):
                        agent_classes.append("LSTM")
                        if i < self.population_size // 2:
                            lstm_agent_indices.append(i)
                    elif isinstance(agent, GRUAgent):
                        agent_classes.append("GRU")
                        if i < self.population_size // 2:
                            gru_agent_indices.append(i)
                pickle.dump((agent_state_dicts, agent_classes), f)  # Save agents in order of fitness

            # Create new population
            new_population = []

            num_elites = self.population_size // 10
            #num_parents = self.population_size // 2 - num_elites
            child_counts = [0] * self.population_size

            # Elitism
            for i in range(num_elites):
                new_population.append(self.population[i])

            # Calculate number of children needed to fill population
            diff = self.population_size - num_elites

            # Tournament selection
            for i in range(diff):
                # Tournament selection
                if isinstance(self.population[i], RNNAgent):
                    parent1 = self.tournament_selection(rnn_agent_indices)
                    parent2 = self.tournament_selection(rnn_agent_indices)
                elif isinstance(self.population[i], LSTMAgent):
                    parent1 = self.tournament_selection(lstm_agent_indices)
                    parent2 = self.tournament_selection(lstm_agent_indices)
                elif isinstance(self.population[i], GRUAgent):
                    parent1 = self.tournament_selection(gru_agent_indices)
                    parent2 = self.tournament_selection(gru_agent_indices)
                else:
                    raise ValueError(f"Unknown agent type: {type(self.population[i])}")
                children = self.reproduce(self.population[parent1], self.population[parent2], 1)
                new_population.extend(children)
                child_counts[parent1] += 1
                child_counts[parent2] += 1

            # Pair mating
            '''for i in range(0, num_parents, 2):
                if i+1 >= self.population_size:
                    break
                # Reproduce in pairs: 0: (0,1), 1: (2,3),..., floor(n/4): (floor(n/2)-1,floor(n/2))
                parent1 = self.population[i]
                parent2 = self.population[i + 1]

                # Linear allocation of children based on rank
                a = 4
                b = a / num_parents
                children = self.reproduce(parent1, parent2, round(a-b*i + a-b*(i+1)))
                child_counts[i] = len(children)
                child_counts[i + 1] = len(children)
                new_population.extend(children)'''

            # Print generation statistics
            print(f"Generation {generation} complete")
            print(f"Mean fitness: {total_final_fitness / self.population_size:.2f}")
            print(f"Fitness standard deviation: {np.std([x.avg_fitness for x in self.population]).item():.4f}")
            print(f"Mutation strengths: {self.population[0].hyperparameters.param_mutations}")
            individual_string = ""
            for i in range(num_simulations):
                individual_string += "\t| Sim " + str(i + 1) + " Fitness"
            label_string = f"Type\t| Avg Fitness{individual_string}\t| Children"
            print(label_string)
            print("-" * (len(label_string) + 50))
            for i in range(self.population_size):
                fitnesses_string = ""
                for j in range(num_simulations):
                    fitnesses_string += f"\t| {repr(self.population[i].fitnesses[j] + 0.0000000001)[:10]}"
                print(
                    f"{type(self.population[i]).__name__.replace('Agent', '')}\t| {repr(self.population[i].avg_fitness + 0.0000000001)[:10]}{fitnesses_string}\t| {child_counts[i]}")
            print("-" * (len(label_string) + 50))

            self.population = new_population
            generation += 1
        return self.population


if __name__ == "__main__":
    grid_width = 12
    grid_height = 8
    nodes_per_cell = 3
    num_inputs = grid_width * grid_height * nodes_per_cell
    hyperparameters = Hyperparameters(hidden_layers=[72],
                                      output_size=4,
                                      run_interval=0.1,
                                      param_mutations={"weight": {"strength": 1.0, "chance": 0.05},
                                                       "bias": {"strength": 0.25, "chance": 0.025}},
                                      move_sensitivity=50.0,
                                      grid_width=grid_width,
                                      grid_height=grid_height,
                                      nodes_per_cell=nodes_per_cell)
    fitness_weights = FitnessWeights(food=0.1, time_alive=100.0, cells_eaten=10.0, score=0.9, death=500.0)

    trainer = GeneticTrainer(init_rnn_prop=0.33333,
                             init_lstm_prop=0.33333,
                             init_gru_prop=0.33334,
                             population_size=int(input("Enter population size> ")),
                             hyperparameters=hyperparameters,
                             fitness_weights=fitness_weights,
                             save_file="agent_snapshots.pkl")
    trainer.train(load_from_file=True, num_simulations=6)

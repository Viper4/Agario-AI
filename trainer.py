import torch
import numpy as np
import random
import agario_simulation
import json
from agent import RNNAgent, Hyperparameters, FitnessWeights
from multiprocessing import Pool
from tqdm import tqdm


class GeneticTrainer:
    def __init__(self, population_size: int, hyperparameters: Hyperparameters, fitness_weights: FitnessWeights, max_generations: int | None = None):
        self.population_size = population_size
        self.hyperparameters = hyperparameters
        self.fitness_weights = fitness_weights
        self.max_generations = max_generations
        self.population = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_agents(self):
        """
        Initializes the population of agents with randomized parameters
        :return:
        """
        for i in range(self.population_size):
            self.population.append(RNNAgent(self.hyperparameters, fitness_weights=self.fitness_weights,
                                            randomize_params=True, device=self.device))

    def reproduce(self, parent1: RNNAgent, parent2: RNNAgent, num_children: int):
        """
        Performs crossover between two parents and mutation to create n children.
        Each child inherits half of its parameters from each parent
        and is also mutated by Gaussian perturbation.
        :param parent1:
        :param parent2:
        :param num_children: Number of children to create
        :return: List of children
        """
        children = []
        for i in range(num_children):
            child = RNNAgent(self.hyperparameters, fitness_weights=self.fitness_weights,
                             randomize_params=False, device=self.device)

            # Crossover from both parents
            for (param1, param2, param_child) in zip(
                        parent1.rnn.parameters(),
                        parent2.rnn.parameters(),
                        child.rnn.parameters()):
                # Random binary mask for mixing parameters
                mask = torch.rand_like(param1) < 0.5
                param_child.data.copy_(
                    torch.where(mask, param1.data, param2.data)
                )

            child.mutate()
            child.reduce_sigma(0.9)  # Reduce mutation strength
            children.append(child)
        return children

    def train(self, num_simulations: int):
        """
        Starts the training loop
        :param num_simulations: Number of simulations to run per generation.
        :return: Final population
        """
        generation = 0
        self.init_agents()

        with open("cluster_settings.json") as f:
            cluster_settings = json.load(f)

        while self.max_generations is None or generation < self.max_generations:
            simulation = agario_simulation.AgarioSimulation(900, 600, 1500, 500, 20, self.population)
            total_sim_fitnesses = [0.0] * self.population_size  # Element at i = total fitness of agent i over all simulations

            # Run simulations in sequence
            for _ in tqdm(range(num_simulations), desc=f"Generation {generation}", total=num_simulations):
                sim_fitnesses = simulation.run_headless(cluster_settings, 0.01, 240)
                for i in range(self.population_size):
                    total_sim_fitnesses[i] += sim_fitnesses[i]

            # Run simulations in parallel
            # TODO: Implement parallel simulation
            # - Cant pickle the RNNs, so figure out how to pass them to the child processes
            # - Maybe copy over all the agents and simulation from self.population to the child process somehow
            '''pool = Pool(processes=8)
            jobs = []
            for i in range(num_simulations):
                jobs.append(pool.apply_async(simulation.run_headless, args=(cluster_settings, 0.1, 240)))
            pool.close()
            pool.join()

            # Wait for all jobs to finish and collect fitness
            for job in tqdm(jobs, desc=f"Generation {generation}", total=num_simulations):
                sim_fitnesses = job.get()
                for i in range(self.population_size):
                    total_sim_fitnesses[i] += sim_fitnesses[i]'''

            total_final_fitness = 0.0
            for i in range(self.population_size):
                self.population[i].fitness = total_sim_fitnesses[i] / num_simulations
                total_final_fitness += self.population[i].fitness

            # Print generation statistics
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            print(f"Generation {generation} complete")
            print(f"Mean fitness: {total_final_fitness / self.population_size:.2f}")
            print(f"Fitness standard deviation: {repr(np.std([x.fitness for x in self.population]))}")
            print(f"Rank\t\t|\t\tFitness")
            print("-" * 20)
            for i in range(self.population_size):
                print(f"{i}\t\t|\t\t{repr(self.population[i].fitness)}")
            print("-" * 20)

            # Create new population
            new_population = []
            num_parents = self.population_size // 2
            for i in range(0, num_parents, 2):
                if i+1 >= self.population_size:
                    break
                # Reproduce in pairs: 0: (0,1), 1: (2,3),..., floor(n/4): (floor(n/2)-1,floor(n/2))
                parent1 = self.population[i]
                parent2 = self.population[i + 1]

                # Linear allocation of children based on rank
                a = 4
                b = a / num_parents
                children = self.reproduce(parent1, parent2, round(a-b*i + a-b*(i+1)))
                new_population.extend(children)
            diff = self.population_size - len(new_population)
            if diff < 0:  # Too many children
                # Cutoff the bottom
                new_population = new_population[:self.population_size]
            elif diff > 0:  # Not enough children
                # Reproduce random parents from top half
                for i in range(diff):
                    # Small chance to pick the same parent twice, so cloning is sometimes possible
                    parent1 = random.choice(self.population[:self.population_size//2])
                    parent2 = random.choice(self.population[:self.population_size//2])
                    children = self.reproduce(parent1, parent2, 1)
                    new_population.extend(children)

            self.population = new_population
            generation += 1
        return self.population


if __name__ == "__main__":
    # Max number of objects on screen at a time reaches ~50 so define fixed input of 32 objects with 8 nodes per object
    hyperparameters = Hyperparameters(input_size=256,
                                      hidden_layers=[64],
                                      output_size=4,
                                      run_interval=0.2,
                                      param_mutations={"weight": 0.5, "bias": 0.25},
                                      move_sensitivity=50.0)
    fitness_weights = FitnessWeights(food=0.75, time_alive=0.5, cells_eaten=2.0, highest_mass=1.5)

    trainer = GeneticTrainer(population_size=int(input("Enter population size> ")),
                             hyperparameters=hyperparameters,
                             fitness_weights=fitness_weights)
    trainer.train(5)

    '''command = input("Enter command> ")
    if command == "train":
        
    elif command == "test":
        model_selection = input("Select model to test (0=None, 1=Model Based, 2=Neural Network)> ")
        if model_selection == "0":
            agent = BaseAgent(0.25, fitness_weights)
            print(f"Game finished with {agent.run_web_game(visualize=True)} fitness")
        elif model_selection == "1":
            model_based_agent = ModelBasedReflexAgent(run_interval=0.1, fitness_weights=fitness_weights)
            print(f"Game finished with {model_based_agent.run_web_game(True)}")
        elif model_selection == "2":
            network_agent = RNNAgent(hyperparameters=hyperparameters,
                                     fitness_weights=fitness_weights,
                                     randomize_params=False,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print(f"Game finished with {network_agent.run_web_game(True)} fitness")
    else:
        print("Invalid command")'''

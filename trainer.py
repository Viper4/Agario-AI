import os.path

import torch
import numpy as np
import random
import agario_simulation
import json
import pickle
import sys
from agent import RNNAgent, Hyperparameters, FitnessWeights
from multiprocessing import Pool
from tqdm import tqdm


# Function needs to be picklable so keep it out of classes
def run_simulation_worker(fps: int, simulation_duration: float, agent_snapshots: list[dict], pickled_data: bytes):
    hyperparameters, fitness_weights = pickle.loads(pickled_data)
    agents = []
    for agent_snapshot in agent_snapshots:
        # Reconstruct agent from snapshot
        agent = RNNAgent(hyperparameters, fitness_weights, False, torch.device("cpu"))  # CPU only for multiprocessing
        agent.rnn.load_state_dict(agent_snapshot)  # Load agent parameters from snapshot
        agents.append(agent)
    sim = agario_simulation.AgarioSimulation(900, 600, 1500, 600, 20, agents)
    return sim.run(fps, simulation_duration, True)


class GeneticTrainer:
    def __init__(self, population_size: int, hyperparameters: Hyperparameters, fitness_weights: FitnessWeights, max_generations: int | None = None):
        self.population_size = population_size
        self.hyperparameters = hyperparameters
        self.fitness_weights = fitness_weights
        self.max_generations = max_generations
        self.population = []

    def init_agents(self, load_from_file: bool):
        """
        Initializes the population of agents with randomized parameters.
        :param load_from_file: Whether to load the population from file
        :return:
        """
        if load_from_file and os.path.exists("agent_snapshots.pkl"):
            print("Loading population from file...")
            with open("agent_snapshots.pkl", "rb") as f:
                agent_snapshots = pickle.load(f)
                for agent_snapshot in agent_snapshots:
                    agent = RNNAgent(self.hyperparameters, fitness_weights=self.fitness_weights,
                                     randomize_params=False, device=torch.device("cpu"))
                    agent.rnn.load_state_dict(agent_snapshot)
                    self.population.append(agent)

            # Remove extras
            if len(self.population) > self.population_size:
                self.population = self.population[:self.population_size]

        # Fill remaining agents with randomized agents
        for i in range(self.population_size - len(self.population)):
            self.population.append(RNNAgent(self.hyperparameters, fitness_weights=self.fitness_weights,
                                            randomize_params=True, device=torch.device("cpu")))

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
                             randomize_params=False, device=torch.device("cpu"))

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
            child.reduce_sigma(0.95)  # Reduce mutation strength
            children.append(child)
        return children

    def train(self, load_from_file: bool, num_simulations: int):
        """
        Starts the training loop
        :param load_from_file: Whether to load the population from file.
        :param num_simulations: Number of simulations to run per generation.
        :return: Final population
        """
        generation = 0
        self.init_agents(load_from_file)

        with open("cluster_settings.json") as f:
            cluster_settings = json.load(f)

        while self.max_generations is None or generation < self.max_generations:
            total_sim_fitnesses = [0.0] * self.population_size  # Element at i = total fitness of agent i over all simulations

            # Run simulations in parallel, one worker per simulation
            # Prepare state dicts of agents
            agent_snapshots = [agent.rnn.state_dict() for agent in self.population]
            pickled_data = pickle.dumps((self.hyperparameters, self.fitness_weights))

            pool = Pool(processes=max(num_simulations, 8))
            jobs = []
            try:
                for i in range(num_simulations):
                    jobs.append(pool.apply_async(
                        run_simulation_worker,
                        args=(60, 240, agent_snapshots, pickled_data,)
                    ))

                pool.close()  # no more tasks
                # Wait for all jobs to finish and collect fitness
                for job in tqdm(jobs, desc=f"Generation {generation}", total=num_simulations, unit="sims"):
                    try:
                        sim_fitnesses = job.get()
                        for i in range(self.population_size):
                            total_sim_fitnesses[i] += sim_fitnesses[i]
                    except TimeoutError:
                        # job still running, loop back to allow keyboard interrupt checking
                        continue

                pool.join()
            except KeyboardInterrupt:
                print("\nCtrl-C caught: terminating pool...")
                pool.terminate()  # force kill workers
                pool.join()
                sys.exit(1)

            total_final_fitness = 0.0
            for i in range(self.population_size):
                self.population[i].fitness = total_sim_fitnesses[i] / num_simulations
                total_final_fitness += self.population[i].fitness

            # Print generation statistics
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            print(f"Generation {generation} complete")
            print(f"Mean fitness: {total_final_fitness / self.population_size:.2f}")
            print(f"Fitness standard deviation: {np.std([x.fitness for x in self.population]).item():.4f}")
            print(f"Rank\t\t|\t\tFitness")
            print("-" * 40)
            for i in range(self.population_size):
                print(f"{i}\t\t|\t\t{repr(self.population[i].fitness)}")
            print("-" * 40)

            # Save agent parameters to file
            with open("agent_snapshots.pkl", "wb") as f:
                agent_snapshots = [agent.rnn.state_dict() for agent in self.population]
                pickle.dump(agent_snapshots, f)

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

            # Elitism
            num_elites = self.population_size // 20
            new_population[-num_elites:] = self.population[:num_elites]

            self.population = new_population
            generation += 1
        return self.population


if __name__ == "__main__":
    # Max number of objects on screen at a time reaches ~50 so define fixed input of 32 objects with 8 nodes per object
    hyperparameters = Hyperparameters(hidden_layers=[64, 16],
                                      output_size=4,
                                      run_interval=0.1,
                                      param_mutations={"weight": 2.0, "bias": 0.5},
                                      move_sensitivity=50.0,
                                      grid_width=9,
                                      grid_height=6)
    fitness_weights = FitnessWeights(food=0.1, time_alive=10.0, cells_eaten=10.0, highest_mass=1.0, death=100.0)

    trainer = GeneticTrainer(population_size=int(input("Enter population size> ")),
                             hyperparameters=hyperparameters,
                             fitness_weights=fitness_weights)
    trainer.train(True, 5)

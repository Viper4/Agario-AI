import os.path
import torch
import numpy as np
import random
import agario_simulation
import pickle
import sys
from agent import RNNAgent, Hyperparameters, FitnessWeights
from multiprocessing import Pool
from tqdm import tqdm


# Function needs to be picklable so keep it out of classes
def run_simulation_worker(fps: int, simulation_duration: float, agent_snapshots: list[dict], pickled_data: bytes, headless: bool):
    hyperparameters, fitness_weights = pickle.loads(pickled_data)
    rnn_agents = []
    for agent_snapshot in agent_snapshots:
        # Reconstruct agent from snapshot
        agent = RNNAgent(hyperparameters, fitness_weights, False, torch.device("cpu"))  # Overhead of moving to GPU is too high
        agent.rnn.load_state_dict(agent_snapshot)  # Load agent parameters from snapshot
        rnn_agents.append(agent)
    sim = agario_simulation.AgarioSimulation(view_width=900, view_height=600,
                                             bounds=1500,
                                             food_count=600,
                                             virus_count=20)
    return sim.run(rnn_agents, len(rnn_agents) // 5, fps, simulation_duration, headless)


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
                    agent = RNNAgent(hyperparameters=self.hyperparameters.copy(), fitness_weights=self.fitness_weights,
                                     randomize_params=False, device=torch.device("cpu"))
                    agent.rnn.load_state_dict(agent_snapshot)
                    self.population.append(agent)

            # Remove extras
            if len(self.population) > self.population_size:
                self.population = self.population[:self.population_size]

        # Fill remaining agents with randomized agents
        for i in range(self.population_size - len(self.population)):
            self.population.append(RNNAgent(self.hyperparameters.copy(), fitness_weights=self.fitness_weights,
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
            child = RNNAgent(parent1.hyperparameters.copy(), fitness_weights=parent1.fitness_weights,
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

        while self.max_generations is None or generation < self.max_generations:
            for i in range(self.population_size):
                self.population[i].fitnesses.clear()

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
                        args=(60, 240, agent_snapshots, pickled_data, True,)
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

            # Print generation statistics
            self.population.sort(key=lambda x: x.avg_fitness, reverse=True)

            num_elites = self.population_size // 20
            num_parents = self.population_size // 2 - num_elites
            child_counts = []
            for i in range(0, self.population_size, 2):
                if i+1 < num_parents:
                    # Linear allocation based on rank
                    a = 4
                    b = a / num_parents
                    num_children = round(a-b*i + a-b*(i+1))
                    child_counts.append(num_children)
                    child_counts.append(num_children)
                else:
                    child_counts.append(0)
                    child_counts.append(0)

            print(f"Generation {generation} complete")
            print(f"Mean fitness: {total_final_fitness / self.population_size:.2f}")
            print(f"Fitness standard deviation: {np.std([x.avg_fitness for x in self.population]).item():.4f}")
            print(f"Mutation strengths: {self.population[0].hyperparameters.param_mutations}")
            individual_string = ""
            for i in range(num_simulations):
                individual_string += "\t|\tSim " + str(i+1) + " Fitness"
            label_string = f"Rank\t|\tAvg Fitness{individual_string}\t|\tChildren"
            print(label_string)
            print("-" * (len(label_string)+50))
            for i in range(self.population_size):
                fitnesses_string = ""
                for j in range(num_simulations):
                    fitnesses_string += f"\t|\t{repr(self.population[i].fitnesses[j] + 0.0000000001)[:10]}"
                print(f"{i}\t|\t{repr(self.population[i].avg_fitness + 0.0000000001)[:10]}{fitnesses_string}\t|\t{child_counts[i]}")
            print("-" * (len(label_string)+50))

            # Save agent parameters to file
            with open("agent_snapshots.pkl", "wb") as f:
                agent_snapshots = [agent.rnn.state_dict() for agent in self.population]
                pickle.dump(agent_snapshots, f)  # Save agent parameters in order of fitness

            # Create new population
            new_population = []

            # Elitism
            for i in range(num_elites):
                new_population.append(self.population[i])

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
    hyperparameters = Hyperparameters(hidden_layers=[64, 16],
                                      output_size=4,
                                      run_interval=0.1,
                                      param_mutations={"weight": 2.0, "bias": 0.5},
                                      move_sensitivity=50.0,
                                      grid_width=9,
                                      grid_height=6,
                                      nodes_per_cell=4)
    fitness_weights = FitnessWeights(food=0.5, time_alive=100.0, cells_eaten=50.0, score=0.75, death=500.0)

    trainer = GeneticTrainer(population_size=int(input("Enter population size> ")),
                             hyperparameters=hyperparameters,
                             fitness_weights=fitness_weights)
    trainer.train(True, 5)

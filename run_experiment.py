"""
Experiment script for comparing RNN agents vs MBRA agents.
Runs multiple simulations and collects fitness statistics for the paper.

Based on paper requirements:
- 5 RNN agents vs 5 MBRA agents
- 10 simulations
- 1500x1500 environment
- 600 food, 20 viruses
"""

import time
import numpy as np
from tqdm import tqdm
from game.opencv_view import OCView, OCCamera
from game.model import Model
from game.entities import Player
from agent import RNNAgent, ModelBasedReflexAgent, FitnessWeights, Hyperparameters
from agario_simulation import AgarioSimulation


def run_rnn_vs_mbra_experiment(
    num_rnn: int = 5,
    num_mbra: int = 5,
    num_simulations: int = 10,
    bounds: int = 1500,
    food_count: int = 600,
    virus_count: int = 20,
    simulation_duration: float = 300.0,  # 5 minutes
    base_fps: int = 60,
    headless: bool = True
):
    """
    Run experiment comparing RNN agents vs MBRA agents.
    
    Returns:
        dict with RNN and MBRA fitness statistics
    """
    print("=" * 70)
    print("RNN vs MBRA Experiment")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - RNN agents: {num_rnn}")
    print(f"  - MBRA agents: {num_mbra}")
    print(f"  - Simulations: {num_simulations}")
    print(f"  - Map size: {bounds}x{bounds}")
    print(f"  - Food: {food_count}, Viruses: {virus_count}")
    print(f"  - Duration: {simulation_duration}s per simulation")
    print(f"  - Headless: {headless}")
    print("=" * 70)
    print()
    
    # Setup hyperparameters and fitness weights
    hyperparameters = Hyperparameters(
        hidden_layers=[72],
        output_size=4,
        run_interval=0.1,
        param_mutations={"weight": 2.0, "bias": 0.5},
        move_sensitivity=50.0,
        grid_width=12,
        grid_height=8,
        nodes_per_cell=3
    )
    fitness_weights = FitnessWeights(
        food=0.5,
        time_alive=100.0,
        cells_eaten=50.0,
        score=0.75,
        death=500.0
    )
    
    # Try to load trained RNN from file
    best_agent = RNNAgent.load_best_agent("gru_agent_snapshots_288i_72h_4o.pkl", hyperparameters, fitness_weights)

    # Storage for results
    all_rnn_fitnesses = []
    all_mbra_fitnesses = []
    
    num_frames = int(simulation_duration * base_fps)
    
    for sim_idx in range(num_simulations):
        print(f"\n{'='*70}")
        print(f"Simulation {sim_idx + 1}/{num_simulations}")
        print(f"{'='*70}")
        
        # Create players
        world_bounds = [bounds, bounds]
        
        # Create RNN agents and players
        rnn_agents = []
        rnn_players = []
        for i in range(num_rnn):
            agent = best_agent.copy()
            player = Player.make_random(f"RNN_{i}", world_bounds, color=(0, 255, 0))  # Green
            rnn_agents.append(agent)
            rnn_players.append(player)
        
        # Create MBRA agents and players
        mbra_agents = []
        mbra_players = []
        for i in range(num_mbra):
            agent = ModelBasedReflexAgent(
                run_interval=0.1,
                fitness_weights=fitness_weights,
                move_sensitivity=50.0,
                decay_factor=0.985,
                priority_threshold=0.05,
                max_memory_size=100,
                distance_weight_factor=500.0
            )
            player = Player.make_random(f"MBRA_{i}", world_bounds, color=(255, 0, 0))  # Red
            mbra_agents.append(agent)
            mbra_players.append(player)
        
        # Create model with all players
        all_players = rnn_players + mbra_players
        model = Model(all_players, bounds=world_bounds, chunk_size=bounds // 10)
        model.spawn_cells(food_count)
        model.spawn_viruses(virus_count)
        
        # Track previous target positions
        prev_target_pos = {}
        for player in all_players:
            prev_target_pos[player.nick] = player.center()
        
        # Create view for visualization (if not headless)
        view = None
        if not headless:
            view = OCView(900, 600, model, rnn_players[0] if rnn_players else mbra_players[0], debug=False)
        
        # Run simulation
        start_time = time.time()
        last_agent_run_frame = -100
        
        for frame in tqdm(range(num_frames), desc=f"Sim {sim_idx+1}", unit="frames"):
            # Maintain food and virus counts
            if model.num_viruses < virus_count:
                model.spawn_viruses(virus_count - model.num_viruses)
            if model.num_cells < food_count:
                model.spawn_cells(food_count - model.num_cells)
            
            run_agent = frame - last_agent_run_frame >= 0.1 * base_fps  # run_interval = 0.1
            
            if run_agent:
                # Run RNN agents
                for i, (agent, player) in enumerate(zip(rnn_agents, rnn_players)):
                    if not player.alive:
                        continue
                    player.ticks_alive += 1
                    
                    target_pos, split, eject = AgarioSimulation.rnn_tick(
                        900, 600, agent, player, model
                    )
                    
                    if split > 0:
                        model.split(player, target_pos)
                    if eject > 0:
                        model.shoot(player, target_pos)
                    model.update_velocity(player, target_pos)
                    prev_target_pos[player.nick] = target_pos
                
                # Run MBRA agents
                for i, (agent, player) in enumerate(zip(mbra_agents, mbra_players)):
                    if not player.alive:
                        continue
                    player.ticks_alive += 1
                    
                    target_pos, split, eject = AgarioSimulation.mbra_tick(
                        900, 600, agent, player, model, frame
                    )
                    
                    if split > 0:
                        model.split(player, target_pos)
                    if eject > 0:
                        model.shoot(player, target_pos)
                    model.update_velocity(player, target_pos)
                    prev_target_pos[player.nick] = target_pos
                
                last_agent_run_frame = frame
            else:
                # Just update velocities with previous targets
                for player in all_players:
                    if player.alive:
                        model.update_velocity(player, prev_target_pos[player.nick])
            
            # Update model
            model.update()
            
            # Redraw if not headless
            if view is not None:
                target_score = view.player.score()
                inv_scale = OCCamera.get_inverse_scale(target_score)
                view.camera.scale = 1.0 / inv_scale
                view.redraw(spectate_mode=True)
            
            # Check if only one type of agent survives
            rnn_alive = sum(1 for p in rnn_players if p.alive)
            mbra_alive = sum(1 for p in mbra_players if p.alive)
            if rnn_alive == 0 or mbra_alive == 0:
                if rnn_alive == 0 and mbra_alive == 0:
                    break  # All dead
                # Continue until time runs out or all of one type die
        
        # Calculate fitnesses
        sim_duration = time.time() - start_time
        
        rnn_fitnesses = []
        for i, (agent, player) in enumerate(zip(rnn_agents, rnn_players)):
            fitness = agent.calculate_fitness(
                player.num_food_eaten,
                player.ticks_alive / num_frames,
                player.num_players_eaten,
                player.score() + player.highest_score,
                int(not player.alive)
            )
            rnn_fitnesses.append(fitness)
        
        mbra_fitnesses = []
        for i, (agent, player) in enumerate(zip(mbra_agents, mbra_players)):
            fitness = agent.calculate_fitness(
                player.num_food_eaten,
                player.ticks_alive / num_frames,
                player.num_players_eaten,
                player.score() + player.highest_score,
                int(not player.alive)
            )
            mbra_fitnesses.append(fitness)
        
        all_rnn_fitnesses.extend(rnn_fitnesses)
        all_mbra_fitnesses.extend(mbra_fitnesses)
        
        # Print simulation results
        print(f"\nSimulation {sim_idx + 1} completed in {sim_duration:.2f}s")
        print(f"  RNN agents alive: {sum(1 for p in rnn_players if p.alive)}/{num_rnn}")
        print(f"  MBRA agents alive: {sum(1 for p in mbra_players if p.alive)}/{num_mbra}")
        print(f"  RNN avg fitness: {np.mean(rnn_fitnesses):.2f}")
        print(f"  MBRA avg fitness: {np.mean(mbra_fitnesses):.2f}")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    rnn_mean = np.mean(all_rnn_fitnesses)
    rnn_std = np.std(all_rnn_fitnesses)
    mbra_mean = np.mean(all_mbra_fitnesses)
    mbra_std = np.std(all_mbra_fitnesses)
    
    print(f"\nRNN Agents ({num_rnn} agents × {num_simulations} simulations = {len(all_rnn_fitnesses)} samples):")
    print(f"  Mean Fitness: {rnn_mean:.2f}")
    print(f"  Std Dev: {rnn_std:.2f}")
    print(f"  Min: {np.min(all_rnn_fitnesses):.2f}")
    print(f"  Max: {np.max(all_rnn_fitnesses):.2f}")
    
    print(f"\nMBRA Agents ({num_mbra} agents × {num_simulations} simulations = {len(all_mbra_fitnesses)} samples):")
    print(f"  Mean Fitness: {mbra_mean:.2f}")
    print(f"  Std Dev: {mbra_std:.2f}")
    print(f"  Min: {np.min(all_mbra_fitnesses):.2f}")
    print(f"  Max: {np.max(all_mbra_fitnesses):.2f}")
    
    print("\n" + "=" * 70)
    print("FOR PAPER ABSTRACT:")
    print("=" * 70)
    print(f"The RNNs achieved an average fitness score of {rnn_mean:.2f} (σ={rnn_std:.2f})")
    print(f"compared to the MBRAs average fitness of {mbra_mean:.2f} (σ={mbra_std:.2f}).")
    print("=" * 70)
    
    return {
        "rnn": {
            "fitnesses": all_rnn_fitnesses,
            "mean": rnn_mean,
            "std": rnn_std,
            "min": np.min(all_rnn_fitnesses),
            "max": np.max(all_rnn_fitnesses)
        },
        "mbra": {
            "fitnesses": all_mbra_fitnesses,
            "mean": mbra_mean,
            "std": mbra_std,
            "min": np.min(all_mbra_fitnesses),
            "max": np.max(all_mbra_fitnesses)
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RNN vs MBRA experiment")
    parser.add_argument('--rnn', type=int, default=5, help='Number of RNN agents')
    parser.add_argument('--mbra', type=int, default=5, help='Number of MBRA agents')
    parser.add_argument('--sims', type=int, default=10, help='Number of simulations')
    parser.add_argument('--duration', type=float, default=300.0, help='Simulation duration in seconds')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--quick', action='store_true', help='Quick test (2 sims, 60s each)')
    args = parser.parse_args()
    
    if args.quick:
        # Quick test mode
        results = run_rnn_vs_mbra_experiment(
            num_rnn=2,
            num_mbra=2,
            num_simulations=2,
            simulation_duration=60.0,
            headless=args.headless
        )
    else:
        results = run_rnn_vs_mbra_experiment(
            num_rnn=args.rnn,
            num_mbra=args.mbra,
            num_simulations=args.sims,
            simulation_duration=args.duration,
            headless=args.headless
        )
    
    # Save results to CSV files
    import csv
    
    # Summary CSV
    with open("experiment_results_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "RNN", "MBRA"])
        writer.writerow(["Mean Fitness", f"{results['rnn']['mean']:.2f}", f"{results['mbra']['mean']:.2f}"])
        writer.writerow(["Std Dev", f"{results['rnn']['std']:.2f}", f"{results['mbra']['std']:.2f}"])
        writer.writerow(["Min Fitness", f"{results['rnn']['min']:.2f}", f"{results['mbra']['min']:.2f}"])
        writer.writerow(["Max Fitness", f"{results['rnn']['max']:.2f}", f"{results['mbra']['max']:.2f}"])
    
    # Detailed CSV with all fitness values
    with open("experiment_results_detailed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample", "Agent_Type", "Fitness"])
        for i, fitness in enumerate(results["rnn"]["fitnesses"]):
            writer.writerow([i+1, "RNN", f"{fitness:.2f}"])
        for i, fitness in enumerate(results["mbra"]["fitnesses"]):
            writer.writerow([i+1, "MBRA", f"{fitness:.2f}"])
    
    print("\nResults saved to experiment_results_summary.csv and experiment_results_detailed.csv")


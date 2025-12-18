import argparse
import pickle
import csv
import time
import random
import os
import sys

import torch
import numpy as np

from agent import RNNAgent, Hyperparameters, FitnessWeights
import agario_simulation


def build_rnn_agents(num_rnns: int, snapshots_path: str, device=torch.device("cpu")):
    if num_rnns <= 0:
        return []
    if not os.path.exists(snapshots_path):
        print(f"Snapshots file not found: {snapshots_path}")
        sys.exit(1)
    with open(snapshots_path, "rb") as f:
        snapshots = pickle.load(f)
    # Hyperparameters / fitness weights must match training
    hyperparameters = Hyperparameters(hidden_layers=[64, 16],
                                      output_size=4,
                                      run_interval=0.1,
                                      param_mutations={"weight": 2.0, "bias": 0.5},
                                      move_sensitivity=50.0,
                                      grid_width=9,
                                      grid_height=6,
                                      nodes_per_cell=4)
    fitness_weights = FitnessWeights(food=0.5, time_alive=100.0, cells_eaten=50.0, score=0.75, death=500.0)

    agents = []
    # If fewer snapshots than requested, repeat the last available snapshot
    for i in range(num_rnns):
        idx = min(i, len(snapshots) - 1)
        a = RNNAgent(hyperparameters.copy(), fitness_weights, randomize_params=False, device=device)
        a.rnn.load_state_dict(snapshots[idx])
        agents.append(a)
    return agents


def run_experiments(args):
    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # write header
        header = ["run", "seed"] + [f"rnn_{i}" for i in range(args.rnns)]
        writer.writerow(header)

        for run_idx in range(args.num_runs):
            seed = args.seed if args.seed is not None else int(time.time()) + run_idx
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            rnn_agents = build_rnn_agents(args.rnns, args.snapshots)

            sim = agario_simulation.AgarioSimulation(view_width=args.width,
                                                    view_height=args.height,
                                                    bounds=args.bounds,
                                                    food_count=args.food,
                                                    virus_count=args.viruses)

            fitnesses = sim.run(rnn_agents, args.mbras, args.fps, args.duration, headless=args.headless)

            row = [run_idx, seed] + fitnesses
            writer.writerow(row)
            print(f"Run {run_idx} seed={seed} -> fitnesses={fitnesses}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run headless MBRA vs RNN simulations and save CSV results")
    parser.add_argument('--mbras', type=int, default=2, help='number of MBRA agents')
    parser.add_argument('--rnns', type=int, default=2, help='number of RNN agents')
    parser.add_argument('--snapshots', type=str, default='agent_snapshots.pkl', help='path to agent snapshots pickle')
    parser.add_argument('--num-runs', type=int, default=1, dest='num_runs', help='number of runs (seeds)')
    parser.add_argument('--seed', type=int, default=None, help='optional fixed random seed')
    parser.add_argument('--duration', type=float, default=240.0, help='simulation duration in seconds')
    parser.add_argument('--fps', type=int, default=60, help='base fps for simulation')
    parser.add_argument('--width', type=int, default=900)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--bounds', type=int, default=1500)
    parser.add_argument('--food', type=int, default=700)
    parser.add_argument('--viruses', type=int, default=20)
    parser.add_argument('--output', type=str, default='results.csv', help='CSV output file')
    parser.add_argument('--headless', action='store_true', help='run in headless mode')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiments(args)

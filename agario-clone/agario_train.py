import argparse
import pygame
import time
import threading
import json
import image_processing
from geometry_utils import GameObject, Vector

from game.view import View
from game.model import Model
from game.entities import Virus, PlayerCell


def game_loop(food_count: int, virus_count: int, model: Model):
    with open('cluster_settings.json', 'r') as f:
        cluster_settings = json.load(f)
        
    # Clustering for each cell type to reduce input size to our RNNs
    # The way the JSON settings are set up is stupid but it's what works best with object recognition
    cluster_by = {
        "virus": cluster_settings["virus"]["cluster_by"],
        "player": cluster_settings["unknown"]["variants"]["player"]["cluster_by"],
        "food": cluster_settings["unknown"]["variants"]["food"]["cluster_by"]
    }
    max_cluster_counts = {
        "virus": cluster_settings["virus"]["max_count"],
        "player": cluster_settings["unknown"]["variants"]["player"]["max_count"],
        "food": cluster_settings["unknown"]["variants"]["food"]["max_count"]
    }
    cluster_distances = {
        "virus": cluster_settings["virus"]["cluster_distance"],
        "player": cluster_settings["unknown"]["variants"]["player"]["cluster_distance"],
        "food": cluster_settings["unknown"]["variants"]["food"]["cluster_distance"]
    }

    while True:
        cells = model.cells
        
        virus_cells = []
        player_cells = []
        food_cells = []

        # Lists of objects we'll use as input to the RNNs
        virus_objs = []
        player_objs = []
        food_objs = []

        for cell in cells:
            pos = Vector(cell.pos[0], cell.pos[1])
            top_left = Vector(pos.x - cell.radius, pos.y + cell.radius)
            bottom_right = Vector(pos.x + cell.radius, pos.y - cell.radius)
            bounds = (top_left, bottom_right)
            
            if isinstance(cell, Virus):
                virus_cells.append(cell)
                virus_obj = GameObject("virus", pos, cell.area(), cell.perimeter(), 1.0, 1, bounds)
                image_processing.ImageProcessing.cluster_or_add(virus_obj, virus_objs, "virus", cluster_by["virus"], max_cluster_counts["virus"], cluster_distances["virus"])
            elif isinstance(cell, PlayerCell):
                player_cells.append(cell)
                player_obj = GameObject("player", pos, cell.area(), cell.perimeter(), 1.0, 1, bounds)
                image_processing.ImageProcessing.cluster_or_add(player_obj, virus_objs, "player", cluster_by["player"], max_cluster_counts["player"], cluster_distances["player"])
            else:
                food_cells.append(cell)
                food_obj = GameObject("food", pos, cell.area(), cell.perimeter(), 1.0, 1, bounds)
                image_processing.ImageProcessing.cluster_or_add(food_obj, virus_objs, "food", cluster_by["food"], max_cluster_counts["food"], cluster_distances["food"])

        # Maintain virus count
        if len(virus_cells) < virus_count:
            model.spawn_viruses(virus_count - len(virus_cells))

        # Maintain food count
        if len(food_cells) < food_count:
            model.spawn_cells(food_count - len(food_cells))

        time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser(description="Offline Agar.io (no networking)")
    parser.add_argument('-wt', '--width', dest='width', type=int, default=900, help='screen width')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=600, help='screen height')
    parser.add_argument('-b', '--bounds', dest='bounds', type=int, default=1000, help='half-size of world bounds (world is [-b,b] x [-b,b])')
    parser.add_argument('-c', '--cells', dest='cells', type=int, default=500, help='initial food cell count')
    parser.add_argument('-v', '--viruses', dest='viruses', type=int, default=15, help='initial virus count')
    parser.add_argument('-n', '--nick', dest='nick', type=str, default='Player', help='your nickname')
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption('Agar.io Offline')

    bounds = [args.bounds, args.bounds]

    # Create player and model
    player = Player.make_random(args.nick, bounds)
    model = Model([player], bounds=bounds)
    model.spawn_cells(args.cells)
    model.spawn_viruses(args.viruses)

    # Start game loop
    game_loop_thread = threading.Thread(target=game_loop, args=(args.cells, args.viruses, model), daemon=True)
    game_loop_thread.start()

    # Start view loop (handles input: mouse to move, W to shoot, SPACE to split)
    view = View(screen, model, player, debug=False)
    view.start()


if __name__ == '__main__':
    main()

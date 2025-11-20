import argparse
import pygame
import time
import threading

from game.view import View
from game.model import Model
from game.entities import Player
from game.entities import Virus


def game_loop(food_count: int, virus_count: int, model: Model):
    while True:
        cells = model.cells
        # Maintain food (non-virus) count
        food_cells = [c for c in cells if not isinstance(c, Virus)]
        if len(food_cells) < food_count:
            model.spawn_cells(food_count - len(food_cells))
        # Maintain virus count
        virus_cells = [c for c in cells if isinstance(c, Virus)]
        if len(virus_cells) < virus_count:
            model.spawn_viruses(virus_count - len(virus_cells))
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

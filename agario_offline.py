import argparse
import pygame
import time
import threading

from game.view import View
from game.new_model import Model
#from game.model import Model
from game.entities import Player


def game_loop(food_count: int, virus_count: int, model: Model):
    while True:
        # Maintain virus count
        if model.num_viruses < virus_count:
            model.spawn_viruses(virus_count - model.num_viruses)

        # Maintain food count
        if model.num_cells < food_count:
            model.spawn_cells(food_count - model.num_cells)
        time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser(description="Offline Agar.io (no networking)")
    parser.add_argument('-wt', '--width', dest='width', type=int, default=900, help='screen width')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=600, help='screen height')
    parser.add_argument('-b', '--bounds', dest='bounds', type=int, default=2000, help='half-size of world bounds (world is [-b,b] x [-b,b])')
    parser.add_argument('-f', '--food', dest='food', type=int, default=900, help='initial food cell count')
    parser.add_argument('-v', '--viruses', dest='viruses', type=int, default=25, help='initial virus count')
    parser.add_argument('-n', '--nick', dest='nick', type=str, default='Player', help='your nickname')
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption('Agar.io Offline')

    bounds = [args.bounds, args.bounds]

    # Create player and model
    players = [Player.make_random(f"Husk {i}", bounds) for i in range(5)]
    player = Player.make_random(args.nick, bounds)
    player.parts[0].radius = 100
    players.append(player)
    model = Model(players, bounds=bounds)
    model.spawn_cells(args.food)
    model.spawn_viruses(args.viruses)

    # Start game loop
    game_loop_thread = threading.Thread(target=game_loop, args=(args.food, args.viruses, model), daemon=True)
    game_loop_thread.start()

    # Start view loop (handles input: mouse to move, W to shoot, SPACE to split)
    view = View(screen, model, player, debug=False)
    view.start_human_game()


if __name__ == '__main__':
    main()

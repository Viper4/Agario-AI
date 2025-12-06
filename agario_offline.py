import argparse
import pygame
import time
import threading

from game.opencv_view import OCView
from game.view import View
from game.model import Model
from game.entities import Player


def main():
    parser = argparse.ArgumentParser(description="Offline Agar.io (no networking)")
    parser.add_argument('-wt', '--width', dest='width', type=int, default=900, help='screen width')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=600, help='screen height')
    parser.add_argument('-b', '--bounds', dest='bounds', type=int, default=1500, help='half-size of world bounds (world is [-b,b] x [-b,b])')
    parser.add_argument('-f', '--food', dest='food', type=int, default=800, help='initial food cell count')
    parser.add_argument('-v', '--viruses', dest='viruses', type=int, default=20, help='initial virus count')
    parser.add_argument('-n', '--nick', dest='nick', type=str, default='Player', help='your nickname')
    parser.add_argument('-oc', '--opencv', dest='opencv', type=bool, default=True, help='Whether to use OpenCV view')
    args = parser.parse_args()

    bounds = [args.bounds, args.bounds]

    # Create player and model
    players = []
    for i in range(5):
        new_player = Player.make_random(f"Husk {i}", bounds, 75)
        players.append(new_player)

    player = Player.make_random(args.nick, bounds, 100)
    players.append(player)

    model = Model(players, bounds=bounds, chunk_size=args.bounds // 10)
    model.spawn_cells(args.food)
    model.spawn_viruses(args.viruses)

    if args.opencv:
        view = OCView(900, 600, model, player)

        while True:
            # Maintain virus count
            if model.num_viruses < args.viruses:
                model.spawn_viruses(args.viruses - model.num_viruses)

            # Maintain food count
            if model.num_cells < args.food:
                model.spawn_cells(args.food - model.num_cells)

            for p in players:
                if p.alive and p != player:
                    model.split(p, p.center())  # Split all husks constantly

            view.redraw(spectate_mode=False)
            model.update()
            time.sleep(0.008333)  # 60 FPS
    else:
        # Start view loop (handles input: mouse to move, W to shoot, SPACE to split)
        pygame.init()
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption('Agar.io Offline')
        view = View(screen, model, player, debug=False)
        view.start_human_game(args.food, args.viruses)


if __name__ == '__main__':
    main()

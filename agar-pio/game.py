import pygame as pg
from pygame.locals import QUIT, KEYDOWN, K_w, K_SPACE, K_BACKSPACE, K_RETURN
import random
from settings import WIDTH, HEIGHT, TITLE, FPS, BACKGROUND
from math import sqrt, pi
from food import Food
from player import Player
from camera import Camera


class Game():
    def __init__(self):
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        pg.font.init()
        self.clock = pg.time.Clock()
        self.player = Player(self, WIDTH/2, HEIGHT/2)
        self.running = True
        self.playing = False
        self.food = []
        self.tick_count = 0
        self.camera = Camera()
        self.small_font = pg.font.SysFont('Comic Sans MS', 30)
        self.medium_font = pg.font.SysFont('Comic Sans MS', 60)
        self.big_font = pg.font.SysFont('Comic Sans MS', 120)

    def new(self):
        self.reset()
        self.players = [self.player]

    def reset(self):
        self.food = []
        self.player.reset()
        self.tick_count = 0

    def events(self):
        for event in pg.event.get():
            if event.type == QUIT:
                self.quit()
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    self.player.split()
                if event.key == K_w:
                    self.player.feed()

    def generate_food(self, amount: int):
        for _ in range(0, amount):
            temp = Food(self, random.choice(range(-WIDTH, WIDTH*2)),
                        random.choice(range(-HEIGHT, HEIGHT*2)))
            self.food.append(temp)

    def update(self):
        pg.display.update()

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.score_table()
        for food in self.food:
            food.draw(WIDTH / 2 - food.pos.x, HEIGHT / 2 - food.pos.y)
            food.update()
            for ball in self.player.balls:
                ball.eat(food)
        for ball in self.player.balls:
            ball.draw()

    def run(self):
        while self.playing:
            self.clock.tick(FPS)
            self.camera.zoom = 100/(self.player.radius)+0.3
            self.camera.centre(self.player)
            self.events()
            self.draw()
            if self.tick_count % 40 == 0:
                for ball in self.player.balls:
                    ball.loss()

            if self.tick_count % 1000 == 0:
                area_sum = self.player.area
                for ball in self.player.balls[1:]:
                    area_sum += ball.area
                    self.player.balls.remove(ball)
                self.player.set_radius(sqrt(area_sum / pi))

            # 18 000 == 5 min (5x60x60)
            if self.tick_count % 7200 == 0:
                self.generate_food(1000)

            self.tick_count += 1

            if self.tick_count % 18000 == 0:
                self.playing = False
                self.winner_screen()

            self.update()

    def score_table(self):
        table_rect = pg.Rect((WIDTH - 300, 0, 300, len(self.players * 30)))
        pg.draw.rect(self.screen, (211, 211, 211), table_rect)

        for player in self.players:
            text = f"{player.name} {round(player.area / 1000)}"
            text_surface = self.small_font.render(text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=table_rect.center)
            self.screen.blit(text_surface, text_rect)

    def winner_screen(self):
        areas = [player.area for player in self.players]
        winner = next(
            player.name for player in self.players if player.area == max(areas))

        winner_text = f"Winner is {winner} with {round(max(areas)/1000,2)} points"
        winner_text_surface = self.big_font.render(
            winner_text, True, (0, 0, 0))
        winnter_text_rect = winner_text_surface.get_rect(
            center=(int(WIDTH/2), int(HEIGHT/2 - HEIGHT/8)))

        press_text = f"Press Enter"
        press_text_surface = self.medium_font.render(
            press_text, True, (0, 0, 0))
        press_text_rect = press_text_surface.get_rect(
            center=(int(WIDTH/2), int(HEIGHT/2)))

        run = True
        while run:
            for event in pg.event.get():
                if event.type == QUIT:
                    self.quit()
                if event.type == KEYDOWN:
                    if event.key == K_RETURN:
                        run = False
                        self.start_screen()

            self.screen.blit(pg.transform.scale(
                BACKGROUND, (WIDTH, HEIGHT)), (0, 0))
            self.screen.blit(winner_text_surface, winnter_text_rect)
            self.screen.blit(press_text_surface, press_text_rect)
            pg.display.update()

    def start_screen(self):
        self.new()
        title = TITLE
        title_surface = self.big_font.render(title, True, (0, 0, 0))
        title_rect = title_surface.get_rect(
            center=(int(WIDTH/2), int(HEIGHT/2 - HEIGHT/4)))

        text = f"Press Enter"
        text_surface = self.medium_font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(int(WIDTH/2), int(HEIGHT/2)))

        name_surface = self.small_font.render("NAME: ", True, (0, 0, 0))

        input_rect = pg.Rect(int(WIDTH/2), int(HEIGHT/2), 200, 32)
        input_rect.center = (int(WIDTH/2), int(HEIGHT/2 - HEIGHT/8))
        input_text = self.player.name
        while not self.playing:
            for event in pg.event.get():
                if event.type == QUIT:
                    self.quit()
                if event.type == KEYDOWN:
                    if event.key == K_RETURN:
                        if len(input_text) > 0:
                            self.player.set_name(input_text)
                            self.playing = True
                    elif event.key == K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        if input_rect.w == 200:
                            input_text += event.unicode

            self.screen.blit(pg.transform.scale(
                BACKGROUND, (WIDTH, HEIGHT)), (0, 0))

            pg.draw.rect(self.screen, (0, 0, 0), input_rect, 2)

            input_text_surface = self.small_font.render(
                input_text, True, (255, 0, 0))

            self.screen.blit(input_text_surface,
                             (input_rect.x + 5, input_rect.y + 5))
            input_rect.w = max(200, input_text_surface.get_width() + 10)

            self.screen.blit(title_surface, title_rect)
            self.screen.blit(text_surface, text_rect)

            self.screen.blit(
                name_surface, (input_rect.x - 70, input_rect.y + 10))

            pg.display.flip()
            self.clock.tick(FPS)

    def quit(self):
        if self.playing:
            self.playing = False
        self.running = False
        pg.quit()
        exit()
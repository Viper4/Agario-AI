import pygame as pg
import random
from settings import WIDTH, HEIGHT, BASE_FOOD_RADIUS
from math import sqrt, pi


class Food:
    def __init__(self, game: object, x: int, y: int):
        self.game = game
        self.radius = BASE_FOOD_RADIUS
        self.area = int(pi * self.radius ** 2)
        self.color = random.choice([(125, 0, 0), (0, 0, 125), (0, 125, 0)])
        self.pos = pg.math.Vector2(x, y)
        self.speed = 5

    def draw(self, x: int, y: int):
        pg.draw.circle(self.game.screen, self.color,
                       (int(x*self.game.camera.zoom+self.game.camera.x), int(y*self.game.camera.zoom+self.game.camera.y)), int(self.radius*self.game.camera.zoom))
        
    def set_radius(self, radius):
        self.radius = radius
        self.area = int(pi * self.radius ** 2)

    def get_radius(self):
        return self.radius

    def set_color(self, color: tuple):
        self.color = color

    def get_color(self):
        return self.color

    def move(self):
        mouse = pg.mouse.get_pos()
        vec = pg.math.Vector2(mouse[0] - WIDTH/2, mouse[1] - HEIGHT/2)
        if vec.length_squared() > 0:
            vec = vec.normalize()
        vec.scale_to_length(self.speed)
        self.pos += vec

    def update(self):
        self.move()
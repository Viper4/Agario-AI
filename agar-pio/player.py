import pygame as pg
import random
from settings import WIDTH, HEIGHT, BASE_PLAYER_RADIUS, BASE_FOOD_RADIUS, SPLIT_LIMIT
from math import sqrt, pi
from food import Food


class Player(Food):
    def __init__(self, game: object, x: int, y: int):
        super().__init__(game, x, y)
        self.radius = BASE_PLAYER_RADIUS
        self.area = int(pi * self.radius ** 2)
        self.color = (0, 0, 0)
        self.pos = pg.math.Vector2(x, y)
        self.name = ""
        self.balls = [self]
        self.off_set = pg.math.Vector2(0, 0)

    def set_name(self, name: str):
        self.name = name

    def set_offset(self, x, y):
        self.off_set = pg.math.Vector2((-x, -y))

    def eat(self, food):
        temp_pos = pg.math.Vector2(0, 0) - self.off_set
        if temp_pos.distance_to(food.pos) <= self.radius/2:
            if self.radius > food.radius:
                area_sum = self.area + food.area
                self.radius = sqrt(area_sum / pi)
                self.area = int(pi * self.radius ** 2)
                self.radius += 0.5
                self.area = int(pi * self.radius ** 2)
                self.game.food.remove(food)

    def draw(self):
        pg.draw.circle(self.game.screen, self.color, (int(
            self.pos.x*self.game.camera.zoom+self.game.camera.x), int(self.pos.y*self.game.camera.zoom+self.game.camera.y)), int(self.radius/2*self.game.camera.zoom))
      
    def loss(self):
        if self.radius > BASE_PLAYER_RADIUS:
            self.area -= self.speed
            self.radius = sqrt(self.area / pi)

    def split(self):
        mouse = pg.mouse.get_pos()
        vec = pg.math.Vector2(WIDTH/2 - mouse[0], HEIGHT / 2 - mouse[1])
        self.set_radius(self.radius / 2)
        new_ball_x = self.pos.x - vec.x
        new_ball_y = self.pos.y - vec.y
        new_ball = Player(self.game, new_ball_x, new_ball_y)
        new_ball.set_radius(self.radius)
        new_ball.set_offset(vec.x, vec.y)
        self.balls.append(new_ball)

    def feed(self):
        if self.radius > BASE_PLAYER_RADIUS:
            mouse = pg.mouse.get_pos()
            vec = pg.math.Vector2(mouse[0], mouse[1])
            new_food_x = self.pos.x - vec.x
            new_food_y = self.pos.y - vec.y
            new_food = Food(self.game, new_food_x, new_food_y)
            self.set_radius(self.radius - BASE_FOOD_RADIUS)
            self.game.food.append(new_food)

    def reset(self):
        self.set_radius(BASE_PLAYER_RADIUS)
        self.balls = [self]

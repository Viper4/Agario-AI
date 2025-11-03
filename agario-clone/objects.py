import math
from settings import *
import random


class Ball:
    def __init__(self, radius: float, x: float, y: float, color: tuple[3]):
        self.radius = radius
        self.mass = 0.0
        self.area = 0.0
        self.recalculate_area_and_mass()
        self.position = pg.Vector2(x, y)
        self.color = color
        self.velocity = pg.Vector2(0, 0)
        self.eject_radius = self.calculate_radius(EJECT_MASS)

    def draw(self, screen: pg.Surface, relative_to: pg.Vector2):
        pg.draw.circle(screen, self.color, (int(self.position.x - relative_to.x), int(self.position.y - relative_to.y)), int(self.radius))

    def update(self, time_delta: float):
        # Move the ball
        self.position += self.velocity * time_delta
        self.velocity *= SPEED_DECAY
        if self.velocity.x < 0.1 and self.velocity.y < 0.1:
            self.velocity = pg.Vector2(0, 0)

    def recalculate_area_and_mass(self):
        self.area = math.pi * self.radius * self.radius
        self.mass = (self.radius * self.radius) / 100

    @staticmethod
    def calculate_radius(mass: float):
        return math.sqrt(mass) * 10


class Player(Ball):
    def __init__(self, radius: float, x: float, y: float, color: tuple[3], name: str):
        super().__init__(radius, x, y, color)
        self.id = random.randint(0, 1000000)
        self.name = name
        self.balls = [self]
        self.ball_index = 0  # 0 means root
        self.time_since_split = BASE_MERGE_DELAY

    def update(self, time_delta: float):
        super().update(time_delta)
        self.mass *= MASS_LOSS_RATE * time_delta
        self.time_since_split += time_delta

    def eat(self, other: Ball):
        if other.radius < self.radius * EAT_PROPORTION:
            self.radius += other.radius
            self.recalculate_area_and_mass()

    def split(self, new_radius: float, direction: pg.Vector2):
        perimeter_pos = self.position + direction * (self.radius + new_radius + 0.1)
        new_ball = Player(new_radius, perimeter_pos.x, perimeter_pos.y, self.color, self.name)
        new_ball.time_since_split = 0.0
        new_ball.id = self.id
        new_ball.mass = self.mass * 0.5
        new_ball.recalculate_area_and_mass()
        new_ball.velocity = direction * SPLIT_SPEED
        self.balls.append(new_ball)

    def split_all(self, direction: pg.Vector2):
        for ball in self.balls:
            if len(self.balls) >= SPLIT_LIMIT:
                break
            half_mass = ball.mass * 0.5
            if half_mass > MIN_PLAYER_MASS:  # Have enough mass to split
                # Update this ball
                ball.time_since_split = 0.0
                new_radius = self.calculate_radius(half_mass)
                ball.radius = new_radius
                ball.recalculate_area_and_mass()
                
                # Create newly split ball
                self.split(new_radius, direction)

    def eject(self, direction: pg.Vector2):
        new_balls = []
        for ball in self.balls:
            if ball.mass - EJECT_MASS > MIN_PLAYER_MASS:
                # Update the ball's mass
                ball.mass -= EJECT_MASS
                ball.radius = self.calculate_radius(ball.mass)

                # Create ejected ball
                perimeter_pos = ball.position + direction * (ball.radius + self.eject_radius + 0.1)  # Spawn the ejected ball at the edge
                ejected_ball = Ball(self.eject_radius, perimeter_pos.x, perimeter_pos.y, self.color)
                ejected_ball.velocity = direction * EJECT_SPEED
                new_balls.append(ejected_ball)
        return new_balls

    def merge(self):
        merge_delay = BASE_MERGE_DELAY + MERGE_DELAY_INCREMENT * self.mass
        if self.time_since_split > merge_delay:
            pass

    def virus_hit(self):
        if len(self.balls) >= SPLIT_LIMIT:
            # Add full virus mass to the hit ball
            self.mass += VIRUS_EAT_MASS
            self.radius = self.calculate_radius(self.mass)
        else:
            # Add less mass and explode into smaller balls
            self.mass += VIRUS_EAT_MASS / 10

            half_mass = self.mass * 0.5
            if half_mass < MIN_PLAYER_MASS:
                return
            # Update this ball to half mass
            self.mass = half_mass
            self.radius = self.calculate_radius(self.mass)
            self.time_since_split = 0.0

            # Create new exploded balls with remaining half mass
            num_splits = SPLIT_LIMIT - len(self.balls)
            split_mass = half_mass / num_splits
            if split_mass < MIN_PLAYER_MASS:
                return
            split_radius = self.calculate_radius(half_mass / num_splits)
            for i in range(num_splits):
                random_direction = pg.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
                random_direction.normalize()
                self.split(split_radius, random_direction)

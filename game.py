import time


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Object:
    def __init__(self, name: str, point: Point):
        self.name = name
        self.position = point


class Game:
    def __init__(self):
        self.objects = []
        self.player = Object("player", Point(0, 0))
        self.start_time = 0.0

    @staticmethod
    def get_distance(p1: Point, p2: Point):
        """
        Calculates linear distance between point 1 and point 2
        :param p1: Point object
        :param p2: Point object
        :return: float
        """
        return ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5

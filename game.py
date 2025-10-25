import math


class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def magnitude(self):
        """
        Calculates the magnitude or length of this Vector
        :return: magnitude as float
        """
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class GameObject:
    def __init__(self, label: str, pos: Vector, area: float, perimeter: float, circularity: float, density: int, bounding_box: tuple[Vector, Vector]):
        self.label = label
        self.pos = pos
        self.area = area
        self.perimeter = perimeter
        self.circularity = circularity
        self.density = density
        self.bounding_box = bounding_box  # First pair is top left corner, second pair is bottom right: ((x1, y1), (x2, y2))

    def extend_bounds(self, other_bounding_box: tuple[Vector, Vector]):
        """
        Expands this object's bounding box to include the given bounding box
        :param other_bounding_box:
        """

        self.bounding_box = (Vector(min(self.bounding_box[0].x, other_bounding_box[0].x), min(self.bounding_box[0].y, other_bounding_box[0].y)),
                             Vector(max(self.bounding_box[1].x, other_bounding_box[1].x), max(self.bounding_box[1].y, other_bounding_box[1].y)))

    def copy(self):
        return GameObject(self.label, self.pos, self.area, self.perimeter, self.circularity, self.density, self.bounding_box)

    def __str__(self):
        return f"{self.label}: ({self.pos.x}, {self.pos.y})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (self.label == other.label and self.pos == other.pos and self.area == other.area and
                self.perimeter == other.perimeter and self.circularity == other.circularity and
                self.density == other.density)


def get_distance(p1: Vector, p2: Vector):
    """
    Calculates linear distance between Vector 1 and Vector 2
    :param p1:
    :param p2:
    :return: float distance
    """
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5


def get_normalized_direction(p1: Vector, p2: Vector):
    """
    Calculates normalized direction from point 1 to point 2
    :param p1: first point
    :param p2: second point
    :return: direction vector
    """
    direction = Vector(p2.x - p1.x, p2.y - p1.y)
    magnitude = direction.magnitude()
    return Vector(direction.x / magnitude, direction.y / magnitude)

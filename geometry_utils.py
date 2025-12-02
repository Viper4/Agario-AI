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

    def normalize(self):
        """
        Normalizes this vector to a length of 1
        :return: Vector
        """
        magnitude = self.magnitude()
        return Vector(self.x / magnitude, self.y / magnitude)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Vector(self.x * other.x, self.y * other.y)

    def __truediv__(self, other):
        return Vector(self.x / other.x, self.y / other.y)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class GameObject:
    def __init__(self, label: str, pos: Vector, area: float, perimeter: float, circularity: float, count: int, bounding_box: tuple[Vector, Vector]):
        self.label = label
        self.pos = pos
        self.area = area
        self.perimeter = perimeter
        self.circularity = circularity
        self.count = count
        self.bounding_box = bounding_box  # First vector is top left corner, second vector is bottom right: ((x1, y1), (x2, y2))

    def extend_bounds(self, other_bounding_box: tuple[Vector, Vector]):
        """
        Expands this object's bounding box to include the given bounding box
        :param other_bounding_box:
        """
        self.bounding_box = (Vector(min(self.bounding_box[0].x, other_bounding_box[0].x), min(self.bounding_box[0].y, other_bounding_box[0].y)),
                             Vector(max(self.bounding_box[1].x, other_bounding_box[1].x), max(self.bounding_box[1].y, other_bounding_box[1].y)))

    def linear_bounds_distance(self, other_bounding_box: tuple[Vector, Vector]):
        """
        Calculate the distance between the closest edges of this bounding box and the given bounding box
        :param other_bounding_box: (Vector, Vector)
        :return: tuple of (dx, dy)
        """
        x1, y1 = self.bounding_box[0].x, self.bounding_box[0].y  # top left corner
        x2, y2 = self.bounding_box[1].x, self.bounding_box[1].y  # bottom right corner
        x1b, y1b = other_bounding_box[0].x, other_bounding_box[0].y  # top left corner
        x2b, y2b = other_bounding_box[1].x, other_bounding_box[1].y  # bottom right corner

        left = x2b < x1  # other is to the left
        right = x1b > x2  # other is to the right
        top = y2b > y1  # other is above
        bottom = y1b < y2  # other is below

        dx = 0.0
        dy = 0.0
        if left:
            dx = x1 - x2b
        elif right:
            dx = x1b - x2
        if top:
            dy = y1b - y2
        elif bottom:
            dy = y1 - y2b

        return dx, dy

    def copy(self):
        return GameObject(self.label, self.pos, self.area, self.perimeter, self.circularity, self.count, self.bounding_box)

    def check_visible(self, bounds: tuple[Vector, Vector]):
        """
        Checks whether this object is visible within the given bounds
        :param bounds: (Vector, Vector) where first vector is top left corner and second vector is bottom right corner
        :return: bool
        """
        to_right = self.bounding_box[1].x > bounds[0].x  # Bottom right corner is to the right of left edge of view
        to_left = self.bounding_box[0].x < bounds[1].x  # Top left corner is to the left of right edge of view

        below = self.bounding_box[1].y < bounds[0].y  # Bottom right corner is below the top edge of view
        above = self.bounding_box[0].y > bounds[1].y  # Top left corner is above the bottom edge of view

        return to_right and to_left and below and above

    def __str__(self):
        return f"{self.label}: ({self.pos.x}, {self.pos.y})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (self.label == other.label and self.pos == other.pos and self.area == other.area and
                self.perimeter == other.perimeter and self.circularity == other.circularity and
                self.count == other.count)


def get_distance(p1: Vector, p2: Vector):
    """
    Calculates linear distance between Vector 1 and Vector 2
    :param p1:
    :param p2:
    :return: float distance
    """
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5


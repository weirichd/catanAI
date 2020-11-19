from itertools import product

import numpy as np
from enum import Enum

# Constants

S = np.sqrt(3) / 2

HEXAGON_COORDS = np.array([[0, -S, -S, 0, S, S], [1, 0.5, -0.5, -1, -0.5, 0.5]]).T

HEXAGON_CENTERS = np.array(
    [
        # 1st row
        [-2 * S, -3],
        [-0, -3],
        [2 * S, -3],
        # 2nd row
        [-3 * S, -1.5],
        [-S, -1.5],
        [S, -1.5],
        [3 * S, -1.5],
        # 3rd row
        [-4 * S, 0],
        [-2 * S, 0],
        [0, 0],
        [2 * S, 0],
        [4 * S, 0],
        # 4th row
        [-3 * S, 1.5],
        [-S, 1.5],
        [S, 1.5],
        [3 * S, 1.5],
        # 5th row
        [-2 * S, 3],
        [-0, 3],
        [2 * S, 3],
    ]
)

VERTICES = np.array(
    [
        # 1st Row
        [-3 * S, -3.5],  # Harbor 0
        [-2 * S, -4],  # Harbor 0
        [-S, -3.5],
        [0, -4],  # Harbor 1
        [S, -3.5],  # Harbor 1
        [2 * S, -4],
        [3 * S, -3.5],
        # 2nd Row
        [-4 * S, -2],  # Harbor 8
        [-3 * S, -2.5],
        [-2 * S, -2],
        [-S, -2.5],
        [0, -2],
        [S, -2.5],
        [2 * S, -2],
        [3 * S, -2.5],  # Harbor 2
        [4 * S, -2],  # Harbor 2
        # 3rd Row
        [-5 * S, -0.5],
        [-4 * S, -1],  # Harbor 8
        [-3 * S, -0.5],
        [-2 * S, -1],
        [-S, -0.5],
        [0, -1],
        [S, -0.5],
        [2 * S, -1],
        [3 * S, -0.5],
        [4 * S, -1],
        [5 * S, -0.5],  # Harbor 3
        # 4th Row
        [-5 * S, 0.5],
        [-4 * S, 1],  # Harbor 7
        [-3 * S, 0.5],
        [-2 * S, 1],
        [-S, 0.5],
        [0, 1],
        [S, 0.5],
        [2 * S, 1],
        [3 * S, 0.5],
        [4 * S, 1],
        [5 * S, 0.5],  # Harbor 3
        # 5th Row
        [-4 * S, 2],  # Harbor 7
        [-3 * S, 2.5],
        [-2 * S, 2],
        [-S, 2.5],
        [0, 2],
        [S, 2.5],
        [2 * S, 2],
        [3 * S, 2.5],  # Harbor 4
        [4 * S, 2],  # Harbor 4
        # 6th Row
        [-3 * S, 3.5],  # Harbor 6
        [-2 * S, 4],  # Harbor 6
        [-S, 3.5],
        [0, 4],  # Harbor 5
        [S, 3.5],  # Harbor 5
        [2 * S, 4],
        [3 * S, 3.5],
    ]
)

EDGES = [
    np.stack([v, w]) for v, w in product(VERTICES, VERTICES)
    if np.isclose(np.linalg.norm(v - w), 1.0) and ((v[0] < w[0]) or (v[0] == w[0] and v[1] < w[1]))
]

# Indices into the VERTICES array for the harbors
HARBOR_INDS = [
    (0, 1),
    (3, 4),
    (14, 15),
    (26, 37),
    (45, 46),
    (50, 51),
    (47, 48),
    (28, 38),
    (7, 17),
]

#
HARBOR_COORDS = np.array([
    # Harbor 0
    [-3 * S, -4.5],
    [-3 * S, -3.5],
    [-2 * S, -4],
    # Harbor 1
    [S, -4.5],
    [0, -4],
    [S, -3.5],
    # Harbor 2
    [4 * S, -3],
    [3 * S, -2.5],
    [4 * S, -2],
    # Harbor 3
    [6 * S, 0],
    [5 * S, -0.5],
    [5 * S, 0.5],
    # Harbor 4
    [4 * S, 3],
    [3 * S, 2.5],
    [4 * S, 2],
    # Harbor 5
    [S, 4.5],
    [0, 4],
    [S, 3.5],
    # Harbor 6
    [-3 * S, 4.5],
    [-3 * S, 3.5],
    [-2 * S, 4],
    # Harbor 7
    [-5 * S, 1.5],
    [-4 * S, 1],
    [-4 * S, 2],
    # Harbor 8
    [-5 * S, -1.5],
    [-4 * S, -2],
    [-4 * S, -1],
]
)


def points_to_str_list(coords):
    """
    Accept a rank 2 array and return a list of coordinates in a string for SVG polygon.
    """
    result = []

    for x_, y_ in coords:
        result.append(str(x_))
        result.append(str(y_))

    return " ".join(result)


class Tile(Enum):
    FOREST = "Forest", "Wood", "forestgreen", np.array([1, 0, 0, 0, 0, 0])
    PASTURE = "Pasture", "Sheep", "yellowgreen", np.array([0, 1, 0, 0, 0, 0])
    FIELD = "Field", "Wheat", "gold", np.array([0, 0, 1, 0, 0, 0])
    HILL = "Hill", "Brick", "firebrick", np.array([0, 0, 0, 1, 0, 0])
    MOUNTAIN = "Mountain", "Ore", "gray", np.array([0, 0, 0, 0, 1, 0])
    DESERT = "Desert", None, "sandybrown", np.array([0, 0, 0, 0, 0, 1])

    def __init__(self, terrain, resource, color, array):
        self.terrain = terrain
        self.resource = resource
        self.color = color
        self.array = array


class Harbor(Enum):
    WOOD = "Wood 2:1", np.array([1, 0, 0, 0, 0, 0])
    SHEEP = "Sheep 2:1", np.array([0, 1, 0, 0, 0, 0])
    WHEAT = "Wheat 2:1", np.array([0, 0, 1, 0, 0, 0])
    BRICK = "Brick 2:1", np.array([0, 0, 0, 1, 0, 0])
    ORE = "Ore 2:1", np.array([0, 0, 0, 0, 1, 0])
    THREE_FOR_ONE = "Any 3:1", np.array([0, 0, 0, 0, 0, 1])

    def __init__(self, type, array):
        self.type = type
        self.array = array


class Board:
    def __init__(self, random_seed=None):
        tiles = (
            [Tile.FOREST] * 4
            + [Tile.PASTURE] * 4
            + [Tile.FIELD] * 4
            + [Tile.HILL] * 3
            + [Tile.MOUNTAIN] * 3
            + [Tile.DESERT]
        )
        rand = np.random.RandomState(seed=random_seed)

        self.tiles = rand.choice(tiles, len(tiles), replace=False)

        numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        numbers = rand.choice(numbers, len(numbers), replace=False)
        self.numbers = np.insert(numbers, np.where(self.tiles == Tile.DESERT)[0], 0)

        harbors = [Harbor.WOOD, Harbor.SHEEP, Harbor.WHEAT, Harbor.BRICK, Harbor.ORE] + [Harbor.THREE_FOR_ONE] * 4
        self.harbors = rand.choice(harbors, len(harbors), replace=False)

    @property
    def svg(self):
        svg = str()

        for c, t, n in zip(HEXAGON_CENTERS, self.tiles, self.numbers):
            points = points_to_str_list((c + HEXAGON_COORDS) * 50 + 300)
            svg += (
                "<polygon "
                f'points="{points}" '
                'stroke="black" '
                f'fill="{t.color}" '
                'stroke="red" '
                'stroke-width="2"'
                "/>"
            )
            svg += (
                f"<text "
                f'x="{c[0] * 50 + 300}" '
                f'y="{c[1] * 50 + 290}" '
                'text-anchor="middle"'
                f">{t.terrain}</text>"
            )
            if n != 0:
                num_color = "black" if n not in [6, 8] else "red"
                svg += (
                    "<circle "
                    f'cx="{c[0] * 50 + 300}" '
                    f'cy="{c[1] * 50 + 320}" '
                    'stroke="black" '
                    'fill="white" '
                    'r="15" '
                    'stroke-width="2"'
                    "/>"
                )
                svg += (
                    "<text "
                    f'x="{c[0] * 50 + 300}" '
                    f'y="{c[1] * 50 + 325}" '
                    'text-anchor="middle" '
                    f'fill="{num_color}'
                    f'">{n}</text>'
                )

        for i, n in enumerate(self.harbors):
            points = HARBOR_COORDS[3*i:3*i+3] * 50 + 300

            points_str = points_to_str_list(points)

            svg += (
                "<polygon "
                f'points="{points_str}" '
                'stroke="black" '
                'fill="white" '
                'stroke="red" '
                'stroke-width="2"'
                "/>"
            )

            svg += (
                "<text "
                f'x="{points[0, 0]}" '
                f'y="{points[0, 1]}" '
                'text-anchor="middle" '
                f'">{n.type}</text>'
            )

        return svg

    @property
    def tiles_tensor(self):
        return np.stack([t.array for t in self.tiles])

    @property
    def harbor_tensor(self):
        return np.stack([h.array for h in self.harbors])

    @property
    def dots(self):
        n_dots = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        return np.array([n_dots[_] for _ in self.numbers])

    def _repr_svg_(self):
        svg = '<svg width="600" height="600">'
        svg += '<rect x="0" y="0" width="600" height="600" fill="deepskyblue"/>'

        svg += self.svg

        svg += "</svg>"

        return svg

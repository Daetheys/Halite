from engine.player import ShipMove,ShipyardMove
import copy
import numpy as np

_action_dict = {ShipyardMove.HOLD:[],
                ShipyardMove.SPAWN:[],
                ShipMove.LEFT:[],
                ShipMove.DOWN:[],
                ShipMove.RIGHT:[],
                ShipMove.UP:[],
                ShipMove.CONVERT:[],
                ShipMove.COLLECT:[]}

def actions_dict():
    return copy.deepcopy(_action_dict)

ship_actions = [ShipMove.LEFT,ShipMove.DOWN,ShipMove.RIGHT,ShipMove.UP,ShipMove.CONVERT,ShipMove.COLLECT]

ship_move_actions = [ShipMove.LEFT,ShipMove.DOWN,ShipMove.RIGHT,ShipMove.UP]

shipyard_actions = [ShipyardMove.HOLD,ShipyardMove.SPAWN]

def build_halite():
    starting_halite = 24000
    
    randint = np.random.randint
    size = 21
     # Distribute Halite evenly into quartiles.
    half = int(np.ceil(21/2))
    grid = [[0] * half for _ in range(half)]

    # Randomly place a few halite "seeds".
    for i in range(half):
        # random distribution across entire quartile
        grid[randint(0, half - 1)][randint(0, half - 1)] = i ** 2

        # as well as a particular distribution weighted toward the center of the map
        grid[randint(half // 2, half - 1)][randint(half // 2, half - 1)] = i ** 2

    # Spread the seeds radially.
    radius_grid = copy.deepcopy(grid)
    for r in range(half):
        for c in range(half):
            value = grid[r][c]
            if value == 0:
                continue

            # keep initial seed values, but constrain radius of clusters
            radius = min(round((value / half) ** 0.5), 1)
            for r2 in range(r - radius + 1, r + radius):
                for c2 in range(c - radius + 1, c + radius):
                    if 0 <= r2 < half and 0 <= c2 < half:
                        distance = (abs(r2 - r) ** 2 + abs(c2 - c) ** 2) ** 0.5
                        radius_grid[r2][c2] += int(value / max(1, distance) ** distance)

    # add some random sprouts of halite
    radius_grid = np.asarray(radius_grid)
    add_grid = np.random.gumbel(0, 300.0, size=(half, half)).astype(int)
    sparse_radius_grid = np.random.binomial(1, 0.5, size=(half, half))
    add_grid = np.clip(add_grid, 0, a_max=None) * sparse_radius_grid
    radius_grid += add_grid

    # add another set of random locations to the center corner
    corner_grid = np.random.gumbel(0, 500.0, size=(half // 4, half // 4)).astype(int)
    corner_grid = np.clip(corner_grid, 0, a_max=None)
    radius_grid[half - (half // 4):, half - (half // 4):] += corner_grid

    # Normalize the available halite against the defined configuration starting halite.
    total = sum([sum(row) for row in radius_grid])
    halite = [0] * (size ** 2)
    for r, row in enumerate(radius_grid):
        for c, val in enumerate(row):
            val = int(val * starting_halite / total / 4)
            halite[size * r + c] = val
            halite[size * r + (size - c - 1)] = val
            halite[size * (size - 1) - (size * r) + c] = val
            halite[size * (size - 1) - (size * r) + (size - c - 1)] = val
    return np.reshape(halite,(21,21))

"""
Zdefinuj problem optymalizacyjny jako minimalizacje dlugości łamanej złożonej z 20 odcinków, przy ograni-
czeniach wynikających z unikania kolizji z 3 kolistymi przeszkodami o różnych promieniach.
"""

import numpy as np

from path_optimizer import PathOptimizer
from visualizer import Visualizer


SPACE = {
    "x": (0.0, 20.0),
    "y": (0.0, 10.0),
}

FIRST_POINT = (SPACE["x"][0], SPACE["y"][0])
LAST_POINT = (SPACE["x"][1], SPACE["y"][1])
N_SEGMENTS = 20


def random_circles(n, space):
    circles = []
    for _ in range(n):
        x = np.random.uniform(space["x"][0], space["x"][1])
        y = np.random.uniform(space["y"][0], space["y"][1])
        r = np.random.uniform(1.0, 3.0)
        circles.append({"x": x, "y": y, "r": r})
    return circles


def main():
    circles = random_circles(3, SPACE)

    opt = PathOptimizer(
        space=SPACE,
        circles=circles,
        n_segments=N_SEGMENTS,
        first_point=FIRST_POINT,
        last_point=LAST_POINT,
    )
    viz = Visualizer(
        space=SPACE,
        circles=circles,
    )

    optimized_points = opt.solve(maxiter=1000)

    initial_guess_points = np.vstack(
        (FIRST_POINT, opt._initial_guess_points, LAST_POINT)
    )
    points = np.vstack((FIRST_POINT, optimized_points, LAST_POINT))

    viz.draw(initial_guess_points)
    viz.draw(points)


if __name__ == "__main__":
    main()

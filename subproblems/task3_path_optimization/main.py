"""
Zdefinuj problem optymalizacyjny jako minimalizacje dlugości łamanej złożonej z 20 odcinków, przy ograni-
czeniach wynikających z unikania kolizji z 3 kolistymi przeszkodami o różnych promieniach.
"""

import numpy as np

from path_optimizer import PathOptimizer
from visualizer import Visualizer

CIRCLES = (
    {
        "x": 3,
        "y": 5,
        "r": 1,
    },
    {
        "x": 7,
        "y": 4,
        "r": 2,
    },
    {
        "x": 15,
        "y": 8,
        "r": 2,
    },
)

SPACE = {
    "x": (0.0, 20.0),
    "y": (0.0, 10.0),
}

FIRST_POINT = (SPACE["x"][0], SPACE["y"][0])
LAST_POINT = (SPACE["x"][1], SPACE["y"][1])
N_SEGMENTS = 20


def main():
    opt = PathOptimizer(
        space=SPACE,
        circles=CIRCLES,
        n_segments=N_SEGMENTS,
        first_point=FIRST_POINT,
        last_point=LAST_POINT,
    )
    viz = Visualizer(
        space=SPACE,
        circles=CIRCLES,
    )

    optimized_points = opt.solve(maxiter=1000)
    
    initial_guess_points = np.vstack((FIRST_POINT, opt._initial_guess_points, LAST_POINT))
    points = np.vstack((FIRST_POINT, optimized_points, LAST_POINT))

    viz.draw(initial_guess_points)
    viz.draw(points)

if __name__ == "__main__":
    main()

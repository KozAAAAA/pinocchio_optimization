import numpy as np
from scipy.optimize import minimize


class PathOptimizer:
    def __init__(self, space, circles, n_segments, first_point, last_point):
        self._circles = circles
        self._n_optimizable_points = n_segments - 3
        self._first_point = first_point
        self._last_point = last_point

        bound_x = (space["x"][0], space["x"][1])
        bound_y = (space["y"][0], space["y"][1])

        self._initial_guess_points = np.array(
            [
                [
                    np.random.uniform(bound_x[0], bound_x[1]),
                    np.random.uniform(bound_y[0], bound_y[1]),
                ]
                for _ in range(self._n_optimizable_points)
            ]
        )

        self._bounds = (bound_x, bound_y) * self._n_optimizable_points
        self._constraints = ()

    def _objective(self, x):
        optimized_points = x.reshape(-1, 2)
        points = np.vstack((self._first_point, optimized_points, self._last_point))
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

    def _equal_length_constraint(self, x):
        pass

    def solve(self):
        res = minimize(
            self._objective,
            self._initial_guess_points.flatten(),
            constraints=self._constraints,
            bounds=self._bounds,
        )
        optimized_points = res.x.reshape(-1, 2)
        return optimized_points

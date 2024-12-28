import numpy as np
from scipy.optimize import minimize


class PathOptimizer:
    def __init__(self, space, circles, n_segments, first_point, last_point):
        self._iter = 0
        self._n_optimizable_points = n_segments - 1
        self._first_point = first_point
        self._last_point = last_point

        bound_x = (space["x"][0], space["x"][1])
        bound_y = (space["y"][0], space["y"][1])

        self._initial_guess_points = np.linspace(
            start=first_point,
            stop=last_point,
            num=self._n_optimizable_points + 2,
        )[1:-1]

        self._bounds = (bound_x, bound_y) * self._n_optimizable_points
        self._constraints = [
            {
                "type": "ineq",
                "fun": self._circle_constraint,
                "args": (
                    circle,
                    5,
                ),
            }
            for circle in circles
        ] + [
            {
                "type": "eq",
                "fun": self._equal_distances_constraint,
            }
        ]

    def _objective(self, x):
        """
        Calculate distance between individual points and sum them up.
        diff - calculate the difference between the points
        norm - calculate the euclidean distance between the points
        sum - sum up all the distances
        """
        optimized_points = x.reshape(-1, 2)
        points = np.vstack((self._first_point, optimized_points, self._last_point))
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

    def _circle_constraint(self, x, circle, resolution: int):
        """I dont get this part"""
        optimized_points = x.reshape(-1, 2)
        intersections = []
        for i in range(len(optimized_points) - 1):
            xp = np.linspace(
                optimized_points[i][0], optimized_points[i + 1][0], resolution
            )
            yp = np.linspace(
                optimized_points[i][1], optimized_points[i + 1][1], resolution
            )
            for x, y in zip(xp, yp):
                intersections.append(
                    np.sqrt((x - circle["x"]) ** 2 + (y - circle["y"]) ** 2)
                    - circle["r"]
                )
        return np.min(intersections)
        """I dont get this part"""

    def _equal_distances_constraint(self, x):
        """
        Make sure that the path is divided into equal length segments
        distances - euclidean distance between the points
        diff - calculate the difference between the distances (later we will check if the difference is 0)
        """
        optimized_points = x.reshape(-1, 2)
        points = np.vstack((self._first_point, optimized_points, self._last_point))
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.diff(distances)

    def _iter_callback(self, xk):
        self._iter += 1
        print(f"Iteration: {self._iter}")

    def solve(self, maxiter):
        res = minimize(
            self._objective,
            self._initial_guess_points.flatten(),
            constraints=self._constraints,
            bounds=self._bounds,
            options={"maxiter": maxiter},
            callback=self._iter_callback,
        )
        optimized_points = res.x.reshape(-1, 2)
        return optimized_points

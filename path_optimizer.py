"""Wykorzystaj model robota do zdefiniowania problemu optymalizacyjnego w którym robot ma się przemieścić
możliwie jak najkrótszą ścieżką w przestrzeni zadania z konfiguracji początkowej do konfiguracji w której jego
końcówka robocza znajdzie się w wewnątrz pewnej kuli, z zachowaniem orientacji pionowej przez cały czas
trwania ruchu."""

from scipy.optimize import minimize
import pinocchio as pin
import numpy as np


class PathOptimizer:

    DIM = 3

    def __init__(self, circles, n_segments, first_point, last_point, R):
        self._iter = 0
        self._n_optimizable_points = n_segments - 1
        self._first_point = first_point
        self._last_point = last_point
        self._R = R

        self._p0 = np.linspace(
            start=first_point,
            stop=last_point,
            num=self._n_optimizable_points + 2,
        )[1:-1]

        self._constraints = [
            {
                "type": "ineq",
                "fun": self._circle_constraint,
                "args": (circle,),
            }
            for circle in circles
        ] + [
            {
                "type": "eq",
                "fun": self._equal_distances_constraint,
            }
        ]

    def _objective(self, x):
        p = self._x2p(x)
        return np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))

    def _circle_constraint(self, x, circle):
        p = self._x2p(x)
        center = np.array([circle["x"], circle["y"], circle["z"]])
        c2p = np.linalg.norm(p - center, axis=1)
        return c2p - circle["r"]

    def _equal_distances_constraint(self, x):
        p = self._x2p(x)
        distances = np.linalg.norm(np.diff(p, axis=0), axis=1)
        return np.diff(distances)

    def solve(self, maxiter):
        res = minimize(
            self._objective,
            x0=self._p0.flatten(),
            constraints=self._constraints,
            options={"maxiter": maxiter},
            callback=self._iter_callback,
        )
        p = self._x2p(res.x)
        oMdes = [pin.SE3(self._R, pi) for pi in p]
        print(f"Finished!")
        return oMdes

    def _iter_callback(self, xk):
        self._iter += 1
        print(f"Iteration: {self._iter}")

    def _x2p(self, x):
        return np.vstack((self._first_point, x.reshape(-1, self.DIM), self._last_point))

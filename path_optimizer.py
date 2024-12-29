"""Wykorzystaj model robota do zdefiniowania problemu optymalizacyjnego w którym robot ma się przemieścić
możliwie jak najkrótszą ścieżką w przestrzeni zadania z konfiguracji początkowej do konfiguracji w której jego
końcówka robocza znajdzie się w wewnątrz pewnej kuli, z zachowaniem orientacji pionowej przez cały czas
trwania ruchu."""

from scipy.optimize import minimize
import pinocchio as pin
import numpy as np


class PathOptimizer:

    DIM = 3

    def __init__(self, robot, circle, n_segments, first_point, last_point, R):
        self._iter = 0
        self._robot = robot
        self._n_optimizable_points = n_segments - 1
        self._first_point = first_point
        self._last_point = last_point
        self._R = R

        self._p0 = np.linspace(
            start=first_point,
            stop=last_point,
            num=self._n_optimizable_points + 2,
        )[1:-1]

        bound_x = (0, 2)
        bound_y = (0, 2)
        bound_z = (0, 2)
        self._bounds = (bound_x, bound_y, bound_z) * self._n_optimizable_points
        self._constraints = []

    def _objective(self, x):
        p = self._x2p(x)
        return np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))

    def solve(self, maxiter):
        res = minimize(
            self._objective,
            x0=self._p0.flatten(),
            constraints=self._constraints,
            bounds=self._bounds,
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

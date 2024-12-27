import numpy as np
from scipy.optimize import minimize


"""
a) Linear functions satisfy the following properties:
    - Additivity
        f(x1+x2)=f(x1)+f(x2)

    - Homogeneity (Scaling) 
        f(c⋅x)=c⋅f(x)


b) Equality vs Inequality Constraints:
    - Equality
        g(x) = 0

    - Inequality
        g(x) >= 0
"""


def quadratic_objective(x):
    return x[0] ** 2 + x[1] ** 2


def non_quadratic_objective_2(x):
    return (x[0] ** 2 + x[1] ** 2) ** (1 / 2)


def main():
    x0 = np.array([1, 1])
    print(minimize(quadratic_objective, x0, method="SLSQP").x)
    print(minimize(non_quadratic_objective_2, x0, method="SLSQP").x)

    bounds = ((0.5, None), (0.5, None))
    constraints = (
        # Linear:
        {"type": "eq", "fun": lambda x: x[1] - 4},
        {"type": "ineq", "fun": lambda x: x[0] - 4},
        # # Non-linear:
        {"type": "eq", "fun": lambda x: x[0] ** 2},
        {"type": "ineq", "fun": lambda x: x[1] ** 2},
    )

    res = minimize(quadratic_objective, x0, method="SLSQP", constraints=constraints)
    print(res.x)


if __name__ == "__main__":
    main()

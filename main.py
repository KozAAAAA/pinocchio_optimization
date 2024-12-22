from time import sleep
import numpy as np
from scipy.interpolate import interp1d
from environment import Environment
from robot import Robot


def main():
    Tp = 1.0 / 500
    end_time = 3.0
    urdf_path = "iiwa_cup.urdf"

    robot = Robot(urdf_path)

    q = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    t = np.linspace(0.0, end_time, q.shape[0])

    qi = interp1d(t, q, axis=0)

    env = Environment(urdf_path, Tp, q[0], [0.5, 0.5, 0.5], 0.15)

    for i in range(int(end_time / Tp)):
        t_act = i * Tp

        q_act, dq_act = env.get_state()
        robot.forward_kinematics(q_act)

        q_i = qi(t_act)
        env.reset_joints_state(q_i, np.zeros_like(q_i))
        env.simulation_step()
        sleep(Tp)

if __name__ == "__main__":
    main()

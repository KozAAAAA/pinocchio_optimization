from time import sleep
import numpy as np
from scipy.interpolate import interp1d
from environment import Environment
from robot import Robot
from path_optimizer import PathOptimizer
import pinocchio as pin

Z = 0.7
TP = 1.0 / 500
SIM_TIME = 10.0
URDF_PATH = "iiwa_cup.urdf"
CIRCLE = {
    "x": 0.5,
    "y": 0.5,
    "z": 0.5,
    "r": 0.15,
}


def main():
    Rv = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    first_point = np.array([0.0, 0.0, Z])
    last_point = np.array([0.4, 0.4, Z])

    robot = Robot(URDF_PATH)
    opt = PathOptimizer(
        robot=robot,
        circle=CIRCLE,
        n_segments=10,
        first_point=first_point,
        last_point=last_point,
        R=Rv,
    )

    oMdes = opt.solve(1000)

    q = np.empty((0, 7))
    for i, oMd in enumerate(oMdes):
        sucess, qd, err = robot.inverse_kinematics(oMd, "F_joint_7")
        if not sucess:
            print(f"oMdes[{i}] not solved")
            return
        q = np.vstack((q, qd))

    t = np.linspace(0.0, SIM_TIME, q.shape[0])
    qi = interp1d(t, q, axis=0)

    env = Environment(
        urdf_path=URDF_PATH,
        timestep=TP,
        q0l=q[0],
        circle=CIRCLE,
    )
    for i in range(int(SIM_TIME / TP)):
        t_act = i * TP
        q_act, dq_act = env.get_state()

        M = robot.forward_kinematics(q_act)
        tau = robot.inverse_dynamics(q_act, dq_act, np.array([1.0] * 7))
        ddq = robot.forward_dynamics(q_act, dq_act, tau)
        J = robot.jacobian(q_act, "F_joint_7")

        q_i = qi(t_act)
        env.reset_joints_state(q_i, np.zeros_like(q_i))
        env.simulation_step()
        sleep(TP)


if __name__ == "__main__":
    main()

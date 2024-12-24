from time import sleep
import numpy as np
from scipy.interpolate import interp1d
from environment import Environment
from robot import Robot
import pinocchio as pin

TP = 1.0 / 500
SIM_TIME = 10.0
URDF_PATH = "iiwa_cup.urdf"


def main():
    robot = Robot(URDF_PATH)

    desired_ddq = np.array([1.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    oMdes = [
        pin.SE3(np.array([[-0.859547, 0.505643, -0.0741905], [-0.505646, -0.820367, 0.267059], [0.0741733, 0.267064, 0.96082]]), np.array([0.468445, 0.236337, 0.934516])),
        pin.SE3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0.0, 0.0, 1.261])),
        pin.SE3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0.5, 0.5, 0.5])),
        pin.SE3(np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0]]), np.array([0.3, 0.3, 0.3])),
    ]
    
    q = np.zeros(robot.model.nq)
    for i, oMd in enumerate(oMdes):
        sucess, qd, err = robot.inverse_kinematics(oMd, "F_joint_7")
        if sucess:
            print(f"oMdes[{i}] solved")
            qd = [(angle + np.pi) % (2 * np.pi) - np.pi for angle in qd]
            q = np.vstack((q, qd))
        else:
            print(f"oMdes[{i}] not solved")
            return

    t = np.linspace(0.0, SIM_TIME, q.shape[0])
    qi = interp1d(t, q, axis=0)

    env = Environment(
        urdf_path=URDF_PATH,
        timestep=TP,
        q0l=q[0],  # Initial joint positions
        xyz=[0.5, 0.5, 0.5],  # Generate sphere at xyz
        r=0.15,  # Radius of the sphere
    )

    for i in range(int(SIM_TIME / TP)):
        print(robot.data.oMi[-1])
        t_act = i * TP
        q_act, dq_act = env.get_state()

        M = robot.forward_kinematics(q_act)
        tau = robot.inverse_dynamics(q_act, dq_act, desired_ddq)
        ddq = robot.forward_dynamics(q_act, dq_act, tau)
        J = robot.jacobian(q_act, "F_joint_7")

        q_i = qi(t_act)
        env.reset_joints_state(q_i, np.zeros_like(q_i))
        env.simulation_step()
        sleep(TP)


if __name__ == "__main__":
    main()

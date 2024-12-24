import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve

class Robot:
    def __init__(self, urdf_path):
        """
        self.data - structure used to store non constant data
        self.data.oMi:
            - o - origin
            - M - transformation matrix
            - i - ith link
        """

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    def forward_kinematics(self, q):
        """Find the transformation matrix of the end effector given the joint positions"""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMi[-1]

    def inverse_kinematics(
        self,
        oMdes,
        joint_name,
        eps=1e-4,
        it_max=1000,
        dt=1e-1,
        damp=1e-12,
    ):
        """Find the joint positions given the transformation matrix of the end effector"""
        q = pin.neutral(self.model)
        joint_id = self.model.getJointId(joint_name)
        i = 0
        while True:
            pin.forwardKinematics(self.model, self.data, q)
            dMi = oMdes.actInv(self.data.oMi[joint_id])
            err = pin.log(dMi).vector
            if norm(err) < eps:
                success = True
                break
            if i >= it_max:
                success = False
                break
            J = pin.computeJointJacobian(self.model, self.data, q, joint_id)
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * dt)
            i += 1

        return success, q, err

    def forward_dynamics(self, q, dq, tau):
        """
        Find accelerations given the joint positions, velocities and torques
        """
        ddq = pin.aba(self.model, self.data, q, dq, tau)
        return ddq

    def inverse_dynamics(self, q, dq, ddq):
        """
        Find torques given the joint positions, velocities and accelerations
        q - current joint positions
        dq - current joint velocities

        ddq - desired joint accelerations
        """
        tau = pin.rnea(self.model, self.data, q, dq, ddq)
        return tau

    def jacobian(self, q, frame_name):
        """
        Calculates the Jacobian matrix of size 6 x n, where n is the number of joints (DOF).
        The number 6 comes from 3 linear velocities and 3 angular velocities.

        dX = [dx, dy, dz, droll, dpitch, dyaw] - end effector velocities
        dq = [dq1, dq2, dq3, dq4, dq5, dq6, dq7] - joint velocities


               l1   l2   l3   l4   l5   l6   l7

        J = | J11  J12  J13  J14  J15  J16  J17 |   dx
            | J21  J22  J23  J24  J25  J26  J27 |   dy
            | J31  J32  J33  J34  J35  J36  J37 |   dz
            | J41  J42  J43  J44  J45  J46  J47 |   droll
            | J51  J52  J53  J54  J55  J56  J57 |   dpitch
            | J61  J62  J63  J64  J65  J66  J67 |   dyaw

        Jij - how j-th joint velocity affects i-th end effector velocity

        dX = J * dq
        """
        J = pin.computeFrameJacobian(
            model=self.model,
            data=self.data,
            q=q,
            frame_id=self.model.getFrameId(frame_name),
            reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return J

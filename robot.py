import pinocchio as pin

class Robot():
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        # Data needed to store non constant data
        self.data = self.model.createData()

    def forward_kinematics(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        for name, oMi in zip(self.model.names, self.data.oMi):
            print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
        

    def inverse_kinematics(self, x):
        '''has to be efficient'''
        pass

    def forward_dynamics(self, q, dq, tau):
        pass

    def inverse_dynamics(self, q, dq, ddq):
        pass


        pass


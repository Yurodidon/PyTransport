import numpy as np

# Needs to be redefined for specific setup (V)
class Potential:

    # p -> parameters for this potential
    def __init__(self, p, nF):
        self.p = p
        self.nF = nF

    # V and its derivatives
    def V(self, Q):
        V = 0
        return V

    def dV(self, Q):
        dV = np.zeros(self.nF)
        return dV

    def ddV(self, Q):
        ddV = np.zeros((self.nF, self.nF))
        return ddV

    def dddV(self, Q):
        dddV = np.zeros((self.nF, self.nF, self.nF))
        return dddV
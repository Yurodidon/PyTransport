import numpy as np


# Needs to be redefined for specific setup (V)
class Potential:
    # p -> parameters for this potential
    def __init__(self, p, nF):
        self.p = p
        self.nF = nF

    # V and its derivatives
    def V(self, Q):
        return 0

    def dV(self, Q):
        return np.zeros(self.nF)

    def ddV(self, Q):
        return np.zeros((self.nF, self.nF))

    def dddV(self, Q):
        return np.zeros((self.nF, self.nF, self.nF))

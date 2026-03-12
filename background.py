import numpy as np
from utilities import inner
from potential import Potential

class Background:

    def __init__(self, pot: Potential):
        self.pot = pot
        pass

    def H(self, Q):  # Eq 4.3
        H2 = (1 / 6) * inner(Q[:, 1], Q[:, 1]) + (1 / 3) * self.pot.V(Q)
        return np.sqrt(H2)

    def epsilon(self, Q):  # Eq 4.4
        epsilon = -1 * inner(Q[:, 1], Q[:, 1]) / (2 * np.square(self.H(Q)))
        return epsilon

    @staticmethod
    def a(N):
        return np.exp(N)



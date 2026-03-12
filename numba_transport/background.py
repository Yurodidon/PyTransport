import numpy as np

from kernels import H_nb, as_scalar, epsilon_nb
from potential import Potential


class Background:
    def __init__(self, pot: Potential):
        self.pot = pot

    def H(self, Q):  # Eq 4.3
        Q = np.asarray(Q, dtype=float)
        return float(H_nb(Q, as_scalar(self.pot.V(Q))))

    def epsilon(self, Q):  # Eq 4.4
        Q = np.asarray(Q, dtype=float)
        H = self.H(Q)
        return float(epsilon_nb(Q, H))

    @staticmethod
    def a(N):
        return np.exp(N)

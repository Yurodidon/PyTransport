import numpy as np

from background import Background
from kernels import (
    A_unsym_nb,
    B_unsym_nb,
    C_unsym_nb,
    M_nb,
    N_a_nb,
    N_ab_nb,
    as_matrix,
    as_scalar,
    as_tensor3,
    as_vector,
    calc_dpi_nb,
    inner_nb,
    symmetrize_A_nb,
    symmetrize_BC_nb,
    u2_nb,
    u3_nb,
    w_nb,
    xi_nb,
)
from potential import Potential


class Matrices:
    # V -> The potential class defined in potential.py
    # nF -> # of fields.
    def __init__(self, nF, pot: Potential, bkg: Background, N):
        self.nF = nF
        self.pot = pot
        self.bkg = bkg
        self.N = N

    def xi(self, Q):
        Q = np.asarray(Q, dtype=float)
        H = self.bkg.H(Q)
        pi = np.asarray(Q[:, 1], dtype=float)
        dV = as_vector(self.pot.dV(Q), self.nF)
        dpi = calc_dpi_nb(pi, H, dV)
        return xi_nb(pi, dpi, H)

    # Calculate the A matrix. Eq 4.8a.
    # If slow is set to True, then the last term of k^2/a^2 will be dropped
    # As explained on the bottom of Page 29
    def A(self, k, Q, slow=False):
        Q = np.asarray(Q, dtype=float)
        dddV = as_tensor3(self.pot.dddV(Q), self.nF)
        ddV = as_matrix(self.pot.ddV(Q), self.nF)
        V = as_scalar(self.pot.V(Q))
        H = self.bkg.H(Q)
        a_scale = self.bkg.a(self.N)  # Scale factor a
        pi = np.asarray(Q[:, 1], dtype=float)
        dV = as_vector(self.pot.dV(Q), self.nF)
        dpi = calc_dpi_nb(pi, H, dV)
        xi = xi_nb(pi, dpi, H)

        # We first calculate the unsymmetrized one
        A012 = A_unsym_nb(k[0], k[1], k[2], dddV, ddV, V, H, a_scale, pi, xi, slow)
        A021 = A_unsym_nb(k[0], k[2], k[1], dddV, ddV, V, H, a_scale, pi, xi, slow)
        A102 = A_unsym_nb(k[1], k[0], k[2], dddV, ddV, V, H, a_scale, pi, xi, slow)
        A120 = A_unsym_nb(k[1], k[2], k[0], dddV, ddV, V, H, a_scale, pi, xi, slow)
        A201 = A_unsym_nb(k[2], k[0], k[1], dddV, ddV, V, H, a_scale, pi, xi, slow)
        A210 = A_unsym_nb(k[2], k[1], k[0], dddV, ddV, V, H, a_scale, pi, xi, slow)

        # The time efficiency of this is O(nF^6). For small nF this shouldn't be a problem
        return symmetrize_A_nb(A012, A021, A102, A120, A201, A210)

    def B(self, k, Q):  # Eq 4.8b
        Q = np.asarray(Q, dtype=float)
        H = self.bkg.H(Q)
        pi = np.asarray(Q[:, 1], dtype=float)
        dV = as_vector(self.pot.dV(Q), self.nF)
        dpi = calc_dpi_nb(pi, H, dV)
        xi = xi_nb(pi, dpi, H)

        B012 = B_unsym_nb(k[0], k[1], k[2], H, pi, xi)
        B102 = B_unsym_nb(k[1], k[0], k[2], H, pi, xi)
        return symmetrize_BC_nb(B012, B102)

    def C(self, k, Q):  # Eq 4.8c
        Q = np.asarray(Q, dtype=float)
        H = self.bkg.H(Q)
        pi = np.asarray(Q[:, 1], dtype=float)

        C012 = C_unsym_nb(k[0], k[1], k[2], H, pi)
        C102 = C_unsym_nb(k[1], k[0], k[2], H, pi)
        return symmetrize_BC_nb(C012, C102)

    def M(self, k, Q, return_m=False):  # Matrix M with Eq 5.3
        # Note, as the small matrix m in 4.7b is used as well
        # set return_m to True for the small m
        Q = np.asarray(Q, dtype=float)
        ddV = as_matrix(self.pot.ddV(Q), self.nF)
        epsilon = self.bkg.epsilon(Q)
        H = self.bkg.H(Q)
        a = self.bkg.a(self.N)
        k_scalar = float(k[0])  # Note for 2pf, we are only using one k

        pi = np.asarray(Q[:, 1], dtype=float)
        dV = as_vector(self.pot.dV(Q), self.nF)
        dpi = calc_dpi_nb(pi, H, dV)

        M, m_mat = M_nb(k_scalar, ddV, epsilon, H, a, pi, dpi)
        return M if not return_m else m_mat

    ###################################################################################
    # For the evolution matrix, we will follow the paper and use N as the time variable
    # Note d/dN = 1/H * d/dt. So we will divide u2 and u3, w by H
    ###################################################################################

    def u2(self, k, Q):  # Eq 5.2, we dont care the untiled one
        m_tiled = self.M(k, Q)
        H = self.bkg.H(np.asarray(Q, dtype=float))
        return u2_nb(m_tiled, H)

    def u3(self, k, Q):
        Q = np.asarray(Q, dtype=float)
        # We need to swap k for the evolution
        A = self.A(k, Q)

        B1 = self.B([k[1], k[2], k[0]], Q)
        B2 = self.B([k[0], k[2], k[1]], Q)
        B3 = self.B([k[0], k[1], k[2]], Q)

        C1 = self.C([k[0], k[1], k[2]], Q)
        C2 = self.C([k[0], k[2], k[1]], Q)
        C3 = self.C([k[2], k[1], k[0]], Q)

        H = self.bkg.H(Q)

        # Note u3 is defined as u^a_bc. a for rows, b for columns, c for the term in {}.
        # See the text under 4.19 for clear explanation
        # We index as u3[a, b, c, ...]
        return u3_nb(A, B1, B2, B3, C1, C2, C3, H)

    # Eq 5.12. Evolution matrix for tensor modes
    def w(self, k, Q):
        Q = np.asarray(Q, dtype=float)
        H = self.bkg.H(Q)
        a = self.bkg.a(self.N)
        return w_nb(float(k[0]), H, a)


# The final gauge transformation to get the 2pf and 3pf
class N:
    def __init__(self, nF, pot: Potential):
        self.nF = nF
        self.pot = pot

    # Eq 7.4a
    def N_a(self, Q):
        Q = np.asarray(Q, dtype=float)
        bkg = Background(self.pot)
        pi = np.asarray(Q[:, 1], dtype=float)
        H = bkg.H(Q)
        epsilon = bkg.epsilon(Q)
        return N_a_nb(pi, H, epsilon)

    # Eq 7.4b
    def N_ab(self, Q, k):
        Q = np.asarray(Q, dtype=float)
        bkg = Background(self.pot)
        pi = np.asarray(Q[:, 1], dtype=float)
        H = bkg.H(Q)
        dV = as_vector(self.pot.dV(Q), self.nF)
        epsilon = -bkg.epsilon(Q)  # To match the official code
        return N_ab_nb(pi, H, dV, epsilon, float(k[0]), float(k[1]), float(k[2]))

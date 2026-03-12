import numpy as np
from potential import Potential
from background import Background
from utilities import *
import numba

class Matrices:

    # V -> The potential class defined in potential.py
    # nF -> # of fields.
    def __init__(self, nF, pot: Potential, bkg: Background, N):
        self.nF = nF
        self.pot = pot
        self.bkg = bkg
        self.N = N

    def xi(self, Q):
        H = self.bkg.H(Q)
        pi = Q[:, 1]
        dpi = calc_dpi(Q, self.bkg, self.pot)
        nF = self.nF

        xi = np.zeros(nF)
        for i in range(nF):
            xi[i] = 2 * dpi[i] + (pi[i] / H) * inner(pi, pi)

        return xi

    # Calculate the A matrix. Eq 4.8a.
    # If slow is set to True, then the last term of k^2/a^2 will be dropped
    # As explained on the bottom of Page 29
    def A(self, k, Q, slow=False):

        dddV = self.pot.dddV(Q)
        ddV = self.pot.ddV(Q)
        V = self.pot.V(Q)
        H = self.bkg.H(Q)
        a_scale = self.bkg.a(self.N) # Scale factor a
        nF = self.nF

        pi = Q[:, 1]
        xi = self.xi(Q)

        # We first calculate the unsymmetrized one
        def calc_unsym(i, j, m):
            A_unsym = np.zeros((nF, nF, nF))
            k_dot_k = (k[i] * k[i] - k[j] * k[j] - k[m] * k[m]) / 2
            kk = k[j] * k[j] * k[m] * k[m]

            for a in range(nF): # alpha
                for b in range(nF): # beta
                    for g in range(nF): # gamma
                        A_unsym[a][b][g] = (-1 / 3) * dddV[a][b][g] - (pi[a] * ddV[b][g]) / (2 * H) \
                                           + (pi[a] * pi[b] * xi[g]) / (8 * np.square(H)) \
                                           + (pi[a] * xi[b] * xi[g]) / (32 * H ** 3) * (1 - np.square(k_dot_k) / kk) \
                                           + (pi[a] * pi[b] * pi[g]) / (8 * H ** 3) * 2 * V
                                                # Note the last term is simplified with Eq 4.3
                        if not slow:
                            A_unsym[a][b][g] = A_unsym[a][b][g] + (pi[a] * delta(b, g)) / (2 * H) * (k_dot_k / np.square(a_scale))

            return A_unsym

        A = np.zeros((nF, nF, nF))
        # The time efficiency of this is O(nF^6). For small nF this shouldn't be a problem
        A012 = calc_unsym(0, 1, 2); A021 = calc_unsym(0, 2, 1); A102 = calc_unsym(1, 0, 2)
        A120 = calc_unsym(1, 2, 0); A201 = calc_unsym(2, 0, 1); A210 = calc_unsym(2, 1, 0)
        for a in range(nF):
            for b in range(nF):
                for g in range(nF): # a -> 0, b -> 1, g -> 2
                    A[a][b][g] = A012[a][b][g] + A021[a][g][b] + A102[b][a][g] \
                              + A120[b][g][a] + A201[g][a][b] + A210[g][b][a]

                    A[a][b][g] /= 6

        return A

    def B(self, k, Q): # Eq 4.8b
        H = self.bkg.H(Q)
        nF = self.nF
        pi = Q[:, 1]
        xi = self.xi(Q)

        def calc_unsym(i, j, m):
            B_unsym = np.zeros((nF, nF, nF))
            k_dot_k_1 = (k[i] * k[i] - k[j] * k[j] - k[m] * k[m]) / 2
            k_dot_k_2 = (k[m] * k[m] - k[i] * k[i] - k[j] * k[j]) / 2
            kk = k[j] * k[j] * k[m] * k[m]

            for a in range(nF): # alpha
                for b in range(nF): # beta
                    for g in range(nF): # gamma
                        B_unsym[a][b][g] = (pi[a] * pi[b] * pi[g]) / (4 * H**2) \
                                        - (pi[a] * xi[b] * pi[g]) / (8 * H**3) * (1 - k_dot_k_1**2 / kk) \
                                        - (xi[a] * delta(b, g)) / (2 * H) * (k_dot_k_2 / k[i]**2)

            return B_unsym

        B = np.zeros((nF, nF, nF))
        B012 = calc_unsym(0, 1, 2); B102 = calc_unsym(1, 0, 2)
        for a in range(nF): # Note for B we are only symmetrizing over alpha & beta
            for b in range(nF):
                for g in range(nF):  # a -> 0, b -> 1, g -> 2
                    B[a][b][g] = B012[a][b][g] + B102[b][a][g]

                    B[a][b][g] /= 2

        return B

    def C(self, k, Q):  # Eq 4.8c
        H = self.bkg.H(Q)
        nF = self.nF
        pi = Q[:, 1]

        def calc_unsym(i, j, m):
            C_unsym = np.zeros((nF, nF, nF))
            k_dot_k_1 = (k[m] * k[m] - k[i] * k[i] - k[j] * k[j]) / 2
            k_dot_k_2 = (k[j] * k[j] - k[i] * k[i] - k[m] * k[m]) / 2
            kk = k[i] * k[i] * k[j] * k[j]

            for a in range(nF):  # alpha
                for b in range(nF):  # beta
                    for g in range(nF):  # gamma
                        C_unsym[a][b][g] = -(delta(a, b) * pi[g]) / (2 * H) \
                                           + (pi[a] * pi[b] * pi[g]) / (8 * H ** 3) * (1 - k_dot_k_1 ** 2 / kk) \
                                           + (delta(b, g) * pi[a]) / H * (k_dot_k_2 / k[i] ** 2)

            return C_unsym

        C = np.zeros((nF, nF, nF))
        C012 = calc_unsym(0, 1, 2); C102 = calc_unsym(1, 0, 2)

        for a in range(nF):  # Note for C we are only symmetrizing over alpha & beta
            for b in range(nF):
                for g in range(nF):  # a -> 0, b -> 1, g -> 2
                    C[a][b][g] = C012[a][b][g] + C102[b][a][g]

                    C[a][b][g] /= 2

        return C

    def M(self, k, Q, return_m=False): # Matrix M with Eq 5.3
        # Note, as the small matrix m in 4.7b is used as well
        # set return_m to True for the small m
        ddV = self.pot.ddV(Q)
        epsilon = self.bkg.epsilon(Q)
        H = self.bkg.H(Q)
        a = self.bkg.a(self.N)
        k = k[0] # Note for 2pf, we are only using one k

        pi = Q[:, 1]
        dpi = calc_dpi(Q, self.bkg, self.pot)
        nF = self.nF

        M = np.zeros((nF, nF))
        m_mat = np.zeros((nF, nF))
        for i in range(nF):
            for j in range(nF):
                m = ddV[i][j] - (3 - epsilon) * pi[i] * pi[j] \
                            - (pi[i] * dpi[j] + pi[j] * dpi[i]) / H # Cautious: dpi is a function of t
                M[i][j] = -1 * delta(i, j) * k**2 / np.square(a) - m

                if return_m:
                    m_mat[i, j] = m

        return M if not return_m else m_mat

    ###################################################################################
    # For the evolution matrix, we will follow the paper and use N as the time variable
    # Note d/dN = 1/H * d/dt. So we will divide u2 and u3, w by H
    ###################################################################################

    def u2(self, k, Q):  # Eq 5.2, we dont care the untiled one
        m_tiled = self.M(k, Q)
        H = self.bkg.H(Q)
        nF = self.nF

        u2 = np.zeros((2, 2, nF, nF))

        for a in range(nF):
            for b in range(nF):
                u2[0, 1, a, b] = delta(a, b)
                u2[1, 0, a, b] = m_tiled[a][b]
                u2[1, 1, a, b] = -3 * H * delta(a, b)

        # Use N as time variable
        u2 = u2 / H

        return u2

    def u3(self, k, Q):
        k1, k2, k3 = k # We need to swap k for the evolution
        A = self.A(k, Q)

        B1 = self.B([k2, k3, k1], Q)
        B2 = self.B([k1, k3, k2], Q)
        B3 = self.B([k1, k2, k3], Q)

        C1 = self.C([k1, k2, k3], Q)
        C2 = self.C([k1, k3, k2], Q)
        C3 = self.C([k3, k2, k1], Q)

        nF = self.nF
        H = self.bkg.H(Q)

        # Note u3 is defined as u^a_bc. a for rows, b for columns, c for the term in {}.
        # See the text under 4.19 for clear explanation
        # We index as u3[a, b, c, ...]
        u3 = np.zeros((2, 2, 2, nF, nF, nF))

        for a in range(nF):
            for b in range(nF):
                for g in range(nF):
                    u3[0, 0, 0, a, b, g] = -B1[b][g][a]
                    u3[0, 1, 0, a, b, g] = -C1[a][b][g]
                    u3[1, 0, 0, a, b, g] = 3 * A[a][b][g]
                    u3[1, 1, 0, a, b, g] = B2[a][g][b]

                    u3[0, 0, 1, a, b, g] = -C2[a][g][b]
                    u3[0, 1, 1, a, b, g] = 0
                    u3[1, 0, 1, a, b, g] = B3[a][b][g]
                    u3[1, 1, 1, a, b, g] = C3[g][b][a]

        u3 = u3 / H

        return u3

    # Eq 5.12. Evolution matrix for tensor modes
    def w(self, k, Q):
        bkg = self.bkg
        H = bkg.H(Q)
        N = self.N
        a = bkg.a(N)
        k2 = k[0]**2 # Still, we only use one k here

        w = np.zeros((2, 2))

        w[0, 0] = 0
        w[0, 1] = 1
        w[1, 0] = -1 * k2 / (a**2)
        w[1, 1] = -3 * H

        w = w / H

        return w


# The final gauge transformation to get the 2pf and 3pf
class N:
    def __init__(self, nF, pot: Potential):
        self.nF = nF
        self.pot = pot

    # Eq 7.4a
    def N_a(self, Q):
        pot = self.pot
        bkg = Background(pot)
        nF = self.nF
        N_a = np.zeros((2, nF))
        pi = Q[:, 1]
        H = bkg.H(Q)
        epsilon = bkg.epsilon(Q)

        for i in range(nF):
            # In the official code, there is no negative sign
            N_a[0, i] = pi[i] / (2 * H * epsilon)
            N_a[1, i] = 0

        return N_a

    # Eq 7.4b
    def N_ab(self, Q, k):
        pot = self.pot
        bkg = Background(pot)
        nF = self.nF
        pi = Q[:, 1]
        H = bkg.H(Q)
        dV = pot.dV(Q)
        epsilon = -1 * bkg.epsilon(Q) # To match the official code
        k_dot_k = (k[0]**2 - k[1]**2 - k[2]**2) / 2

        N_ab = np.zeros((2, 2, nF, nF))
        for i in range(nF):
            for j in range(nF):
                N_ab[0, 0, i, j] = pi[i] * pi[j] * (-3 / 2 + 9 / (2 * epsilon) + 3 / (4 * epsilon**2) * inner(dV, pi) / (H**3))
                N_ab[0, 1, i, j] = 3 / (H * epsilon) * pi[i] * pi[j] - (3 * H) / (k[0]**2) * (k_dot_k + k[2]**2) * delta(i, j)
                N_ab[1, 0, i, j] = 3 / (H * epsilon) * pi[i] * pi[j] - (3 * H) / (k[0]**2) * (k_dot_k + k[1]**2) * delta(i, j)
                N_ab[1, 1, i, j] = 0

                # This is in the official code. But the reason of doing this is still unclear.
                N_ab[0, 1, i, j] /= 2
                N_ab[1, 0, i, j] /= 2
                ############### Needs further checks ###############

        N_ab = N_ab / (3 * H**2 * epsilon)


        return N_ab





"""
Important Note: The matrices implemented have several coefficient difference from the paper
We followed the official code. The difference is manifested on the H factors
Without these modifications the evolution diverges. With them we got appropriate results
"""

import numpy as np

from evolve import EvolveFields
from potential import Potential
from background import Background
from matrices import Matrices
from utilities import *

# Calculate the sum of the permutation. Used in initial.py
def permu_sum(permu):
    return permu(0, 1, 2) + permu(0, 2, 1) + permu(1, 2, 0) \
            + permu(1, 0, 2) + permu(2, 0, 1) + permu(2, 1, 0) # 6 terms

# For this class, we need to calculate the initial time (t_init)
# And evaluate the initial Sigma and B matrices
class Initial:

    # We find a general N_int regarding the smallest k. N_pre should be in range of 3-5
    def __init__(self, nF, pot: Potential, k_min, Q_N, N_pre=5):
        self.nF = nF
        self.pot = pot

        self.init = self.find_init(k_min, Q_N, N_pre)

    ################################################################################################

    # Logic: Q_N is already fixed, start from small N such as N=0. And we believe that N=0 is deep enough
    # So we increase N, until we found the zero crossing point of k/a-m or k-aH, as the initial point of the evolution
    # Of course we substract N_pre to make sure the (k/a)>>m
    # Then we return the mat, bkg, N_init and Q at this point
    # Now the input k is a SCALAR!!!

    def find_m_init(self, k_min, Q_N, N_pre):
        bkg = Background(self.pot)
        sub_m = -1; Q_m = None
        for Q in Q_N:
            Q_m = Q
            N = Q['N']; q = Q['Q']
            a = bkg.a(N)
            ddV = self.pot.ddV(q)
            w, _ = np.linalg.eig(ddV)
            m2 = np.max(w)
            sub_m = m2 - k_min ** 2 / (a ** 2)

            if sub_m > 0: break  # The zero-crossing point

        if sub_m < 0: return None, None

        N_init = Q_m['N'] - N_pre
        for Q in Q_N:
            if Q['N'] >= N_init: 
                Q_m = Q
                break
        return N_init, Q_m

    def find_H_init(self, k_min, Q_N, N_pre):
        bkg = Background(self.pot)
        sub_h = -1; Q_h = None
        for Q in Q_N:
            Q_h = Q
            N = Q['N']; q = Q['Q']
            a = bkg.a(N)
            H = bkg.H(q)
            sub_h = a * H - k_min
            if sub_h > 0: break  # The zero-crossing point

        if sub_h < 0: return None, None

        N_init = Q_h['N'] - N_pre
        for Q in Q_N:
            if Q['N'] >= N_init: 
                Q_h = Q
                break
        return N_init, Q_h

    def find_init(self, k_min, Q_N, N_pre):
        # Consider the mass and H
        N_init_m, Q_m = self.find_m_init(k_min, Q_N, N_pre)
        N_init_h, Q_h = self.find_H_init(k_min, Q_N, N_pre)

        # Both are smaller than 0, no crossing point found
        if N_init_m is None and N_init_h is None: raise ValueError(f"No crossing point found for k={k_min}!!!!!!!")

        N_init = N_init_m if (N_init_h is None or N_init_m < N_init_h) else N_init_h
        Q_init = Q_m['Q'] if (N_init_h is None or N_init_m < N_init_h) else Q_h['Q']

        bkg = Background(self.pot)
        mat = Matrices(self.nF, self.pot, bkg, N_init)
        
        return mat, bkg, N_init, Q_init

    ################################################################################################

    def get_init(self):
        return self.init

    # Implementation of Eq 6.2
    def sigma(self, k):
        # Note for sigma we only use one k. We take the first component
        mat, bkg, N_init, Q = self.init
        nF = self.nF
        k = k[0] # Here!!
        a = bkg.a(N_init)
        H = bkg.H(Q)

        sigma_re = np.zeros((2, 2, nF, nF))
        sigma_im = np.zeros((2, 2, nF, nF))
        for i in range(nF):
            for j in range(nF):
                sigma_re[0, 0, i, j] = a * delta(i, j)
                sigma_re[0, 1, i, j] = -1 * a * H * delta(i, j)
                sigma_re[1, 0, i, j] = -1 * a * H * delta(i, j)
                sigma_re[1, 1, i, j] = (k**2 / a) * delta(i, j)

                sigma_im[0, 0, i, j] = 0
                sigma_im[0, 1, i, j] = k * delta(i, j)
                sigma_im[1, 0, i, j] = -k * delta(i, j) # There is a typo at Eq 6.2b
                sigma_im[1, 1, i, j] = 0

        sigma_im = sigma_im / (2 * a**3 * k)
        sigma_re = sigma_re / (2 * a**3 * k)

        return sigma_re, sigma_im

    # The full B matrix
    def B_full(self, k):
        nF = self.nF
        mat, bkg, N_init, Q = self.init
        a = bkg.a(N_init)

        B = np.zeros((2, 2, 2, nF, nF, nF))

        # The full B matrix. Note some part of it may be unncessary but we still keep them.
        B[0, 0, 0] = self.B_fff(k)

        B[1, 0, 0] = self.B_pff(k) / a
        B[0, 1, 0] = np.transpose(self.B_pff((k[1], k[0], k[2])) / a, (1, 0, 2))
        B[0, 0, 1] = np.transpose(self.B_pff((k[2], k[0], k[1])) / a, (2, 0, 1))

        B[1, 1, 0] = self.B_ppf(k) / (a**2)
        B[1, 0, 1] = np.transpose(self.B_ppf((k[0], k[2], k[1])) / (a**2), (0, 2, 1))
        B[0, 1, 1] = np.transpose(self.B_ppf((k[1], k[2], k[0])) / (a**2), (1, 2, 0))

        B[1, 1, 1] = self.B_ppp(k) / (a**3)

        return B


    # Eq 6.7a
    def B_fff(self, k):
        # Note: we take mat, bkg and N_init as input.
        # The permutation of k1, k2, k3 will not change N_init (only depend on modulus)
        # Hence mat and bkg are not changed as well
        mat, bkg, N_init, Q = self.init
        pi = Q[:, 1]
        H = bkg.H(Q)
        a = bkg.a(N_init)
        nF = self.nF
        Ks = k[0] * k[1] + k[0] * k[2] + k[1] * k[2]

        # Implement the permutation
        def permu(x, y, z):
            # x -> 1, y -> 2, z -> 3
            k_dot_k = (k[x]**2 - k[y]**2 - k[z]**2) / 2

            p = np.zeros((nF, nF, nF))

            A_slow = mat.A((k[x], k[y], k[z]), Q, slow=True)
            B = mat.B((k[x], k[y], k[z]), Q)
            C = mat.C((k[x], k[y], k[z]), Q)

            for i in range(nF):
                for j in range(nF):
                    for m in range(nF):
                        p[i, j, m] = pi[i] / (4 * H) * delta(j, m) * k_dot_k - (1 / 2) * C[i, j, m] * k[x] * k[y] \
                                    + (a**2 / 2) * A_slow[i, j, m] \
                                    + (a**2 * H / 2) * B[i, j, m] * ((k[x] + k[y]) * k[z] / (k[x] * k[y]) - Ks / (k[x] * k[y]))

            return p

        # Sum all 6 permutations
        B_fff = permu_sum(permu) / (4 * a**4) / (k[0] * k[1] * k[2] * (k[0] + k[1] + k[2]))

        return B_fff

    # Eq 6.7b
    def B_pff(self, k):
        mat, bkg, N_init, Q = self.init
        pi = Q[:, 1]
        H = bkg.H(Q)
        a = bkg.a(N_init)
        nF = self.nF
        kt = k[0] + k[1] + k[2]
        Ks = k[0] * k[1] + k[0] * k[2] + k[1] * k[2]

        # First square bracket
        def permu1(x, y, z):
            k_dot_k = (k[x] ** 2 - k[y] ** 2 - k[z] ** 2) / 2
            A_slow = mat.A((k[x], k[y], k[z]), Q,slow=True)
            C = mat.C((k[x], k[y], k[z]), Q)

            p = np.zeros((nF, nF, nF))

            for i in range(nF):
                for j in range(nF):
                    for m in range(nF):
                        p[i, j, m] = pi[i] / (4 * H) * delta(j, m) * k_dot_k + (a**2 / 2) * A_slow[i, j, m] \
                                    - (1 / 2) * C[i, j, m] * k[x] * k[y]

            return p

        # Second square bracket
        def permu2(x, y, z):
            k_dot_k = (k[x] ** 2 - k[y] ** 2 - k[z] ** 2) / 2
            A_slow = mat.A((k[x], k[y], k[z]), Q, slow=True)
            B = mat.B((k[x], k[y], k[z]), Q)
            C = mat.C((k[x], k[y], k[z]), Q)

            p = np.zeros((nF, nF, nF))

            for i in range(nF):
                for j in range(nF):
                    for m in range(nF):
                        p[i, j, m] = (k[x]**2 * k[y]**2 / 2) * (1 + k[z] / kt) * C[i, j, m] \
                                    + (1 / (2 * H)) * B[i, j, m] * k[x] * k[y] * k[z]**2 \
                                    - pi[i] / (4 * H) * delta(j, m) * (Ks + k[x] * k[y] * k[z] / kt) * k_dot_k \
                                    - a**2 / 2 * A_slow[i, j, m] * (Ks - k[x] * k[y] * k[z] / kt)

            return p

        B_pff = k[0]**2 * (k[1] + k[2]) * permu_sum(permu1)
        B_pff += k[0] * permu_sum(permu2)
        B_pff = B_pff * H / (4 * a**3) * (1 / ((k[0] * k[1] * k[2])**2 * kt))
        # The coefficient is different. I will use the one in the official code.

        return B_pff

    def B_ppf(self, k):
        mat, bkg, N_init, Q = self.init
        pi = Q[:, 1]
        H = bkg.H(Q)
        a = bkg.a(N_init)
        nF = self.nF
        kt = k[0] + k[1] + k[2]

        # First square bracket
        def permu1(x, y, z):
            k_dot_k = (k[x] ** 2 - k[y] ** 2 - k[z] ** 2) / 2
            A_slow = mat.A((k[x], k[y], k[z]), Q, slow=True)
            B = mat.B((k[x], k[y], k[z]), Q)
            C = mat.C((k[x], k[y], k[z]), Q)

            p = np.zeros((nF, nF, nF))

            for i in range(nF):
                for j in range(nF):
                    for m in range(nF):
                        p[i, j, m] = (1 / 2) * C[i, j, m] * k[x] * k[y] \
                                    - pi[i] / (4 * H) * delta(j, m) * k_dot_k \
                                    - (a**2 / 2) * A_slow[i, j, m] \
                                    - (a**2 * H) / 2 * B[i, j, m] * (k[x] + k[y]) * k[z] / (k[x] * k[y])

            return p

        # Second square bracket
        def permu2(x, y, z):
            B = mat.B((k[x], k[y], k[z]), Q)

            p = np.zeros((nF, nF, nF))

            for i in range(nF):
                for j in range(nF):
                    for m in range(nF):
                        p[i, j, m] = a**2 * H / 2 * B[i, j, m] * k[z]

            return p

        B_ppf = k[0]**2 * k[1]**2 * k[2] * permu_sum(permu1)
        B_ppf += k[0]**2 * k[1]**2 * k[0] * k[1] * k[2] * permu_sum(permu2) # This is also in the official code.
        B_ppf = B_ppf * 1 / (4 * a**4) * (1 / ((k[0] * k[1] * k[2])**2 * kt))
        # NOTE!! In the paper there is a factor of 1/H^2. But in the actual implementation there is not
        # And the a**6 is a**4. I suspect they cancelled aH=1? But Why??
        return B_ppf

    def B_ppp(self, k):
        mat, bkg, N_init, Q = self.init
        pi = Q[:, 1]
        H = bkg.H(Q)
        a = bkg.a(N_init)
        nF = self.nF
        Ks = k[0] * k[1] + k[0] * k[2] + k[1] * k[2]
        kt = k[0] + k[1] + k[2]

        # Implement the permutation
        def permu(x, y, z):
            # x -> 1, y -> 2, z -> 3
            k_dot_k = (k[x] ** 2 - k[y] ** 2 - k[z] ** 2) / 2

            p = np.zeros((nF, nF, nF))

            A_slow = mat.A((k[x], k[y], k[z]), Q, slow=True)
            B = mat.B((k[x], k[y], k[z]), Q)
            C = mat.C((k[x], k[y], k[z]), Q)

            for i in range(nF):
                for j in range(nF):
                    for m in range(nF):
                        p[i, j, m] = pi[i] / (4 * H) * delta(j, m) * (Ks + k[x] * k[y] * k[z] / kt) * k_dot_k \
                                      - (1 / (2 * H)) * B[i, j, m] * (k[x]**2 * k[y] * k[z]) \
                                      - (k[x]**2 * k[y]**2 / 2) * (1 + k[z] / kt) * C[i, j, m] \
                                      + (a**2 / 2) * A_slow[i, j, m] * (Ks - (k[x] * k[y] * k[z]) / kt)

            return p

        # Sum all 6 permutations. Again coefficient is different
        B_ppp = permu_sum(permu) * H / (4 * a ** 3) / (k[0] * k[1] * k[2] * kt)

        return B_ppp

    # Eq 6.3. The initial condition for tensor modes
    def gamma(self, k):
        mat, bkg, N_init, Q = self.init
        a = bkg.a(N_init)
        H = bkg.H(Q)
        k = k[0] # Again we are only using one k

        gamma = np.zeros((2, 2))

        gamma[0, 0] = a
        gamma[0, 1] = -1 * a * H
        gamma[1, 0] = -1 * a * H
        gamma[1, 1] = k**2 / a

        gamma = gamma / (a**3 * k) / 2 # In official code there is a factor of 0.5

        return gamma
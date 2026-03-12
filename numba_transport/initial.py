"""
Important Note: The matrices implemented have several coefficient difference from the paper
We followed the official code. The difference is manifested on the H factors
Without these modifications the evolution diverges. With them we got appropriate results
"""

import numpy as np

from background import Background
from kernels import (
    B_fff_permu_nb,
    B_pff_permu1_nb,
    B_pff_permu2_nb,
    B_ppf_permu1_nb,
    B_ppf_permu2_nb,
    B_ppp_permu_nb,
    gamma_nb,
    sigma_nb,
)
from matrices import Matrices
from potential import Potential


# Calculate the sum of the permutation. Used in initial.py
def permu_sum(permu):
    return (
        permu(0, 1, 2)
        + permu(0, 2, 1)
        + permu(1, 2, 0)
        + permu(1, 0, 2)
        + permu(2, 0, 1)
        + permu(2, 1, 0)
    )


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
        sub_m = -1
        Q_m = None
        for Q in Q_N:
            Q_m = Q
            N = Q["N"]
            q = Q["Q"]
            a = bkg.a(N)
            ddV = np.asarray(self.pot.ddV(q), dtype=float)
            w, _ = np.linalg.eig(ddV)
            m2 = np.max(w)
            sub_m = m2 - k_min ** 2 / (a ** 2)
            if sub_m > 0:
                break  # The zero-crossing point

        if sub_m < 0:
            return None, None

        N_init = Q_m["N"] - N_pre
        for Q in Q_N:
            if Q["N"] >= N_init:
                Q_m = Q
                break
        return N_init, Q_m

    def find_H_init(self, k_min, Q_N, N_pre):
        bkg = Background(self.pot)
        sub_h = -1
        Q_h = None
        for Q in Q_N:
            Q_h = Q
            N = Q["N"]
            q = Q["Q"]
            a = bkg.a(N)
            H = bkg.H(q)
            sub_h = a * H - k_min
            if sub_h > 0:
                break  # The zero-crossing point

        if sub_h < 0:
            return None, None

        N_init = Q_h["N"] - N_pre
        for Q in Q_N:
            if Q["N"] >= N_init:
                Q_h = Q
                break
        return N_init, Q_h

    def find_init(self, k_min, Q_N, N_pre):
        # Consider the mass and H
        N_init_m, Q_m = self.find_m_init(k_min, Q_N, N_pre)
        N_init_h, Q_h = self.find_H_init(k_min, Q_N, N_pre)

        # Both are smaller than 0, no crossing point found
        if N_init_m is None and N_init_h is None:
            raise ValueError(f"No crossing point found for k={k_min}!!!!!!!")

        use_m = N_init_h is None or N_init_m < N_init_h
        N_init = N_init_m if use_m else N_init_h
        Q_init = Q_m["Q"] if use_m else Q_h["Q"]

        bkg = Background(self.pot)
        mat = Matrices(self.nF, self.pot, bkg, N_init)
        return mat, bkg, N_init, Q_init

    ################################################################################################

    def get_init(self):
        return self.init

    # Implementation of Eq 6.2
    def sigma(self, k):
        # Note for sigma we only use one k. We take the first component
        _, bkg, N_init, Q = self.init
        k_scalar = float(k[0])  # Here!!
        a = bkg.a(N_init)
        H = bkg.H(Q)
        return sigma_nb(k_scalar, a, H, self.nF)

    # The full B matrix
    def B_full(self, k):
        _, bkg, N_init, _ = self.init
        a = bkg.a(N_init)
        nF = self.nF

        B = np.zeros((2, 2, 2, nF, nF, nF), dtype=float)
        # The full B matrix. Note some part of it may be unncessary but we still keep them.
        B[0, 0, 0] = self.B_fff(k)

        B[1, 0, 0] = self.B_pff(k) / a
        B[0, 1, 0] = np.transpose(self.B_pff((k[1], k[0], k[2])) / a, (1, 0, 2))
        B[0, 0, 1] = np.transpose(self.B_pff((k[2], k[0], k[1])) / a, (2, 0, 1))

        B[1, 1, 0] = self.B_ppf(k) / (a ** 2)
        B[1, 0, 1] = np.transpose(self.B_ppf((k[0], k[2], k[1])) / (a ** 2), (0, 2, 1))
        B[0, 1, 1] = np.transpose(self.B_ppf((k[1], k[2], k[0])) / (a ** 2), (1, 2, 0))

        B[1, 1, 1] = self.B_ppp(k) / (a ** 3)
        return B

    # Eq 6.7a
    def B_fff(self, k):
        # Note: we take mat, bkg and N_init as input.
        # The permutation of k1, k2, k3 will not change N_init (only depend on modulus)
        # Hence mat and bkg are not changed as well
        mat, bkg, N_init, Q = self.init
        pi = np.asarray(Q[:, 1], dtype=float)
        H = bkg.H(Q)
        a = bkg.a(N_init)
        Ks = k[0] * k[1] + k[0] * k[2] + k[1] * k[2]

        # Implement the permutation
        def permu(x, y, z):
            # x -> 1, y -> 2, z -> 3
            A_slow = mat.A((k[x], k[y], k[z]), Q, slow=True)
            B = mat.B((k[x], k[y], k[z]), Q)
            C = mat.C((k[x], k[y], k[z]), Q)
            return B_fff_permu_nb(pi, H, a, Ks, k[x], k[y], k[z], A_slow, B, C)

        # Sum all 6 permutations
        return permu_sum(permu) / (4 * a ** 4) / (k[0] * k[1] * k[2] * (k[0] + k[1] + k[2]))

    # Eq 6.7b
    def B_pff(self, k):
        mat, bkg, N_init, Q = self.init
        pi = np.asarray(Q[:, 1], dtype=float)
        H = bkg.H(Q)
        a = bkg.a(N_init)
        kt = k[0] + k[1] + k[2]
        Ks = k[0] * k[1] + k[0] * k[2] + k[1] * k[2]

        # First square bracket
        def permu1(x, y, z):
            A_slow = mat.A((k[x], k[y], k[z]), Q, slow=True)
            C = mat.C((k[x], k[y], k[z]), Q)
            return B_pff_permu1_nb(pi, H, a, k[x], k[y], k[z], A_slow, C)

        # Second square bracket
        def permu2(x, y, z):
            A_slow = mat.A((k[x], k[y], k[z]), Q, slow=True)
            B = mat.B((k[x], k[y], k[z]), Q)
            C = mat.C((k[x], k[y], k[z]), Q)
            return B_pff_permu2_nb(pi, H, a, Ks, kt, k[x], k[y], k[z], A_slow, B, C)

        B_pff = k[0] ** 2 * (k[1] + k[2]) * permu_sum(permu1)
        B_pff += k[0] * permu_sum(permu2)
        B_pff = B_pff * H / (4 * a ** 3) * (1 / ((k[0] * k[1] * k[2]) ** 2 * kt))
        # The coefficient is different. I will use the one in the official code.
        return B_pff

    def B_ppf(self, k):
        mat, bkg, N_init, Q = self.init
        pi = np.asarray(Q[:, 1], dtype=float)
        H = bkg.H(Q)
        a = bkg.a(N_init)
        kt = k[0] + k[1] + k[2]

        # First square bracket
        def permu1(x, y, z):
            A_slow = mat.A((k[x], k[y], k[z]), Q, slow=True)
            B = mat.B((k[x], k[y], k[z]), Q)
            C = mat.C((k[x], k[y], k[z]), Q)
            return B_ppf_permu1_nb(pi, H, a, k[x], k[y], k[z], A_slow, B, C)

        # Second square bracket
        def permu2(x, y, z):
            B = mat.B((k[x], k[y], k[z]), Q)
            return B_ppf_permu2_nb(a, H, k[z], B)

        B_ppf = k[0] ** 2 * k[1] ** 2 * k[2] * permu_sum(permu1)
        B_ppf += k[0] ** 2 * k[1] ** 2 * k[0] * k[1] * k[2] * permu_sum(permu2)  # This is also in the official code.
        B_ppf = B_ppf * 1 / (4 * a ** 4) * (1 / ((k[0] * k[1] * k[2]) ** 2 * kt))
        # NOTE!! In the paper there is a factor of 1/H^2. But in the actual implementation there is not
        # And the a**6 is a**4. I suspect they cancelled aH=1? But Why??
        return B_ppf

    def B_ppp(self, k):
        mat, bkg, N_init, Q = self.init
        pi = np.asarray(Q[:, 1], dtype=float)
        H = bkg.H(Q)
        a = bkg.a(N_init)
        Ks = k[0] * k[1] + k[0] * k[2] + k[1] * k[2]
        kt = k[0] + k[1] + k[2]

        # Implement the permutation
        def permu(x, y, z):
            # x -> 1, y -> 2, z -> 3
            A_slow = mat.A((k[x], k[y], k[z]), Q, slow=True)
            B = mat.B((k[x], k[y], k[z]), Q)
            C = mat.C((k[x], k[y], k[z]), Q)
            return B_ppp_permu_nb(pi, H, a, Ks, kt, k[x], k[y], k[z], A_slow, B, C)

        # Sum all 6 permutations. Again coefficient is different
        return permu_sum(permu) * H / (4 * a ** 3) / (k[0] * k[1] * k[2] * kt)

    # Eq 6.3. The initial condition for tensor modes
    def gamma(self, k):
        _, bkg, N_init, Q = self.init
        a = bkg.a(N_init)
        H = bkg.H(Q)
        # Again we are only using one k
        return gamma_nb(float(k[0]), a, H)

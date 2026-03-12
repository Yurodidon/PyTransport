from background import Background
from matrices import Matrices, N
from potential import Potential
import time
from tqdm import tqdm
import numba

import numpy as np

def rk4_step(y, dy_dt, h):
    # A Runge-Kunta Method for differential equations
    y = np.asarray(y, dtype=float)

    k1 = np.asarray(dy_dt(y), dtype=float)
    k2 = np.asarray(dy_dt(y + 0.5 * h * k1), dtype=float)
    k3 = np.asarray(dy_dt(y + 0.5 * h * k2), dtype=float)
    k4 = np.asarray(dy_dt(y + h * k3), dtype=float)

    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Note, to find the starting point N_init, we need to 'back evolve' the initial condition
class EvolveFields:
    def __init__(self, nF, N0, Q_0, pot: Potential):
        self.nF = nF
        self.N0 = N0
        self.pot = pot
        self.bkg = Background(pot)
        self.Q_0 = np.zeros((nF, 2))

        dV = pot.dV(Q_0)
        H = self.bkg.H(Q_0)

        # set up Q_0
        for i in range(nF):
            self.Q_0[i, 0] = Q_0[i, 0]
            # pi[i] may have been determined from backEvolve
            if Q_0[i, 1] != 0:
                self.Q_0[i, 1] = Q_0[i, 1]
            else: # If not, then use Eq 4.2b and with the slow-roll approximation.
                self.Q_0[i, 1] = -1 * dV[i] / (3 * H)

    # Note N_stop < N_0 as we are going backward
    def backEvolve(self, N_stop, N_step=0.001):
        N0 = self.N0
        bkg = self.bkg
        pot = self.pot

        if N_stop > N0:
            raise "N_stop must be less than N0 in back evolve"

        N = N0
        Q = self.Q_0
        ret = [] # Store the phi and pi at each N as a dictionary for convenience

        # Update the phi and pi by Eq 4.2
        # Note we are still using N as time variable, so we divide by H
        def dQ_dN(Q):
            H = bkg.H(Q)
            dV = pot.dV(Q)
            dQ = np.zeros_like(Q)

            dQ[:, 0] = Q[:, 1] / H
            dQ[:, 1] = (-3 * H * Q[:, 1] - dV) / H
            return dQ

        while N >= N_stop:
            ret.append({'N': N, 'Q': Q})

            new_Q = rk4_step(Q, dQ_dN, -1 * N_step)

            Q = new_Q; N -= N_step

        return ret

    # Note N_stop > N_0 as we are going forward
    def forwardEvolve(self, N_stop, N_step=0.001):
        N0 = self.N0
        bkg = self.bkg
        pot = self.pot

        if N_stop < N0:
            raise "N_stop must be larger than N0 in forward evolve"

        N = N0
        Q = self.Q_0
        ret = [] # Store the phi and pi at each N as a dictionary for convenience

        # Update the phi and pi by Eq 4.2
        # Note we are still using N as time variable, so we divide by H
        def dQ_dN(Q):
            H = bkg.H(Q)
            dV = pot.dV(Q)
            dQ = np.zeros_like(Q)

            dQ[:, 0] = Q[:, 1] / H
            dQ[:, 1] = (-3 * H * Q[:, 1] - dV) / H
            return dQ

        while N <= N_stop:
            ret.append({'N': N, 'Q': Q})

            new_Q = rk4_step(Q, dQ_dN, N_step)

            Q = new_Q; N += N_step

        return ret

# Actually only used once. But still defined as a function
def two_to_one(M, nF):
    S = np.zeros((2 * nF), dtype=float)
    S[:nF] = M[0]
    S[nF:] = M[1]
    return S

# Functions to flatten the matrix for matrix production
def four_to_two(S, nF):
    # S: (2,2,nF,nF) -> (2nF,2nF)
    dim = 2 * nF
    M = np.zeros((dim, dim), dtype=float)
    for a in range(2):
        for b in range(2):
            M[a * nF:(a + 1) * nF, b * nF:(b + 1) * nF] = S[a, b, :, :]
    return M

# Convert back to the shape of sigma
def two_to_four(M, nF):
    # M: (2nF,2nF) -> (2,2,nF,nF)
    S = np.zeros((2, 2, nF, nF), dtype=float)
    for a in range(2):
        for b in range(2):
            S[a, b, :, :] = M[a * nF:(a + 1) * nF, b * nF:(b + 1) * nF]
    return S

# For matrix of (2, 2, 2, nF nF, nF)
def six_to_three(B, nF):
    dim = 2 * nF
    T = np.zeros((dim, dim, dim), dtype=float)
    for ia in range(2):
        for ib in range(2):
            for ic in range(2):
                for a in range(nF):
                    I = ia * nF + a
                    for b in range(nF):
                        J = ib * nF + b
                        for c in range(nF):
                            K = ic * nF + c
                            T[I, J, K] = B[ia, ib, ic, a, b, c]
    return T

# Convert back
def three_to_six(T, nF):
    B = np.zeros((2, 2, 2, nF, nF, nF), dtype=float)
    for ia in range(2):
        for ib in range(2):
            for ic in range(2):
                for a in range(nF):
                    I = ia * nF + a
                    for b in range(nF):
                        J = ib * nF + b
                        for c in range(nF):
                            K = ic * nF + c
                            B[ia, ib, ic, a, b, c] = T[I, J, K]
    return B

# Interpolation for Runge-Kunta method
def interp_on_grid(N, N_grid, Y_grid):
    if N <= N_grid[0]:
        return Y_grid[0]
    if N >= N_grid[-1]:
        return Y_grid[-1]
    j = int(np.searchsorted(N_grid, N))
    N0, N1 = N_grid[j - 1], N_grid[j]
    w = (N - N0) / (N1 - N0)
    return (1.0 - w) * Y_grid[j - 1] + w * Y_grid[j]

# We still use Runge-Kunta methods
class EvolveSigma:
    def __init__(self, nF, pot):
        self.nF = nF
        self.pot = pot

    def forwardEvolve(self, k, Q_N, sigma_re_0, sigma_im_0, N_stop=-114514):
        nF = self.nF
        pot = self.pot
        bkg = Background(pot)

        # Extract arrays for easier indexing
        N_grid = np.array([q['N'] for q in Q_N], dtype=float)
        Q_grid = np.array([q['Q'] for q in Q_N], dtype=float)

        # Find the matrix U2 for given N and Q
        def U_of_NQ(N, Q):
            mat = Matrices(nF, pot, bkg, N)
            return four_to_two(mat.u2(k, Q), nF)

        def dSigma_dN(Sigma_mat, U_mat):
            # Sigma' = U Sigma + Sigma U^T
            return U_mat @ Sigma_mat + Sigma_mat @ U_mat.T

        # Initialize Sigma in flattened form
        Sigma_re = four_to_two(sigma_re_0, nF)
        Sigma_im = four_to_two(sigma_im_0, nF)

        ret = []

        # Main RK4 loop over segments
        for i in range(len(N_grid) - 1):
            N0 = float(N_grid[i])
            N1 = float(N_grid[i + 1])
            h = N1 - N0
            Q0 = Q_grid[i]
            Q1 = Q_grid[i + 1]

            ret.append({
                'N': N0,
                'Q': Q0,
                'sigma_re': two_to_four(Sigma_re, nF),
                'sigma_im': two_to_four(Sigma_im, nF),
            })


            # Midpoint (linear) for Q, and midpoint N
            N_mid = N0 + 0.5 * h
            Q_mid = 0.5 * (Q0 + Q1)  # Extrapolation

            U0 = U_of_NQ(N0, Q0)
            Um = U_of_NQ(N_mid, Q_mid)
            U1 = U_of_NQ(N1, Q1)

            # RK4 for real part
            k1 = dSigma_dN(Sigma_re, U0)
            k2 = dSigma_dN(Sigma_re + 0.5 * h * k1, Um)
            k3 = dSigma_dN(Sigma_re + 0.5 * h * k2, Um)
            k4 = dSigma_dN(Sigma_re + h * k3, U1)
            Sigma_re = Sigma_re + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # RK4 for imaginary part
            k1 = dSigma_dN(Sigma_im, U0)
            k2 = dSigma_dN(Sigma_im + 0.5 * h * k1, Um)
            k3 = dSigma_dN(Sigma_im + 0.5 * h * k2, Um)
            k4 = dSigma_dN(Sigma_im + h * k3, U1)
            Sigma_im = Sigma_im + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Enforce expected symmetries to control drift:
            # Re part symmetric, Im part antisymmetric (in flattened indices)
            # Note: This can prevent the matrix from getting diverged
            Sigma_re = 0.5 * (Sigma_re + Sigma_re.T)
            Sigma_im = 0.5 * (Sigma_im - Sigma_im.T)

            if N_stop != -114514 and N0 > N_stop:
                break

            # Catch NaN to ensure every result is valid
            if not np.isfinite(Sigma_re).all() or not np.isfinite(Sigma_im).all():
                raise FloatingPointError(f"Sigma became non-finite at N={N1}.")

        return ret

# Eq 5.16 for the evolution of B matrix. Use Runge-Kunta as well
class EvolveB:
    def __init__(self, nF, pot):
        self.nF = nF
        self.pot = pot
        self.bkg = Background(pot)

    # ------------------- RHS for Eq. (5.16) (in N-time) -------------------
    @staticmethod
    def dB_dN(B, Ua, Ub, Uc, U3a, U3b, U3c, Ra, Ia, Rb, Ib, Rc, Ic):
        """
        B' = Ua*B + Ub*B + Uc*B  +  source(Sigma_re, Sigma_im)
        with explicit index order as in Eq. (5.16).

        All objects are flattened:
          B: (dim,dim,dim)
          Ua,Ub,Uc: (dim,dim)
          U3*: (dim,dim,dim) with first index = leg acted on
          S*,T*: (dim,dim) = Sigma_re/im for each k-leg
        """
        # Linear terms
        dB = np.einsum('il,ljk->ijk', Ua, B, optimize=True)
        dB += np.einsum('jl,ilk->ijk', Ub, B, optimize=True)
        dB += np.einsum('kl,ijl->ijk', Uc, B, optimize=True)

        # Source terms: (Re Re - Im Im) for each leg, exactly as Eq. (5.16)
        # a-leg source uses Sigma(kb) and Sigma(kc)
        dB += np.einsum('ilm,lj,mk->ijk', U3a, Rb, Rc, optimize=True)
        dB -= np.einsum('ilm,lj,mk->ijk', U3a, Ib, Ic, optimize=True)

        # b-leg source uses Sigma(ka) and Sigma(kc)
        dB += np.einsum('jlm,il,mk->ijk', U3b, Ra, Rc, optimize=True)
        dB -= np.einsum('jlm,il,mk->ijk', U3b, Ia, Ic, optimize=True)

        # c-leg source uses Sigma(ka) and Sigma(kb)
        dB += np.einsum('klm,il,jm->ijk', U3c, Ra, Rb, optimize=True)
        dB -= np.einsum('klm,il,jm->ijk', U3c, Ia, Ib, optimize=True)

        return dB

    def forwardEvolve(self, k, Q_N, Sigma_trajs, B_re_0, N_stop=-114514):
        """
        k : (k1,k2,k3) magnitudes (floats)
        Sigma_trajs : (Sig1, Sig2, Sig3) as described in class docstring
        B_re_0 : ndarray (2,2,2,nF,nF,nF), corresponds to Q_N[0]['N']
        """
        nF = self.nF
        pot = self.pot
        bkg = self.bkg
        k1, k2, k3 = k

        # Background grids
        Nq = np.array([d['N'] for d in Q_N], dtype=float)
        Qg = np.array([d['Q'] for d in Q_N], dtype=float)  # (T,nF,2)

        # Sigma grids for each leg
        def unpack_sigma_traj(Sig):
            Ns = np.array([d['N'] for d in Sig], dtype=float)
            Sre = np.array([four_to_two(d['sigma_re'], nF) for d in Sig], dtype=float)
            Sim = np.array([four_to_two(d['sigma_im'], nF) for d in Sig], dtype=float)

            return Ns, Sre, Sim

        (N1g, S1re_g, S1im_g) = unpack_sigma_traj(Sigma_trajs[0])
        (N2g, S2re_g, S2im_g) = unpack_sigma_traj(Sigma_trajs[1])
        (N3g, S3re_g, S3im_g) = unpack_sigma_traj(Sigma_trajs[2])

        # Build matrices at a given N,Q
        def build_UU3(N, Q):
            mat = Matrices(nF, pot, bkg, N)

            # For u2 we only use one k.
            Ua = four_to_two(mat.u2(np.array([k1, 0.0, 0.0]), Q), nF)
            Ub = four_to_two(mat.u2(np.array([k2, 0.0, 0.0]), Q), nF)
            Uc = four_to_two(mat.u2(np.array([k3, 0.0, 0.0]), Q), nF)

            # For u3 we must pass the triangle of magnitudes in the right ordering
            # a -> 1, b -> 2, c -> 3
            # U3a corresponds to u^a_de(k1,k2,k3)
            U3a = six_to_three(mat.u3((k1, k2, k3), Q), nF)
            # U3b corresponds to u^b_de(k2,k1,k3)
            U3b = six_to_three(mat.u3((k2, k1, k3), Q), nF)
            # U3c corresponds to u^c_de(k3,k1,k2)
            U3c = six_to_three(mat.u3((k3, k1, k2), Q), nF)

            return Ua, Ub, Uc, U3a, U3b, U3c

        # Initial B_re_0 (flattened tensor)
        B = six_to_three(B_re_0, nF)

        ret = [{'N': float(Nq[0]), 'Q': Qg[0], 'B_re': B_re_0}]

        # Segment-wise RK4 using background grid
        # We use interpolation between points
        for i in tqdm(range(len(Nq) - 1)):
            N0, N1 = float(Nq[i]), float(Nq[i + 1])
            h = N1 - N0

            Q0, Q1v = Qg[i], Qg[i + 1]
            Nm = N0 + 0.5 * h
            Qm = 0.5 * (Q0 + Q1v)
            
            ret.append({'N': N1, 'Q': Q1v, 'B_re': three_to_six(B, nF)})

            # Sigma at endpoints/midpoints for each leg (linear interpolation)
            # Where R stands for real. I stands for imaginary
            Ra0 = interp_on_grid(N0, N1g, S1re_g); Ia0 = interp_on_grid(N0, N1g, S1im_g)
            Rb0 = interp_on_grid(N0, N2g, S2re_g); Ib0 = interp_on_grid(N0, N2g, S2im_g)
            Rc0 = interp_on_grid(N0, N3g, S3re_g); Ic0 = interp_on_grid(N0, N3g, S3im_g)

            Ram = interp_on_grid(Nm, N1g, S1re_g); Iam = interp_on_grid(Nm, N1g, S1im_g)
            Rbm = interp_on_grid(Nm, N2g, S2re_g); Ibm = interp_on_grid(Nm, N2g, S2im_g)
            Rcm = interp_on_grid(Nm, N3g, S3re_g); Icm = interp_on_grid(Nm, N3g, S3im_g)

            Ra1 = interp_on_grid(N1, N1g, S1re_g); Ia1 = interp_on_grid(N1, N1g, S1im_g)
            Rb1 = interp_on_grid(N1, N2g, S2re_g); Ib1 = interp_on_grid(N1, N2g, S2im_g)
            Rc1 = interp_on_grid(N1, N3g, S3re_g); Ic1 = interp_on_grid(N1, N3g, S3im_g)

            Ua0, Ub0, Uc0, U3a0, U3b0, U3c0 = build_UU3(N0, Q0)
            Uam, Ubm, Ucm, U3am, U3bm, U3cm = build_UU3(Nm, Qm)
            Ua1, Ub1, Uc1, U3a1, U3b1, U3c1 = build_UU3(N1, Q1v)

            # RK4
            rk1 = self.dB_dN(B, Ua0, Ub0, Uc0, U3a0, U3b0, U3c0, Ra0, Ia0, Rb0, Ib0, Rc0, Ic0)
            rk2 = self.dB_dN(B + 0.5 * h * rk1, Uam, Ubm, Ucm, U3am, U3bm, U3cm, Ram, Iam, Rbm, Ibm, Rcm, Icm)
            rk3 = self.dB_dN(B + 0.5 * h * rk2, Uam, Ubm, Ucm, U3am, U3bm, U3cm, Ram, Iam, Rbm, Ibm, Rcm, Icm)
            rk4 = self.dB_dN(B + h * rk3, Ua1, Ub1, Uc1, U3a1, U3b1, U3c1, Ra1, Ia1, Rb1, Ib1, Rc1, Ic1)

            B = B + (h / 6.0) * (rk1 + 2 * rk2 + 2 * rk3 + rk4)

            if N_stop != -114514 and N0 > N_stop:
                break

            # Check diverges. This is pretty much what I worries the most
            if not np.isfinite(B).all():
                raise FloatingPointError(f"B became non-finite at N={N1}.")

        return ret

# Evolve the tensor mode based on Eq 5.11
class EvolveGamma:
    def __init__(self, nF, pot):
        self.nF = nF
        self.pot = pot
        self.bkg = Background(pot)

    # Note: only the real part matters at the end, but we still consider Imaginary
    def forwardEvolve(self, k_scalar, Q_N, gamma_re_0, gamma_im_0=None):
        if gamma_im_0 is None:
            gamma_im_0 = np.zeros((2, 2), dtype=float)

        Nq = np.array([d['N'] for d in Q_N], dtype=float)
        Qg = np.array([d['Q'] for d in Q_N], dtype=float)

        GRe = np.array(gamma_re_0, dtype=float)
        GIm = np.array(gamma_im_0, dtype=float)

        ret = [{'N': float(Nq[0]), 'Q': Qg[0], 'gamma_re': GRe.copy(), 'gamma_im': GIm.copy()}]

        nF = self.nF
        pot = self.pot
        bkg = self.bkg

        # We still use Runge-Kunta method
        for i in tqdm(range(len(Nq) - 1)):
            N0, N1 = float(Nq[i]), float(Nq[i + 1])
            h = N1 - N0
            Q0, Q1v = Qg[i], Qg[i + 1]
            Nm = N0 + 0.5 * h
            Qm = 0.5 * (Q0 + Q1v)

            def w_N(n, q):
                mat = Matrices(nF, pot, bkg, n)
                return mat.w(k_scalar, q)

            W0 = w_N(N0, Q0)
            Wm = w_N(Nm, Qm)
            W1 = w_N(N1, Q1v)

            def dG(G, W):
                return W @ G + G @ W.T

            # RK4 for Re
            rk1 = dG(GRe, W0)
            rk2 = dG(GRe + 0.5 * h * rk1, Wm)
            rk3 = dG(GRe + 0.5 * h * rk2, Wm)
            rk4 = dG(GRe + h * rk3, W1)
            GRe = GRe + (h / 6.0) * (rk1 + 2 * rk2 + 2 * rk3 + rk4)

            # RK4 for Im
            rk1 = dG(GIm, W0)
            rk2 = dG(GIm + 0.5 * h * rk1, Wm)
            rk3 = dG(GIm + 0.5 * h * rk2, Wm)
            rk4 = dG(GIm + h * rk3, W1)
            GIm = GIm + (h / 6.0) * (rk1 + 2 * rk2 + 2 * rk3 + rk4)

            # Enforce symmetry properties (analogous to Eq. (5.7))
            GRe = 0.5 * (GRe + GRe.T)
            GIm = 0.5 * (GIm - GIm.T)

            if not np.isfinite(GRe).all() or not np.isfinite(GIm).all():
                raise FloatingPointError(f"Gamma became non-finite at N={N1}.")

            ret.append({'N': N1, 'Q': Q1v, 'gamma_re': GRe.copy(), 'gamma_im': GIm.copy()})

        return ret

# Implement Eq 7.6. To get the final power spectra
class Spectra:
    def __init__(self, nF, pot: Potential):
        self.nF = nF
        self.pot = pot
        self.bkg = Background(pot)

    # Evalute the 2pf spectrum P(k).
    def spectrum(self, Q, sigma_re):
        n = N(self.nF, self.pot)
        N_a = two_to_one(n.N_a(Q), self.nF)
        sigma = four_to_two(sigma_re, self.nF)

        return float(N_a @ sigma @ N_a)

    # Note now k and sigma a 3 tuple
    def bispectrum(self, k, Q, B, sigma):
        n = N(self.nF, self.pot)
        N_a = two_to_one(n.N_a(Q), self.nF)
        sigma1, sigma2, sigma3 = map(four_to_two, sigma, (self.nF, self.nF, self.nF))
        B = six_to_three(B, self.nF)

        bispec = np.einsum("a,b,c,abc->", N_a, N_a, N_a, B, optimize=True)

        def permu(k_in, Sa, Sb):
            N_ab = four_to_two(n.N_ab(Q, k_in), self.nF)
            return np.einsum('a,b,cd,ac,bd->', N_a, N_a, N_ab, Sa, Sb, optimize=True)

        k1, k2, k3 = k
        bispec += permu((k3, k1, k2), sigma1, sigma2)
        bispec += permu((k1, k2, k3), sigma2, sigma3)
        bispec += permu((k2, k3, k1), sigma3, sigma1)

        return float(bispec)






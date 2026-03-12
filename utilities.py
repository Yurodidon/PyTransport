import numpy as np

# Inner product of a function
def inner(vec1, vec2):
    s = 0
    for i in range(len(vec1)):
        s += vec1[i] * vec2[i]
    return s

def delta(a, b):
    return 1 if a == b else 0

# Calculate dpi with Eq 4.2b
def calc_dpi(Q, bkg, pot):
    phi = Q[:, 0]
    pi = Q[:, 1]
    nF = len(phi)
    H = bkg.H(Q)
    dV = pot.dV(Q)
    res = np.zeros(nF)

    for i in range(nF):
        res[i] = -3 * H * pi[i] - dV[i]

    return res

# A small function to find the horizon exit time N for a given k
def calc_N_exit(k, Q_N, bkg):
    def sign(x): return 1 if x < 0 else -1
    s = 0
    for i in Q_N:
        N = i['N']; Q = i['Q']
        temp = k - bkg.a(N) * bkg.H(Q)
        if s != 0 and sign(temp) != s:
            return N
        s = sign(temp)

    raise RuntimeError(f"No N found for k value of {k}")

# phi_0 is a (nF, 2) vector but [:, 1]=0; only phi term filled
def find_Q_0(nF, phi_0, pot, bkg, N):
    Q_0 = np.zeros((nF, 2))
    dV = pot.dV(phi_0)
    H = bkg.H(phi_0)

    for i in range(nF):
        Q_0[i, 0] = phi_0[i][0]
        Q_0[i, 1] = -1 * dV[i] / (3 * H)

    return Q_0
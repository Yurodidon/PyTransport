"""
To use numba on our code, we need to extract the numerical parts
Note: In the original implementation, lots of Python features were used
So to use njit, we need to have all these in a new file
After experiments, the speed can be increased by roughly 4x

This file is written with aids of Codex
"""

import numpy as np
from numba import njit


def as_scalar(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[0])


def as_vector(value, n):
    return np.asarray(value, dtype=float).reshape(n)


def as_matrix(value, n):
    return np.asarray(value, dtype=float).reshape(n, n)


def as_tensor3(value, n):
    return np.asarray(value, dtype=float).reshape(n, n, n)


@njit(cache=True)
def inner_nb(vec1, vec2):
    s = 0.0
    for i in range(len(vec1)):
        s += vec1[i] * vec2[i]
    return s


@njit(cache=True)
def delta_nb(a, b):
    return 1.0 if a == b else 0.0


@njit(cache=True)
def calc_dpi_nb(pi, H, dV):
    nF = len(pi)
    res = np.zeros(nF, dtype=np.float64)
    for i in range(nF):
        res[i] = -3.0 * H * pi[i] - dV[i]
    return res


@njit(cache=True)
def H_nb(Q, V):
    return np.sqrt((1.0 / 6.0) * inner_nb(Q[:, 1], Q[:, 1]) + (1.0 / 3.0) * V)


@njit(cache=True)
def epsilon_nb(Q, H):
    return -inner_nb(Q[:, 1], Q[:, 1]) / (2.0 * H * H)


@njit(cache=True)
def xi_nb(pi, dpi, H):
    nF = len(pi)
    xi = np.zeros(nF, dtype=np.float64)
    pipi = inner_nb(pi, pi)
    for i in range(nF):
        xi[i] = 2.0 * dpi[i] + (pi[i] / H) * pipi
    return xi


@njit(cache=True)
def A_unsym_nb(k_i, k_j, k_m, dddV, ddV, V, H, a_scale, pi, xi, slow):
    nF = len(pi)
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    k_dot_k = (k_i * k_i - k_j * k_j - k_m * k_m) / 2.0
    kk = k_j * k_j * k_m * k_m

    for a in range(nF):
        for b in range(nF):
            for g in range(nF):
                value = (-1.0 / 3.0) * dddV[a, b, g]
                value -= (pi[a] * ddV[b, g]) / (2.0 * H)
                value += (pi[a] * pi[b] * xi[g]) / (8.0 * H * H)
                value += (pi[a] * xi[b] * xi[g]) / (32.0 * H ** 3) * (1.0 - (k_dot_k * k_dot_k) / kk)
                value += (pi[a] * pi[b] * pi[g]) / (8.0 * H ** 3) * 2.0 * V
                if not slow and b == g:
                    value += (pi[a] / (2.0 * H)) * (k_dot_k / (a_scale * a_scale))
                out[a, b, g] = value
    return out


@njit(cache=True)
def symmetrize_A_nb(A012, A021, A102, A120, A201, A210):
    nF = A012.shape[0]
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    for a in range(nF):
        for b in range(nF):
            for g in range(nF):
                out[a, b, g] = (
                    A012[a, b, g]
                    + A021[a, g, b]
                    + A102[b, a, g]
                    + A120[b, g, a]
                    + A201[g, a, b]
                    + A210[g, b, a]
                ) / 6.0
    return out


@njit(cache=True)
def B_unsym_nb(k_i, k_j, k_m, H, pi, xi):
    nF = len(pi)
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    k_dot_k_1 = (k_i * k_i - k_j * k_j - k_m * k_m) / 2.0
    k_dot_k_2 = (k_m * k_m - k_i * k_i - k_j * k_j) / 2.0
    kk = k_j * k_j * k_m * k_m

    for a in range(nF):
        for b in range(nF):
            for g in range(nF):
                value = (pi[a] * pi[b] * pi[g]) / (4.0 * H * H)
                value -= (pi[a] * xi[b] * pi[g]) / (8.0 * H ** 3) * (1.0 - (k_dot_k_1 * k_dot_k_1) / kk)
                if b == g:
                    value -= (xi[a] / (2.0 * H)) * (k_dot_k_2 / (k_i * k_i))
                out[a, b, g] = value
    return out


@njit(cache=True)
def symmetrize_BC_nb(T012, T102):
    nF = T012.shape[0]
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    for a in range(nF):
        for b in range(nF):
            for g in range(nF):
                out[a, b, g] = (T012[a, b, g] + T102[b, a, g]) / 2.0
    return out


@njit(cache=True)
def C_unsym_nb(k_i, k_j, k_m, H, pi):
    nF = len(pi)
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    k_dot_k_1 = (k_m * k_m - k_i * k_i - k_j * k_j) / 2.0
    k_dot_k_2 = (k_j * k_j - k_i * k_i - k_m * k_m) / 2.0
    kk = k_i * k_i * k_j * k_j

    for a in range(nF):
        for b in range(nF):
            for g in range(nF):
                value = 0.0
                if a == b:
                    value -= pi[g] / (2.0 * H)
                value += (pi[a] * pi[b] * pi[g]) / (8.0 * H ** 3) * (1.0 - (k_dot_k_1 * k_dot_k_1) / kk)
                if b == g:
                    value += (pi[a] / H) * (k_dot_k_2 / (k_i * k_i))
                out[a, b, g] = value
    return out


@njit(cache=True)
def M_nb(k_scalar, ddV, epsilon, H, a, pi, dpi):
    nF = len(pi)
    M = np.zeros((nF, nF), dtype=np.float64)
    m_mat = np.zeros((nF, nF), dtype=np.float64)
    k_term = (k_scalar * k_scalar) / (a * a)

    for i in range(nF):
        for j in range(nF):
            m = ddV[i, j] - (3.0 - epsilon) * pi[i] * pi[j] - (pi[i] * dpi[j] + pi[j] * dpi[i]) / H
            M[i, j] = -k_term * delta_nb(i, j) - m
            m_mat[i, j] = m
    return M, m_mat


@njit(cache=True)
def u2_nb(m_tiled, H):
    nF = m_tiled.shape[0]
    out = np.zeros((2, 2, nF, nF), dtype=np.float64)
    for a in range(nF):
        for b in range(nF):
            out[0, 1, a, b] = delta_nb(a, b)
            out[1, 0, a, b] = m_tiled[a, b]
            out[1, 1, a, b] = -3.0 * H * delta_nb(a, b)
    return out / H


@njit(cache=True)
def u3_nb(A, B1, B2, B3, C1, C2, C3, H):
    nF = A.shape[0]
    out = np.zeros((2, 2, 2, nF, nF, nF), dtype=np.float64)
    for a in range(nF):
        for b in range(nF):
            for g in range(nF):
                out[0, 0, 0, a, b, g] = -B1[b, g, a]
                out[0, 1, 0, a, b, g] = -C1[a, b, g]
                out[1, 0, 0, a, b, g] = 3.0 * A[a, b, g]
                out[1, 1, 0, a, b, g] = B2[a, g, b]

                out[0, 0, 1, a, b, g] = -C2[a, g, b]
                out[0, 1, 1, a, b, g] = 0.0
                out[1, 0, 1, a, b, g] = B3[a, b, g]
                out[1, 1, 1, a, b, g] = C3[g, b, a]
    return out / H


@njit(cache=True)
def w_nb(k_scalar, H, a):
    out = np.zeros((2, 2), dtype=np.float64)
    out[0, 1] = 1.0
    out[1, 0] = -(k_scalar * k_scalar) / (a * a)
    out[1, 1] = -3.0 * H
    return out / H


@njit(cache=True)
def N_a_nb(pi, H, epsilon):
    nF = len(pi)
    out = np.zeros((2, nF), dtype=np.float64)
    for i in range(nF):
        out[0, i] = pi[i] / (2.0 * H * epsilon)
    return out


@njit(cache=True)
def N_ab_nb(pi, H, dV, epsilon, k0, k1, k2):
    nF = len(pi)
    out = np.zeros((2, 2, nF, nF), dtype=np.float64)
    k_dot_k = (k0 * k0 - k1 * k1 - k2 * k2) / 2.0
    dV_dot_pi = inner_nb(dV, pi)

    for i in range(nF):
        for j in range(nF):
            out[0, 0, i, j] = pi[i] * pi[j] * (-1.5 + 9.0 / (2.0 * epsilon) + 3.0 * dV_dot_pi / (4.0 * epsilon * epsilon * H ** 3))
            out[0, 1, i, j] = 3.0 * pi[i] * pi[j] / (H * epsilon) - (3.0 * H / (k0 * k0)) * (k_dot_k + k2 * k2) * delta_nb(i, j)
            out[1, 0, i, j] = 3.0 * pi[i] * pi[j] / (H * epsilon) - (3.0 * H / (k0 * k0)) * (k_dot_k + k1 * k1) * delta_nb(i, j)
            out[0, 1, i, j] /= 2.0
            out[1, 0, i, j] /= 2.0
    return out / (3.0 * H * H * epsilon)


@njit(cache=True)
def sigma_nb(k_scalar, a, H, nF):
    sigma_re = np.zeros((2, 2, nF, nF), dtype=np.float64)
    sigma_im = np.zeros((2, 2, nF, nF), dtype=np.float64)
    for i in range(nF):
        sigma_re[0, 0, i, i] = a
        sigma_re[0, 1, i, i] = -a * H
        sigma_re[1, 0, i, i] = -a * H
        sigma_re[1, 1, i, i] = (k_scalar * k_scalar) / a

        sigma_im[0, 1, i, i] = k_scalar
        sigma_im[1, 0, i, i] = -k_scalar

    norm = 2.0 * a ** 3 * k_scalar
    return sigma_re / norm, sigma_im / norm


@njit(cache=True)
def B_fff_permu_nb(pi, H, a, Ks, kx, ky, kz, A_slow, B, C):
    nF = len(pi)
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    k_dot_k = (kx * kx - ky * ky - kz * kz) / 2.0
    coeff = ((kx + ky) * kz / (kx * ky)) - Ks / (kx * ky)

    for i in range(nF):
        for j in range(nF):
            for m in range(nF):
                value = pi[i] * delta_nb(j, m) * k_dot_k / (4.0 * H)
                value -= 0.5 * C[i, j, m] * kx * ky
                value += 0.5 * a * a * A_slow[i, j, m]
                value += 0.5 * a * a * H * B[i, j, m] * coeff
                out[i, j, m] = value
    return out


@njit(cache=True)
def B_pff_permu1_nb(pi, H, a, kx, ky, kz, A_slow, C):
    nF = len(pi)
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    k_dot_k = (kx * kx - ky * ky - kz * kz) / 2.0
    for i in range(nF):
        for j in range(nF):
            for m in range(nF):
                value = pi[i] * delta_nb(j, m) * k_dot_k / (4.0 * H)
                value += 0.5 * a * a * A_slow[i, j, m]
                value -= 0.5 * C[i, j, m] * kx * ky
                out[i, j, m] = value
    return out


@njit(cache=True)
def B_pff_permu2_nb(pi, H, a, Ks, kt, kx, ky, kz, A_slow, B, C):
    nF = len(pi)
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    k_dot_k = (kx * kx - ky * ky - kz * kz) / 2.0
    for i in range(nF):
        for j in range(nF):
            for m in range(nF):
                value = 0.5 * kx * kx * ky * ky * (1.0 + kz / kt) * C[i, j, m]
                value += B[i, j, m] * kx * ky * kz * kz / (2.0 * H)
                value -= pi[i] * delta_nb(j, m) * (Ks + kx * ky * kz / kt) * k_dot_k / (4.0 * H)
                value -= 0.5 * a * a * A_slow[i, j, m] * (Ks - kx * ky * kz / kt)
                out[i, j, m] = value
    return out


@njit(cache=True)
def B_ppf_permu1_nb(pi, H, a, kx, ky, kz, A_slow, B, C):
    nF = len(pi)
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    k_dot_k = (kx * kx - ky * ky - kz * kz) / 2.0
    coeff = (kx + ky) * kz / (kx * ky)
    for i in range(nF):
        for j in range(nF):
            for m in range(nF):
                value = 0.5 * C[i, j, m] * kx * ky
                value -= pi[i] * delta_nb(j, m) * k_dot_k / (4.0 * H)
                value -= 0.5 * a * a * A_slow[i, j, m]
                value -= 0.5 * a * a * H * B[i, j, m] * coeff
                out[i, j, m] = value
    return out


@njit(cache=True)
def B_ppf_permu2_nb(a, H, kz, B):
    nF = B.shape[0]
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    value_coeff = 0.5 * a * a * H * kz
    for i in range(nF):
        for j in range(nF):
            for m in range(nF):
                out[i, j, m] = value_coeff * B[i, j, m]
    return out


@njit(cache=True)
def B_ppp_permu_nb(pi, H, a, Ks, kt, kx, ky, kz, A_slow, B, C):
    nF = len(pi)
    out = np.zeros((nF, nF, nF), dtype=np.float64)
    k_dot_k = (kx * kx - ky * ky - kz * kz) / 2.0
    for i in range(nF):
        for j in range(nF):
            for m in range(nF):
                value = pi[i] * delta_nb(j, m) * (Ks + kx * ky * kz / kt) * k_dot_k / (4.0 * H)
                value -= B[i, j, m] * (kx * kx * ky * kz) / (2.0 * H)
                value -= 0.5 * kx * kx * ky * ky * (1.0 + kz / kt) * C[i, j, m]
                value += 0.5 * a * a * A_slow[i, j, m] * (Ks - (kx * ky * kz) / kt)
                out[i, j, m] = value
    return out


@njit(cache=True)
def gamma_nb(k_scalar, a, H):
    out = np.zeros((2, 2), dtype=np.float64)
    out[0, 0] = a
    out[0, 1] = -a * H
    out[1, 0] = -a * H
    out[1, 1] = (k_scalar * k_scalar) / a
    return out / (2.0 * a ** 3 * k_scalar)


@njit(cache=True)
def two_to_one_nb(M, nF):
    out = np.zeros(2 * nF, dtype=np.float64)
    out[:nF] = M[0]
    out[nF:] = M[1]
    return out


@njit(cache=True)
def four_to_two_nb(S, nF):
    dim = 2 * nF
    M = np.zeros((dim, dim), dtype=np.float64)
    for a in range(2):
        for b in range(2):
            for i in range(nF):
                for j in range(nF):
                    M[a * nF + i, b * nF + j] = S[a, b, i, j]
    return M


@njit(cache=True)
def two_to_four_nb(M, nF):
    S = np.zeros((2, 2, nF, nF), dtype=np.float64)
    for a in range(2):
        for b in range(2):
            for i in range(nF):
                for j in range(nF):
                    S[a, b, i, j] = M[a * nF + i, b * nF + j]
    return S


@njit(cache=True)
def six_to_three_nb(B, nF):
    dim = 2 * nF
    T = np.zeros((dim, dim, dim), dtype=np.float64)
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


@njit(cache=True)
def three_to_six_nb(T, nF):
    B = np.zeros((2, 2, 2, nF, nF, nF), dtype=np.float64)
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


@njit(cache=True)
def dSigma_dN_nb(Sigma_mat, U_mat):
    return U_mat @ Sigma_mat + Sigma_mat @ U_mat.T


@njit(cache=True)
def symmetrize_sigma_re_nb(Sigma_re):
    return 0.5 * (Sigma_re + Sigma_re.T)


@njit(cache=True)
def symmetrize_sigma_im_nb(Sigma_im):
    return 0.5 * (Sigma_im - Sigma_im.T)


@njit(cache=True)
def dB_dN_nb(B, Ua, Ub, Uc, U3a, U3b, U3c, Ra, Ia, Rb, Ib, Rc, Ic):
    dim = B.shape[0]
    out = np.zeros_like(B)

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                value = 0.0
                for l in range(dim):
                    value += Ua[i, l] * B[l, j, k]
                    value += Ub[j, l] * B[i, l, k]
                    value += Uc[k, l] * B[i, j, l]
                for l in range(dim):
                    for m in range(dim):
                        value += U3a[i, l, m] * (Rb[l, j] * Rc[m, k] - Ib[l, j] * Ic[m, k])
                        value += U3b[j, l, m] * (Ra[i, l] * Rc[m, k] - Ia[i, l] * Ic[m, k])
                        value += U3c[k, l, m] * (Ra[i, l] * Rb[j, m] - Ia[i, l] * Ib[j, m])
                out[i, j, k] = value
    return out


@njit(cache=True)
def dG_dN_nb(G, W):
    return W @ G + G @ W.T


@njit(cache=True)
def symmetrize_G_re_nb(G):
    return 0.5 * (G + G.T)


@njit(cache=True)
def symmetrize_G_im_nb(G):
    return 0.5 * (G - G.T)

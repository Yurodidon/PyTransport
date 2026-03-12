# PyTransport (My implementation)
My implementation of the PyTransport Algorithm (https://doi.org/10.48550/arXiv.1609.00379)

The code is generally finished in Python. When actually running, please use the numba version (roughly 4x faster)

### Important Note: In the official code, there are several inconsistencies with the paper
Manifested in
- The $N_{ab}$ matrix, off by a factor of two.
- The initial B matrix, pff, ppf, ppp are off by several factors of $H$.
- For ppf the k (wavenumber) dependence is different.

I followed the official code, otherwise divergence occurs.

See the spectrum.ipynb as an example. Where I calculated the spectrum of Equation (9.7) in the paper.

The general pattern matches, but the accuracy can be better with a denser sampling.


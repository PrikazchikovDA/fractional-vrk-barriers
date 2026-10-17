"""Reduced system for s=5 under the fractional simplifying assumptions
  C_alpha(1) on stages 3..5  and  annihilated second column (b2 = (b^T A)_2 = (b^T C A)_2 = (b^T A^2)_2 = 0).
Unknowns: p = (c3, c4, c5, a43, a53, a54); b = (b1, b3, b4, b5) is fixed by D0..D3 (quadrature on nodes 0,c3,c4,c5,
exact for degree <= 3 with weight (1-x)^{alpha-1}/Gamma(alpha)). c2 drops out (w_i = a_i2 c2 is determined).
Equations (7):
  E1 = sum b_i w_i, E2 = sum b_i c_i w_i, E3 = b4 a43 w3 + b5 (a53 w3 + a54 w4)          [column 2]
  E4 = b^T dt2, E5 = b^T (c . dt2), E6 = b^T dt3, E7 = b4 a43 dt2_3 + b5 (a53 dt2_3 + a54 dt2_4)
with w3 = g1 c3^{1+a}, w4 = g1 c4^{1+a} - a43 c3, w5 = g1 c5^{1+a} - a53 c3 - a54 c4 and
dt_k,i = sum_{j>=3} a_ij c_j^k - g_k c_i^{k+a} (i = 3,4,5)."""
import numpy as np
from math import factorial
from scipy.special import gamma as G

def gam(k, a): return factorial(k)/G(k+a+1)

def weights(c3, c4, c5, a):
    X = np.array([0.0, c3, c4, c5]); M = np.vstack([X**k for k in range(4)]); M[0] = 1.0
    return np.linalg.solve(M, [gam(k, a) for k in range(4)])     # (b1, b3, b4, b5)

def residual(p, a):
    c3, c4, c5, a43, a53, a54 = p
    b1, b3, b4, b5 = weights(c3, c4, c5, a); b = np.array([b3, b4, b5]); c = np.array([c3, c4, c5])
    g1 = gam(1, a)
    w = np.array([g1*c3**(1+a), g1*c4**(1+a) - a43*c3, g1*c5**(1+a) - a53*c3 - a54*c4])
    def dt(k):
        gk = gam(k, a)
        return np.array([-gk*c3**(k+a), a43*c3**k - gk*c4**(k+a), a53*c3**k + a54*c4**k - gk*c5**(k+a)])
    d2, d3 = dt(2), dt(3)
    return np.array([b @ w, b @ (c*w), b4*a43*w[0] + b5*(a53*w[0] + a54*w[1]),
                     b @ d2, b @ (c*d2), b @ d3, b4*a43*d2[0] + b5*(a53*d2[0] + a54*d2[1])])

def full_tableau(p, a, c2):
    """Rebuild (c, A, b) of the 5-stage method for a given free node c2."""
    c3, c4, c5, a43, a53, a54 = p; b1, b3, b4, b5 = weights(c3, c4, c5, a); g1 = gam(1, a)
    c = np.array([0, c2, c3, c4, c5]); A = np.zeros((5, 5))
    A[2, 1] = g1*c3**(1+a)/c2
    A[3, 1] = (g1*c4**(1+a) - a43*c3)/c2; A[3, 2] = a43
    A[4, 1] = (g1*c5**(1+a) - a53*c3 - a54*c4)/c2; A[4, 2] = a53; A[4, 3] = a54
    for i in range(1, 5): A[i, 0] = c[i]**a/G(a+1) - A[i, 1:i].sum()
    return c, A, np.array([b1, 0.0, b3, b4, b5])

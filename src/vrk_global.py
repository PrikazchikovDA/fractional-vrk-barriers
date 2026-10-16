"""Explicit Volterra (Pouzet/Lubich-type) RK for Caputo FDE  D^a y = f(t,y), y(0)=y0,
in the form y = y0 + J^a f.  Lag term over past intervals by product integration of the
interpolating polynomial through the stage values (nodes c_j) on each past interval."""
import numpy as np
from math import comb
from scipy.special import gamma as G
from numpy.polynomial.legendre import leggauss

def ML(z, a, b, K=300): return sum(z**k / G(a*k + b) for k in range(K))

def moments(x, a, kmax):
    """M_k(x) = (1/G(a)) int_0^1 (x-th)^(a-1) th^k dth, x>=1"""
    M = np.zeros(kmax+1)
    if x < 2:
        for k in range(kmax+1):
            M[k] = sum(comb(k, l) * x**(k-l) * (-1)**l * (x**(a+l) - (x-1)**(a+l)) / (a+l) for l in range(k+1))
    else:
        z, w = leggauss(16); th = (z+1)/2; w = w/2
        for k in range(kmax+1): M[k] = np.sum(w * (x-th)**(a-1) * th**k)
    return M / G(a)

def lagrange_coeffs(c):
    s = len(c); C = np.zeros((s, s))  # C[j,k]: coeff of th^k in L_j
    for j in range(s):
        p = np.poly1d([1.0])
        for m in range(s):
            if m != j: p = p * np.poly1d([1.0, -c[m]]) / (c[j]-c[m])
        C[j, :len(p.c)] = p.c[::-1]
    return C

def solve(tab, a, f, y0, T, N):
    c, A, b = (np.array(t, float) for t in tab); s = len(c); h = T/N
    Lc = lagrange_coeffs(c)
    targets = list(c) + [1.0]
    W = np.zeros((s+1, N+1, s))       # W[i,d,j]: weight of F_{n-d, j} in lag at node i of step n
    for i, ci in enumerate(targets):
        for d in range(1, N+1):
            W[i, d] = Lc @ moments(d + ci, a, s-1)
    F = np.zeros((N, s)); y = y0
    for n in range(N):
        if n > 0:
            Fp = F[n-1::-1][:n]                              # F_{n-1}, F_{n-2}, ...
            lag = y0 + h**a * np.einsum('idj,dj->i', W[:, 1:n+1, :], Fp)
        else:
            lag = np.full(s+1, float(y0))
        for i in range(s):
            Yi = lag[i] + h**a * (A[i, :i] @ F[n, :i])
            F[n, i] = f(n*h + c[i]*h, Yi)
        y = lag[s] + h**a * (b @ F[n])
    return y

def tab2(a, c2=1.0):
    b2 = 1/(G(a+2)*c2); return [0, c2], [[0, 0], [c2**a/G(a+1), 0]], [1/G(a+1)-b2, b2]

def tab3A(a, c2=0.5, c3=1.0):
    # kills D0, D1, B1, B2 (a32 = 0); obstruction D2 at h^{2+a}
    b2 = 1/(G(a+2)*(c2 - c2**(1+a)*c3/c3**(1+a)))
    b3 = -b2*c2**(1+a)/c3**(1+a)
    A = [[0, 0, 0], [c2**a/G(a+1), 0, 0], [c3**a/G(a+1), 0, 0]]
    return [0, c2, c3], A, [1/G(a+1)-b2-b3, b2, b3]

def tab3B(a, c2=0.5, c3=1.0):
    # kills D0, D1, D2, B1; obstruction B2 at h^{1+3a}
    V = np.array([[1, 1, 1], [0, c2, c3], [0, c2**2, c3**2]])
    b = np.linalg.solve(V, [1/G(a+1), 1/G(a+2), 2/G(a+3)])
    a32 = (b[1]*c2**(1+a) + b[2]*c3**(1+a)) / (G(a+2)*b[2]*c2)
    A = [[0, 0, 0], [c2**a/G(a+1), 0, 0], [c3**a/G(a+1)-a32, a32, 0]]
    return [0, c2, c3], A, list(b)

if __name__ == '__main__':
    import sys
    Ns = [40, 80, 160, 320, 640]
    for name, tf in [('2-stage', tab2), ('3-stage A', tab3A), ('3-stage B', tab3B)]:
        print(name)
        for a in [0.3, 0.5, 0.7, 0.9]:
            f = lambda t, y, a=a: -y + t**(4-a)/G(5-a); ex = ML(-1.0, a, 5)
            errs = [abs(solve(tf(a), a, f, 0.0, 1.0, N) - ex) for N in Ns]
            rates = [np.log2(errs[i]/errs[i+1]) for i in range(len(Ns)-1)]
            print(f'  a={a}: ' + ' '.join(f'{e:.2e}' for e in errs) + '  rates ' + ' '.join(f'{r:.2f}' for r in rates), flush=True)

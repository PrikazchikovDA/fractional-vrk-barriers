"""Same explicit VRK stages, but lag term from ACCEPTED grid values F_m=f(t_m,y_m),
piecewise-quadratic product integration (nodes t_{m-1},t_m,t_{m+1}); avoids stage-error leakage."""
import numpy as np, sys

from vrk_global import moments, lagrange_coeffs, ML, tab2, tab3A, tab3B
from scipy.special import gamma as G
def solve_grid(tab, a, f, y0, T, N):
    c, A, b = (np.array(t, float) for t in tab); s = len(c); h = T/N
    targets = list(c) + [1.0]
    Lcen = lagrange_coeffs(np.array([-1.0, 0.0, 1.0])); Lfw = lagrange_coeffs(np.array([0.0, 1.0, 2.0])); Llin = lagrange_coeffs(np.array([0.0, 1.0]))
    Wc = np.array([[Lcen @ moments(d+ci, a, 2) for d in range(N+1)] if True else None for ci in targets])  # [i,d,3], d>=1 used
    Wf = np.array([[Lfw @ moments(d+ci, a, 2) for d in range(N+1)] for ci in targets])
    Wl = np.array([Llin @ moments(1+ci, a, 1) for ci in targets])
    Fg = np.zeros(N+1); y = np.zeros(N+1); y[0] = y0; Fg[0] = f(0.0, y0)
    for n in range(N):
        lag = np.full(s+1, float(y0))
        if n == 1:
            lag += h**a * Wl @ Fg[0:2]
        elif n >= 2:
            lag += h**a * Wf[:, n, :] @ Fg[0:3]                      # interval m=0, d=n
            for i in range(s+1):                                     # intervals m=1..n-1, d=n-m
                d = n - np.arange(1, n)
                M = np.stack([Fg[0:n-1], Fg[1:n], Fg[2:n+1]], 1)    # rows m=1..n-1: F_{m-1},F_m,F_{m+1}
                lag[i] += h**a * np.sum(Wc[i, d, :] * M)
        Fs = np.zeros(s)
        for i in range(s):
            Fs[i] = f(n*h + c[i]*h, lag[i] + h**a * (A[i, :i] @ Fs[:i]))
        y[n+1] = lag[s] + h**a * (b @ Fs); Fg[n+1] = f((n+1)*h, y[n+1])
    return y[-1]
if __name__ == '__main__':
    Ns = [40, 80, 160, 320, 640]
    for name, tf in [('2-stage', tab2), ('3-stage A', tab3A), ('3-stage B', tab3B)]:
        print(name)
        for a in [0.3, 0.5, 0.7, 0.9]:
            f = lambda t, y, a=a: -y + t**(4-a)/G(5-a); ex = ML(-1.0, a, 5)
            errs = [abs(solve_grid(tf(a), a, f, 0.0, 1.0, N) - ex) for N in Ns]
            print(f'  a={a}: ' + ' '.join(f'{e:.2e}' for e in errs) + '  rates ' + ' '.join(f'{np.log2(errs[i]/errs[i+1]):.2f}' for i in range(len(Ns)-1)), flush=True)

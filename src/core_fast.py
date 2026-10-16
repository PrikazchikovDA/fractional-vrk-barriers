"""Linear time-varying core L_s (subset of S_s), fast matrix form.
 D0, D1, D2; B_m = b A^{m-1} d1 (m=1..s-1); g2-chain b A^q d2 (q=0..s-4);
 (f_y)'-insertions b A^j C A^{q-j} d1 (q=0..s-4, j=0..q)  [vector form; 'merged' sums over j]."""
import numpy as np, sys
from scipy.special import gamma as G
from scipy.optimize import least_squares
def unpack(v, s, a):
    c = np.concatenate([[0.0], v[:s-1]]); A = np.zeros((s, s)); idx = s-1
    for i in range(1, s):
        for j in range(1, i): A[i, j] = v[idx]; idx += 1
        A[i, 0] = c[i]**a/G(a+1) - A[i, 1:i].sum()
    return c, A, v[idx:idx+s]
def nunk(s): return (s-1) + (s-1)*(s-2)//2 + s
def core(v, s, a, merged=False, use_g2=True, use_ins=True):
    c, A, b = unpack(v, s, a); C = np.diag(c)
    d1 = A@c - c**(1+a)/G(a+2); d2 = A@c**2 - 2*c**(2+a)/G(a+3)
    r = [b.sum()-1/G(a+1), b@c-1/G(a+2), b@c**2-2/G(a+3)]
    P = np.eye(s)
    for m in range(1, s): r.append(b@P@d1); P = P@A
    for q in range(0, s-3):
        if use_g2: r.append(b@np.linalg.matrix_power(A, q)@d2)
        if use_ins:
            terms = [b@np.linalg.matrix_power(A, j)@C@np.linalg.matrix_power(A, q-j)@d1 for j in range(q+1)]
            r += [sum(terms)] if merged else terms
    return np.array(r)
def best(s, a, tries=15, **kw):
    rng=np.random.default_rng(0); n=nunk(s); lb=np.full(n,-np.inf); lb[:s-1]=1e-8; bst=None
    for t in range(tries):
        x0=rng.uniform(-1,2,n); x0[:s-1]=rng.uniform(0.05,1.5,s-1)
        r=least_squares(lambda v: core(v,s,a,**kw),x0,bounds=(lb,np.inf),xtol=1e-15,ftol=1e-15,gtol=1e-15,max_nfev=3000)
        if bst is None or r.cost<bst.cost: bst=r
        if np.sqrt(2*bst.cost)<1e-12: break
    return np.sqrt(2*bst.cost), np.abs(bst.x).max()
if __name__=='__main__':
    for s in [int(x) for x in sys.argv[1:]]:
        for a in [0.3,0.6]:
            out=[]
            for kw in [dict(), dict(merged=True), dict(use_ins=False), dict(use_g2=False)]:
                r,m=best(s,a,**kw); out.append(f'{r:.0e}')
            print(f's={s} a={a}: vector {out[0]} | merged(scalar) {out[1]} | no (f_y)\' {out[2]} | no g2-chain {out[3]}', flush=True)

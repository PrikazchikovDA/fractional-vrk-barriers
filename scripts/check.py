import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
# Brute-force expansion in the paper's framework (mixed generalized Taylor, base point t_n)
import sympy as sp, itertools
al=sp.Symbol('alpha',positive=True)
G=lambda k: sp.gamma(k*al+1)
P=4  # max order in units of alpha
F={(i,j):sp.Symbol(f'F{i}{j}') for i in range(P) for j in range(P) if i+j<P}
u=sp.Symbol('u')
def trunc(e,var,n):
    e=sp.expand(e); return sum(e.coeff(var,k)*var**k for k in range(n+1))
def fexp(tpow_coef, D, var, n):
    # f(t0 + (c h), y0 + D) = sum_{i,j} (c^alpha var)^i/G(i) * D^j/j! * F_ij
    s=0
    for (i,j),Fij in F.items():
        s+= (tpow_coef*var)**i/G(i) * D**j/sp.factorial(j) * Fij
    return trunc(s,var,n)
# exact: y - y0 = sum_k Y_k u^k / G(k)
Y={}
for k in range(1,P+1):
    D=sum(Y[m]*u**m/G(m) for m in Y)
    fe=fexp(1,D,u,k-1)
    Y[k]=sp.expand(G(k-1)*fe.coeff(u,k-1))
exact=sp.expand(sum(Y[k]*u**k/G(k) for k in Y))
def run(s):
    x=sp.symbols(f'x1:{s+1}')            # x_i = c_i^alpha
    a={(i,j):sp.Symbol(f'a{i+1}{j+1}') for i in range(s) for j in range(i)}
    w=sp.symbols(f'w1:{s+1}')
    K=[]
    for i in range(s):
        D=sum(a[(i,j)]*K[j] for j in range(i))
        K.append(trunc(u*fexp(x[i],D,u,P-1),u,P))
    num=sp.expand(sum(w[i]*K[i] for i in range(s)))
    eqs=[]
    for k in range(1,P+1):
        diff=sp.expand(num.coeff(u,k)-exact.coeff(u,k))
        poly=sp.Poly(diff,*F.values())
        for mon,cf in zip(poly.monoms(),poly.coeffs()):
            eqs.append((k,mon,sp.simplify(cf)))
    return x,a,w,eqs
if __name__=='__main__':
    print('Y2 =',Y[2]); print('Y3 =',sp.simplify(Y[3]))
    x,a,w,eqs=run(3)
    for k,m,e in eqs:
        if k<=3: print(k,[str(v) for v,p in zip(F.values(),m) for _ in range(p)],e)

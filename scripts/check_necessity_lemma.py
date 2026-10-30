# Symbolic check of the necessity lemma (R1): linear scalar problem
#   D^a y = (lam + mu (t - t_n)) y + r(t),  exact lag term.
# Local error expansion in H1=h^a, H=h with atoms lam, mu, g_k.
import sympy as sp
s=4
H1,H,lam,mu,al=sp.symbols('H1 H lam mu alpha',positive=True)
g=sp.symbols('g0:5')
c=[0]+list(sp.symbols('c2:%d'%(s+1)))
A=sp.zeros(s,s)
for i in range(1,s):
    for j in range(i):
        A[i,j]=sp.Symbol(f'a{i+1}{j+1}')
b=sp.Matrix(sp.symbols('b1:%d'%(s+1)))
cv=sp.Matrix(c)
K=4
gam=lambda k:sp.factorial(k)/sp.gamma(k+al+1)
# exact stage integrals as symbols w_{k,i}= c_i^{k+al} k!/Gamma -> keep as symbols E_k_i
E={(k,i):(sp.Symbol(f'E{k}_{i+1}') if i>0 else 0) for k in range(K+1) for i in range(s)}
# d_k = A c^k - E_k ; row-sum condition: d_0=0 -> (A 1)_i = E0_i
d={k:sp.Matrix([sum(A[i,j]*cv[j]**k for j in range(s))-E[(k,i)] for i in range(s)]) for k in range(1,K+1)}
D={k:sum(b[j]*cv[j]**k for j in range(s))-gam(k) for k in range(K+1)}
# stage error recursion e = H1*sum_k H^k g_k/k! d_k + H1*A diag(lam+mu H c) e
src=sp.zeros(s,1)
for k in range(1,K+1): src+=H1*H**k*g[k]/sp.factorial(k)*d[k]
M=H1*A*sp.diag(*[lam+mu*H*cj for cj in c])
e=sp.zeros(s,1); term=src
for m in range(s):           # nilpotent: finite Neumann series
    e+=term; term=M*term
delta=H1*sum(H**k*g[k]/sp.factorial(k)*D[k] for k in range(K+1)) \
      + H1*(b.T*sp.diag(*[lam+mu*H*cj for cj in c])*e)[0]
P=sp.Poly(sp.expand(delta),H1,H,lam,mu,*g)
coef=lambda mon: P.coeff_monomial(mon)
Bm=lambda m:(b.T*A**(m-1)*d[1])[0]
C1=(b.T*d[2])[0]; C2=sum(b[i]*cv[i]*d[1][i] for i in range(s))
chk={
 'D0 (H1 g0)':sp.simplify(coef(H1*g[0])-D[0]),
 'D1 (H1 H g1)':sp.simplify(coef(H1*H*g[1])-D[1]),
 'D2/2 (H1 H^2 g2)':sp.simplify(coef(H1*H**2*g[2])-D[2]/2),
 'C1/2 (H1^2 H^2 lam g2)':sp.expand(coef(H1**2*H**2*lam*g[2])-C1/2),
 'C2 (H1^2 H^2 mu g1)':sp.expand(coef(H1**2*H**2*mu*g[1])-C2),
}
for m in range(1,s+1):
    chk[f'B{m} (H1^{m+1} H lam^{m} g1)']=sp.expand(coef(H1**(m+1)*H*lam**m*g[1])-Bm(m))
for k,v in chk.items(): print(f'{k:32s} residual = {v}')
# list every monomial with H-power <=2 and H1-power <=4 to see nothing else sneaks in
print('\nAll monomials with k<=2, m<=4 (H^k H1^m):')
for mon,cf in sorted(P.terms(),key=lambda t:(t[0][1],t[0][0])):
    if mon[1]<=2 and mon[0]<=4:
        names=['H1','H','lam','mu']+[f'g{i}' for i in range(5)]
        print('  ',' '.join(f'{n}^{p}' for n,p in zip(names,mon) if p))

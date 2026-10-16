import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
# Paper's 2-stage EFORK with base-point modification F_n, test y=t^2 and y=E_alpha(-t^alpha)-like smooth problem
import numpy as np
from scipy.special import gamma as G
def run(alpha,N,f,y0,T=1.0,stages=2):
    h=T/N; t=np.linspace(0,T,N+1); y=np.zeros(N+1); y[0]=y0
    g1,g2=G(alpha+1),G(2*alpha+1)
    c2=(2*g1**2/g2)**(1/alpha); a21=c2**alpha/g1; w1=w2=1/(2*g1)
    F0=[]
    for n in range(N):
        def F(tt,yy):
            s=0.0
            if n>0:
                i=np.arange(n); d=(y[1:n+1]-y[:n])/h
                s=np.sum(d*((tt-t[i])**(1-alpha)-np.maximum(tt-t[i+1],0)**(1-alpha)))/G(2-alpha)
            return f(tt,yy)-s
        K1=h**alpha*F(t[n],y[n]); K2=h**alpha*F(t[n]+c2*h,y[n]+a21*K1)
        F0.append(F(t[n],y[n])/max(abs(f(t[n],y[n])),1e-300))
        y[n+1]=y[n]+w1*K1+w2*K2
    return y,np.array(F0)
for alpha in [0.5,0.8]:
    f=lambda t,y,a=alpha: 2*t**(2-a)/G(3-a)-y+t**2   # exact y=t^2
    print(f'alpha={alpha}, exact y=t^2')
    prev=None
    for N in [20,40,80,160,320,640]:
        y,F0=run(alpha,N,f,0.0)
        e=abs(y[-1]-1.0); print(f'  N={N:4d} err={e:.3e}', f' rate={np.log2(prev/e):.2f}' if prev else '', f' |F_n(t_n,y_n)|/|f| at end={abs(F0[-1]):.2e}'); prev=e
print('--- Example 1 of the paper: D^a y=-y+t^(4-a)/G(5-a), y=t^4 E_{a,5}(-t^a)')
def ML(z,a,b,K=200): return sum(z**k/G(a*k+b) for k in range(K))
for alpha in [0.5,1/3,0.7,0.8,0.9]:
    f=lambda t,y,a=alpha: -y+t**(4-a)/G(5-a); ex=ML(-1.0,alpha,5)
    prev=None; s=f'alpha={alpha:.3f}:'
    for N in [40,80,160,320,640]:
        y,_=run(alpha,N,f,0.0); e=abs(y[-1]-ex)
        s+=f' N={N} err={e:.2e}'+(f'(r={np.log2(prev/e):.2f})' if prev else ''); prev=e
    print(s)

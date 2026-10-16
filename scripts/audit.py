import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
# Audit of EFORK (Ghaffari et al. 2023): local consistency with exact history, long-run global convergence, 3-stage
import numpy as np, sys
from scipy.special import gamma as G
def ML(z,a,b,K=300): return sum(z**k/G(a*k+b) for k in range(K))
def tab2(a):
    g1,g2=G(a+1),G(2*a+1); c2=(2*g1**2/g2)**(1/a); return [0,c2],{(1,0):c2**a/g1},[1/(2*g1)]*2
def tab3(a,c2=0.5,c3=1.0):
    g1,g2,g3=G(a+1),G(2*a+1),G(3*a+1); x2,x3=c2**a,c3**a
    M=np.array([[1,1,1],[0,x2,x3],[0,x2**2,x3**2]]); w=np.linalg.solve(M,[1/g1,g1/g2,g2/g3])
    a21=x2/g1; a32=g1/(g3*w[2]*x2); a31=x3/g1-a32
    return [0,c2,c3],{(1,0):a21,(2,0):a31,(2,1):a32},list(w)
def Fmem(tt,n,t,y,h,a):
    if n==0: return 0.0
    i=np.arange(n); d=(y[1:n+1]-y[:n])/h
    return np.sum(d*((tt-t[i])**(1-a)-np.maximum(tt-t[i+1],0)**(1-a)))/G(2-a)
def step(n,t,y,h,a,f,tab):
    c,A,w=tab; K=[]
    for i in range(len(c)):
        Y=y[n]+sum(A.get((i,j),0)*K[j] for j in range(i)); tt=t[n]+c[i]*h
        K.append(h**a*(f(tt,Y)-Fmem(tt,n,t,y,h,a)))
    return y[n]+sum(wi*Ki for wi,Ki in zip(w,K)), K
def glob(a,N,f,ex,tab):
    h=1/N; t=np.linspace(0,1,N+1); y=np.zeros(N+1)
    for n in range(N): y[n+1],_=step(n,t,y,h,a,f,tab)
    return abs(y[-1]-ex)
mode=sys.argv[1]
alphas=[0.25,1/3,0.4,0.5,0.6,0.7,0.8,0.9]
if mode=='local':
    # exact history on grid, single step from t* = 0.5, compare with exact y(t*+h)
    print('LOCAL error of one step from t=0.5 with EXACT history (y=t^4 E_{a,5}(-t^a))')
    for a in alphas:
        f=lambda t,y,a=a: -y+t**(4-a)/G(5-a); yex=lambda s,a=a: s**4*ML(-s**a,a,5)
        s=f'a={a:.3f}:'; prev=None
        for N in [20,40,80,160,320]:
            h=0.5/N*1.0; t=np.arange(N+1)*h; y=np.array([yex(ti) for ti in t]); tt=np.append(t,t[-1]+h); yy=np.append(y,0)
            y1,K=step(N,tt,yy,h,a,f,tab2(a)); e=abs(y1-yex(tt[-1]))
            s+=f' h={h:.4f}:{e:.2e}'+(f'(p={np.log2(prev/e):.2f})' if prev else ''); prev=e
        print(s, f' |K1|/h={abs(K[0])/h:.1e}')
elif mode=='global':
    tabf={'2':tab2,'3':tab3}[sys.argv[2]]
    print(f'GLOBAL error at T=1, {sys.argv[2]}-stage EFORK, Example 1')
    for a in alphas:
        f=lambda t,y,a=a: -y+t**(4-a)/G(5-a); ex=ML(-1.0,a,5); s=f'a={a:.3f}:'; prev=None
        for N in [40,160,640,2560]:
            e=glob(a,N,f,ex,tabf(a)); s+=f' N={N}:{e:.2e}'+(f'(p/2={np.log(prev/e)/np.log(4):.2f})' if prev else ''); prev=e
        print(s)

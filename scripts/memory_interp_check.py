import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as G
for a in (0.3,0.7):
  y=lambda s: 1+s**a+s**2; yp=lambda s: a*s**(a-1)+2*s
  tn=0.5; errs=[]
  for N in (20,40,80,160,320):
    h=tn/N; t=tn+0.5*h
    ex=quad(lambda s: yp(s),0,tn,weight='alg',wvar=(0,-a),epsabs=1e-15,limit=400)[0]
    # weight alg (s-0)^0 (tn-s)^-a ; need (t-s)^-a: use shift
    ex=quad(lambda u: yp(t-u),t-tn,t,weight=None,limit=400,points=None)[0] if False else None
    f=lambda s:(t-s)**(-a)*yp(s)
    ex=sum(quad(f,k*h,(k+1)*h,epsabs=1e-16,epsrel=1e-13)[0] for k in range(N))
    ap=0.0
    for k in range(N):
      sl=(y((k+1)*h)-y(k*h))/h
      ap+=sl*((t-k*h)**(1-a)-(t-(k+1)*h)**(1-a))/(1-a)
    errs.append(abs(ex-ap)/G(1-a))
  print(a,[round(np.log2(errs[i]/errs[i+1]),2) for i in range(4)],'predicted',min(1+a,2-a))

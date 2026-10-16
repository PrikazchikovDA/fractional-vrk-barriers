import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
import sys, json, time, numpy as np
from fractions import Fraction
from multiprocessing import Pool

from vrk_fast import exists
def exps(alpha, pmax):
    a = Fraction(alpha).limit_denominator(1000); E = set()
    for m in range(1, 60):
        for k in range(0, int(pmax)+1):
            e = m*a + k
            if e <= pmax: E.add(e)
    return sorted(E)
def job(args):
    alpha, s, pmax = args; t0 = time.time(); best = None; log = []
    for E in exps(alpha, pmax):
        ok, res, _ = exists(s, alpha, float(E), tries=12)
        log.append((float(E), bool(ok), float(res)))
        if not ok: break
        best = float(E)
    err_exp = log[-1][0] if not log[-1][1] else None
    return dict(alpha=alpha, s=s, pmax_found=best, err_exp=err_exp, log=log, sec=time.time()-t0)
if __name__ == '__main__':
    alphas = [0.3,0.4]
    jobs = [(a, s, {1: 2.0, 2: 3.0, 3: 3.5, 4: 4.6}[s]) for s in (4,) for a in alphas]
    jobs.sort(key=lambda j: j[0])
    with Pool(int(sys.argv[1])) as P:
        out = []
        for r in P.imap_unordered(job, jobs):
            out.append(r); print(json.dumps({k: r[k] for k in ("alpha","s","pmax_found","err_exp","sec")}), flush=True)
            json.dump(out, open(_os.path.join(DATA,'map4b.json'),'w'))
